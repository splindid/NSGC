import torch
import torch.nn as nn
import torch.nn.functional as F


def hand_crafted_feature(sl, bl, eps=1e-6):
    """
    Build geometry features from two box tensors in xyxy format.
    """
    if sl is None or bl is None:
        raise ValueError("hand_crafted_feature expects both sl and bl tensors")
    
    sl = sl[:, :4].float()
    bl = bl[:, :4].float()

    sx1, sy1, sx2, sy2 = sl.unbind(dim=-1)
    bx1, by1, bx2, by2 = bl.unbind(dim=-1)

    sw = (sx2 - sx1).clamp(min=eps)
    sh = (sy2 - sy1).clamp(min=eps)
    bw = (bx2 - bx1).clamp(min=eps)
    bh = (by2 - by1).clamp(min=eps)

    scx = (sx1 + sx2) * 0.5
    scy = (sy1 + sy2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5

    dx = scx - bcx
    dy = scy - bcy
    dist = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)

    lt_x = torch.maximum(sx1, bx1)
    lt_y = torch.maximum(sy1, by1)
    rb_x = torch.minimum(sx2, bx2)
    rb_y = torch.minimum(sy2, by2)
    inter_w = (rb_x - lt_x).clamp(min=0.0)
    inter_h = (rb_y - lt_y).clamp(min=0.0)
    inter = inter_w * inter_h
    union = sw * sh + bw * bh - inter + eps
    iou = inter / union

    sub_in_obj = ((sx1 >= bx1) & (sy1 >= by1) & (sx2 <= bx2) & (sy2 <= by2)).float()
    obj_in_sub = ((bx1 >= sx1) & (by1 >= sy1) & (bx2 <= sx2) & (by2 <= sy2)).float()

    features = torch.stack([
        dx / sw, dy / sh,
        dx / bw, dy / bh,
        torch.log(sw), torch.log(sh),
        torch.log(bw), torch.log(bh),
        torch.log(sw / bw), torch.log(sh / bh),
        torch.log((sw * sh) / (bw * bh)),
        iou,
        sub_in_obj,
        obj_in_sub,
        dist / torch.sqrt(sw * sh + eps),
        dist / torch.sqrt(bw * bh + eps),
        torch.cos(torch.atan2(dy, dx)),
        torch.sin(torch.atan2(dy, dx)),
        (sw * sh + eps) / (union + eps),
    ], dim=-1)
    
    return features


class InteractionModule(nn.Module):

    def __init__(self, in_dim, out_dim=None, dropout=0.2):
        super(InteractionModule, self).__init__()
        if out_dim is None:
            out_dim = in_dim // 2
        self.lin = nn.Linear(in_dim, out_dim)
        self.quad = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # r = GELU(LayerNorm(xW1 + (x^2)W2))
        out = self.lin(x) + self.quad(x**2)
        return self.dropout(self.act(self.norm(out)))


class PairwiseBiasResidualCorrector(nn.Module):
    preserve_background = False

    def __init__(
        self,
        ds_obj,
        cfg,
        hidden_dim=1024,
        dropout=0.2,
    ):
        super(PairwiseBiasResidualCorrector, self).__init__()
        self.device = ds_obj.device
        self.num_rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.obj_embed = nn.Embedding(self.num_obj_classes, hidden_dim)
        self.geo_fc = nn.Linear(19, hidden_dim)


        self.interact_sb = InteractionModule(hidden_dim * 2, out_dim=hidden_dim, dropout=dropout)
        self.interact_ob = InteractionModule(hidden_dim * 2, out_dim=hidden_dim, dropout=dropout)
        self.interact_so = InteractionModule(hidden_dim * 2, out_dim=hidden_dim, dropout=dropout)
        

        self.r1_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, self.num_rel_classes)
        )

        self.r2_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_rel_classes)
        )


        self.r_prime_param = nn.Parameter(torch.zeros(self.num_rel_classes))
        self.residual_scale = nn.Parameter(torch.ones(self.num_rel_classes) * 0.1)

        nn.init.constant_(self.r1_mlp[-1].weight, 0)
        nn.init.constant_(self.r1_mlp[-1].bias, 0)
        nn.init.constant_(self.r2_mlp[-1].weight, 0)
        nn.init.constant_(self.r2_mlp[-1].bias, 0)

        self.to(self.device)

    def forward(self, rel_logits=None, sub_labels=None, obj_labels=None, sl_boxes=None, bl_boxes=None, **kwargs):
        if rel_logits is None: return None
        
        rel_logits = rel_logits.to(self.device)
        sub_labels = sub_labels.long().to(self.device)
        obj_labels = obj_labels.long().to(self.device)
        sl_boxes = sl_boxes.to(self.device)
        bl_boxes = bl_boxes.to(self.device)


        s_emb = self.obj_embed(sub_labels)
        o_emb = self.obj_embed(obj_labels)
        b_raw = hand_crafted_feature(sl_boxes, bl_boxes)
        b_emb = self.geo_fc(b_raw)
        
        feat_sb = self.interact_sb(torch.cat([s_emb, b_emb], dim=-1))
        feat_ob = self.interact_ob(torch.cat([o_emb, b_emb], dim=-1))
        feat_so = self.interact_so(torch.cat([s_emb, o_emb], dim=-1))
        
        r1 = self.r1_mlp(torch.cat([feat_sb, feat_ob, feat_so], dim=-1))
        

        r2 = self.r2_mlp(torch.cat([s_emb, o_emb], dim=-1))

        delta = r1 - r2 + self.r_prime_param
        scaled_delta = delta * self.residual_scale


        corrected_logits = rel_logits.clone()
        corrected_logits[:, 1:] = corrected_logits[:, 1:] + scaled_delta[:, 1:]

        return {
            "logits": corrected_logits,
            "r1": r1,
            "r2": r2,
            "delta": delta,
            "base_logits": rel_logits,
        }

    def correct(self, *args, **kwargs):
        res = self.forward(*args, **kwargs)
        return res["logits"] if res is not None else None
