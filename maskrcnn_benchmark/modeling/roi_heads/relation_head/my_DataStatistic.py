import torch
import torch.nn.functional as F
import os
import pickle
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from .CorrectionAlgorithm_V1 import MorphingCorrectorV1
from .CorrectionAlgorithm_BiasResidual import PairwiseBiasResidualCorrector
from tqdm import tqdm

class DataStatistic(object):
    """
    优化后的 DataStatistic：支持 GPU 加速、特征缓存和排除背景类。
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE) # 使用 GPU
        self.num_rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        # 1. 加载统计数据
        statistics_path = "output/baseline/VG_stanford_filtered_with_attribute_train_statistics.cache"
        if os.path.exists(statistics_path):
            try:
                self.statistic = torch.load(statistics_path, map_location=self.device)
                print(f"DataStatistic: Loaded statistics to {self.device}")
            except Exception as e:
                print(f"DataStatistic: Error loading statistics: {e}")
                self.statistic = {}
        else:
            self.statistic = {}
            print(f"DataStatistic: Warning, statistics file not found.")

        # 缓存路径
        self.cache_path = os.path.join(cfg.OUTPUT_DIR, "geo_features_cache.pt")

        if 'fg_matrix' in self.statistic:
            counts = self.statistic['fg_matrix'].sum(dim=(0, 1)).float()
            self.p_k = counts / (counts.sum() + 1e-12)
            
            co_occurrence = self.statistic['fg_matrix'].float()
            rel_vectors = co_occurrence.view(-1, 51).T # [51, 151*151]
            norm = rel_vectors.norm(dim=1, keepdim=True) + 1e-12
            self.omega = (rel_vectors @ rel_vectors.T) / (norm @ norm.T)
            self.omega = F.softmax(self.omega * 5.0, dim=-1)
            
            print("DataStatistic: Theoretical priors (P(k), Omega) initialized.")
        else:
            self.p_k = torch.ones(51).to(self.device) / 51.0
            self.omega = torch.eye(51).to(self.device)
            print("DataStatistic: Warning, using uniform priors (fg_matrix missing).")

        corrector_name = os.environ.get("SGG_CORRECTOR", "MorphingCorrectorV1")
        if corrector_name == "PairwiseBiasResidualCorrector":
            self.corrector = PairwiseBiasResidualCorrector(self, cfg)
        else:
            self.corrector = MorphingCorrectorV1(self, cfg)

        default_weight_path = "output/hb3c_weights_final.pth"
        if corrector_name == "PairwiseBiasResidualCorrector":
            default_weight_path = "output/pairwise_bias_residual_best.pth"
        hb3c_weight_path = os.environ.get("SGG_CORRECTOR_WEIGHTS", default_weight_path)
        self.global_mean = None
        self.global_std = None
        
        if os.path.exists(hb3c_weight_path):
            print(f"DataStatistic: Loading trained HB3C weights from {hb3c_weight_path}")
            try:
                checkpoint = torch.load(hb3c_weight_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.corrector.load_state_dict(checkpoint['state_dict'])
                    self.global_mean = checkpoint['global_mean'].to(self.device)
                    self.global_std = checkpoint['global_std'].to(self.device)
                    print("DataStatistic: Successfully loaded weights and global statistics.")
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.corrector.load_state_dict(checkpoint['model_state_dict'])
                    print("DataStatistic: Loaded model_state_dict checkpoint.")
                else:
                    # 兼容旧版本仅保存 state_dict 的情况
                    self.corrector.load_state_dict(checkpoint)
                    print("DataStatistic: Loaded weights (old format, no global stats).")
            except Exception as e:
                print(f"DataStatistic: Error loading weights: {e}")
        # ----------------------------------

    def extract_geometric_features(self, boxes_sub, boxes_obj):
        """
        在 GPU 上计算几何特征
        """
        boxes_sub = boxes_sub.to(self.device)
        boxes_obj = boxes_obj.to(self.device)

        with torch.no_grad():
            w_sub = boxes_sub[:, 2] - boxes_sub[:, 0] + 1e-6
            h_sub = boxes_sub[:, 3] - boxes_sub[:, 1] + 1e-6
            w_obj = boxes_obj[:, 2] - boxes_obj[:, 0] + 1e-6
            h_obj = boxes_obj[:, 3] - boxes_obj[:, 1] + 1e-6

            center_sub = torch.stack([(boxes_sub[:, 0] + boxes_sub[:, 2]) / 2, (boxes_sub[:, 1] + boxes_sub[:, 3]) / 2], dim=1)
            center_obj = torch.stack([(boxes_obj[:, 0] + boxes_obj[:, 2]) / 2, (boxes_obj[:, 1] + boxes_obj[:, 3]) / 2], dim=1)

            rel_center = (center_sub - center_obj) / torch.stack([w_sub, h_sub], dim=1)

            v_sub = torch.stack([boxes_sub[:, 0], boxes_sub[:, 1], boxes_sub[:, 2], boxes_sub[:, 1], 
                                 boxes_sub[:, 0], boxes_sub[:, 3], boxes_sub[:, 2], boxes_sub[:, 3]], dim=1)
            v_obj = torch.stack([boxes_obj[:, 0], boxes_obj[:, 1], boxes_obj[:, 2], boxes_obj[:, 1], 
                                 boxes_obj[:, 0], boxes_obj[:, 3], boxes_obj[:, 2], boxes_obj[:, 3]], dim=1)


            lt = torch.max(boxes_sub[:, :2], boxes_obj[:, :2])
            rb = torch.min(boxes_sub[:, 2:], boxes_obj[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, 0] * wh[:, 1]
            iou = inter / (w_sub * h_sub + w_obj * h_obj - inter + 1e-6)

            sub_in_obj = (boxes_sub[:, 0] >= boxes_obj[:, 0]) & (boxes_sub[:, 1] >= boxes_obj[:, 1]) & \
                         (boxes_sub[:, 2] <= boxes_obj[:, 2]) & (boxes_sub[:, 3] <= boxes_obj[:, 3])
            obj_in_sub = (boxes_obj[:, 0] >= boxes_sub[:, 0]) & (boxes_obj[:, 1] >= boxes_sub[:, 1]) & \
                         (boxes_obj[:, 2] <= boxes_sub[:, 2]) & (boxes_obj[:, 3] <= boxes_sub[:, 3])
            inclusion = torch.stack([sub_in_obj.float(), obj_in_sub.float()], dim=1)

        return {
            "rel_center": rel_center,
            "iou": iou.unsqueeze(1),
            "inclusion": inclusion,
            "v_sub": v_sub,
            "v_obj": v_obj,
            "rel_wh": torch.stack([w_sub/w_obj, h_sub/h_obj], dim=1)
        }

    def correction(self, refine_logits, relation_logits, rel_pair_idxs, proposals, roi_features=None, union_features=None):
        if self.corrector is None:
            return relation_logits

        if refine_logits is not None:

            if isinstance(refine_logits, (list, tuple)) and len(refine_logits) > 0 and isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits = refine_logits[0]
            else:
                refine_obj_logits = refine_logits
            
            if isinstance(refine_obj_logits, (list, tuple)):
                refine_logits_tensor = cat(refine_obj_logits, dim=0)
            else:
                refine_logits_tensor = refine_obj_logits
                
            if refine_logits_tensor.dim() > 1 and refine_logits_tensor.size(1) > 1:
                obj_labels = refine_logits_tensor.max(1)[1]
            else:
                obj_labels = refine_logits_tensor.squeeze()
        else:
  
            obj_labels = torch.cat([p.get_field("pred_labels") for p in proposals]).to(self.device)
        

        is_list = isinstance(relation_logits, (list, tuple))
        if is_list:
            rel_probs_tensor = cat(relation_logits, dim=0).to(self.device)
        else:
            rel_probs_tensor = relation_logits.to(self.device)
        
        if rel_probs_tensor.min() >= 0 and rel_probs_tensor.max() <= 1.0001:
            rel_logits_tensor = torch.log(rel_probs_tensor + 1e-12)
        else:
            rel_logits_tensor = rel_probs_tensor


        sub_labels_list, obj_labels_list = [], []
        boxes_sub_list, boxes_obj_list = [], []
        offset = 0
        for i, p in enumerate(proposals):
            num_objs = len(p)
            rel_idx = rel_pair_idxs[i]
            boxes = p.bbox
            sub_labels_list.append(obj_labels[offset + rel_idx[:, 0]])
            obj_labels_list.append(obj_labels[offset + rel_idx[:, 1]])
            boxes_sub_list.append(boxes[rel_idx[:, 0]])
            boxes_obj_list.append(boxes[rel_idx[:, 1]])
            offset += num_objs
        
        all_sub_labels = torch.cat(sub_labels_list, dim=0).to(self.device)
        all_obj_labels = torch.cat(obj_labels_list, dim=0).to(self.device)
        all_boxes_sub = torch.cat(boxes_sub_list, dim=0).to(self.device)
        all_boxes_obj = torch.cat(boxes_obj_list, dim=0).to(self.device)

        geo_features = self.extract_geometric_features(all_boxes_sub, all_boxes_obj)
        

        all_rel_pair_idxs = None
        if isinstance(rel_pair_idxs, (list, tuple)):
            all_rel_pair_idxs_list = []
            curr_offset = 0
            for i, p in enumerate(proposals):
                all_rel_pair_idxs_list.append(rel_pair_idxs[i] + curr_offset)
                curr_offset += len(p)
            all_rel_pair_idxs = torch.cat(all_rel_pair_idxs_list, dim=0).to(self.device)
        else:
            all_rel_pair_idxs = rel_pair_idxs.to(self.device) if rel_pair_idxs is not None else None
            
        corrected_logits = self.corrector(
            rel_logits_tensor, 
            sub_labels=all_sub_labels, 
            obj_labels=all_obj_labels,
            geo_features=geo_features,
            sl_boxes=all_boxes_sub,
            bl_boxes=all_boxes_obj,
        )

        if isinstance(corrected_logits, dict):
            corrected_logits = corrected_logits["logits"]

        if not getattr(self.corrector, "preserve_background", True):
            if is_list:
                num_rels = [r.shape[0] for r in rel_pair_idxs]
                return corrected_logits.split(num_rels, dim=0)
            return corrected_logits
    
        orig_probs = torch.softmax(rel_logits_tensor, dim=-1)
        orig_bg_prob = orig_probs[:, 0:1]
        
        corrected_probs = torch.softmax(corrected_logits, dim=-1)
        fg_probs_new = corrected_probs[:, 1:]
        fg_sum = fg_probs_new.sum(dim=-1, keepdim=True) + 1e-12
        fg_probs_final = fg_probs_new * ((1.0 - orig_bg_prob) / fg_sum)
        
        final_probs = torch.cat([orig_bg_prob, fg_probs_final], dim=-1)
        if is_list:
            num_rels = [r.shape[0] for r in rel_pair_idxs]
            return final_probs.split(num_rels, dim=0)
        
        return final_probs
