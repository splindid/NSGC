import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import sys
from maskrcnn_benchmark.modeling.roi_heads.relation_head.CorrectionAlgorithm_BiasResidual import PairwiseBiasResidualCorrector
from maskrcnn_benchmark.config import cfg

class CachedRelationDataset(Dataset):
    def __init__(self, cache_path):
        print(f"Loading cache from {cache_path}...")

        try:
            self.data = torch.load(cache_path, map_location='cpu', weights_only=False)
        except TypeError:

            self.data = torch.load(cache_path, map_location='cpu')
        

        self.predictions = self.data['predictions']
        self.groundtruths = self.data['groundtruths']
        

        self.flat_entries = []
        for i, (pred, gt) in tqdm(enumerate(zip(self.predictions, self.groundtruths)), total=len(self.predictions), desc="Processing data"):

            try:
                rel_logits = pred.get_field("pred_rel_scores")
            except KeyError:

                available_fields = pred.fields()
                if 'relation_logits' in available_fields:
                    rel_logits = pred.get_field("relation_logits")
                elif 'rel_scores' in available_fields:
                    rel_logits = pred.get_field("rel_scores")
                else:
                    raise KeyError(f"Could not find relation scores in BoxList fields: {available_fields}")

           
            gt_tuples = gt.get_field("relation_tuple") # [M, 3] usually (sub, obj, label)
            rel_pair_idxs = pred.get_field("rel_pair_idxs") # [N_pairs, 2]
            
            true_labels = torch.zeros(rel_pair_idxs.size(0), dtype=torch.long)
            
           
            if gt_tuples.size(0) > 0:
                for gt_i in range(gt_tuples.size(0)):
                    sub_gt, obj_gt, label_gt = gt_tuples[gt_i]
                    match_idx = (rel_pair_idxs[:, 0] == sub_gt) & (rel_pair_idxs[:, 1] == obj_gt)
                    true_labels[match_idx] = label_gt
            
            rel_labels = true_labels
            
            obj_labels = pred.get_field("labels")
            obj_boxes = pred.bbox # [M_obj, 4]

            sub_labels = obj_labels[rel_pair_idxs[:, 0]]
            obj_labels_rel = obj_labels[rel_pair_idxs[:, 1]]
            sl_boxes = obj_boxes[rel_pair_idxs[:, 0]]
            bl_boxes = obj_boxes[rel_pair_idxs[:, 1]]
            

            self.flat_entries.append({
                'rel_logits': rel_logits,
                'rel_labels': rel_labels,
                'sub_labels': sub_labels,
                'obj_labels': obj_labels_rel,
                'sl_boxes': sl_boxes,
                'bl_boxes': bl_boxes
            })
    def __len__(self):
        return len(self.flat_entries)
    def __getitem__(self, idx):
        return self.flat_entries[idx]
def train():

    cache_file = "/root/sgg_project/output/motif_eval_train/inference/VG_stanford_filtered_with_attribute_train/eval_results.pytorch"
    config_file = "/root/sgg_project/MotifPredictor/PredCls/config.yml"
    output_model_path = "/root/sgg_project/output/pairwise_bias_residual_v2_best.pth"
    num_epochs = 20
    

    batch_size = 32 
    learning_rate = 1e-4

    if not os.path.exists(cache_file):
        print(f"Error: Cache file not found: {cache_file}")
        print("Please ensure you have generated the train set cache first by running inference on the train set.")
        return

    cfg.merge_from_file(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = CachedRelationDataset(cache_file)
    
    
    def my_collate(batch):
        return {
            'rel_logits': torch.cat([b['rel_logits'] for b in batch], dim=0),
            'rel_labels': torch.cat([b['rel_labels'] for b in batch], dim=0),
            'sub_labels': torch.cat([b['sub_labels'] for b in batch], dim=0),
            'obj_labels': torch.cat([b['obj_labels'] for b in batch], dim=0),
            'sl_boxes': torch.cat([b['sl_boxes'] for b in batch], dim=0),
            'bl_boxes': torch.cat([b['bl_boxes'] for b in batch], dim=0),
        }
        
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=my_collate,
        pin_memory=True,   
        num_workers=0      
    )


    class DummyDS:
        def __init__(self, device): self.device = device
    
    model = PairwiseBiasResidualCorrector(DummyDS(device), cfg)
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

   
    print("Calculating class weights for balanced Loss...")
    all_gt_labels = torch.cat([b['rel_labels'] for b in dataset.flat_entries])
    class_counts = torch.bincount(all_gt_labels, minlength=cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)
    
   
    class_counts = class_counts.float()
    class_counts[class_counts == 0] = 1.0  
    
   
    eps=1e-6
    alpha=2
    weights = 1.0 / class_counts
    
   
    weights[0] = weights.mean() * 0.01 
    
   
    class_weights_tensor = weights.to(device)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * len(class_weights_tensor)
    
  
    class JointHarmonicLoss(nn.Module):
        def __init__(self, balanced_weights):
            super().__init__()
            self.ce_standard = nn.CrossEntropyLoss()
            self.ce_balanced = nn.CrossEntropyLoss(weight=balanced_weights)
            
        def forward(self, logits, targets):
            l_r = self.ce_standard(logits, targets)
            l_mr = self.ce_balanced(logits, targets)
            joint_loss = torch.abs(l_r) + torch.abs(l_mr)
            return joint_loss, l_r, l_mr

    criterion = JointHarmonicLoss(class_weights_tensor)

    print(f"Starting training on {device} with (1-R)*(1-mR) custom loss...")
    best_loss = float('inf')

    
    scaler = torch.cuda.amp.GradScaler() 

    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            
            rel_logits = batch['rel_logits'].to(device, non_blocking=True)
            rel_labels = batch['rel_labels'].to(device, non_blocking=True).long()
            sub_labels = batch['sub_labels'].to(device, non_blocking=True)
            obj_labels = batch['obj_labels'].to(device, non_blocking=True)
            sl_boxes = batch['sl_boxes'].to(device, non_blocking=True)
            bl_boxes = batch['bl_boxes'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = model(
                    rel_logits=rel_logits,
                    sub_labels=sub_labels,
                    obj_labels=obj_labels,
                    sl_boxes=sl_boxes,
                    bl_boxes=bl_boxes
                )
                
                corrected_logits = outputs['logits']
                fg_mask = rel_labels > 0
                if fg_mask.sum() == 0:
                    continue
                    
                loss, loss_r, loss_mr = criterion(corrected_logits[fg_mask], rel_labels[fg_mask])
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'l_R': f"{loss_r.item():.2f}", 'l_mR': f"{loss_mr.item():.2f}", 'loss': f"{loss.item():.2f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.1e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            output_dir = os.path.dirname(output_model_path)
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            torch.save(model.state_dict(), output_model_path)
            print(f"Saved best model to {output_model_path}")

if __name__ == "__main__":
    train()