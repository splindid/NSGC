import torch
import os
import argparse
import copy
from tqdm import tqdm
import torch.nn.functional as F
import json

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation.vg import vg_evaluation
from maskrcnn_benchmark.modeling.roi_heads.relation_head.CorrectionAlgorithm_BiasResidual import PairwiseBiasResidualCorrector
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.structures.bounding_box import BoxList

class DummyDataset:
    def __init__(self, groundtruths, ind_to_classes, ind_to_predicates):
        self.groundtruths = groundtruths
        self.ind_to_classes = ind_to_classes
        self.ind_to_predicates = ind_to_predicates
        self.filenames = [f"image_{i}.jpg" for i in range(len(groundtruths))]
        self.categories = {i: name for i, name in enumerate(ind_to_classes)}

    def get_img_info(self, image_id):
        gt = self.groundtruths[image_id]
        return {"width": gt.size[0], "height": gt.size[1]}

    def get_groundtruth(self, image_id, evaluation=True):
        return self.groundtruths[image_id]

class DummyDS:
    def __init__(self, device): 
        self.device = device

def apply_corrector_to_predictions(predictions, model, device='cpu', batch_size=8192):
    model.eval()
    print("Collecting all relations from cache...")
    
    all_rel_logits = []
    all_sub_labels = []
    all_obj_labels = []
    all_sub_boxes = []
    all_obj_boxes = []
    rel_counts = []
    
    for img_idx, pred in enumerate(predictions):
        rel_pair_idxs = pred.get_field('rel_pair_idxs')
        n_rels = len(rel_pair_idxs)
        rel_counts.append(n_rels)
        
        if n_rels == 0:
            continue
            
        try:
            rel_logits = pred.get_field("pred_rel_scores")
        except KeyError:
            available_fields = pred.fields()
            if 'relation_logits' in available_fields:
                rel_logits = pred.get_field("relation_logits")
            elif 'rel_scores' in available_fields:
                rel_logits = pred.get_field("rel_scores")
            else:
                raise KeyError("Could not find relation scores")
                
        pred_labels = pred.get_field('labels')
        if pred_labels is None:
            pred_labels = pred.get_field('pred_labels')
            
        boxes = pred.bbox
        sub_idx = rel_pair_idxs[:, 0]
        obj_idx = rel_pair_idxs[:, 1]
        
        all_rel_logits.append(rel_logits)
        all_sub_labels.append(pred_labels[sub_idx])
        all_obj_labels.append(pred_labels[obj_idx])
        all_sub_boxes.append(boxes[sub_idx])
        all_obj_boxes.append(boxes[obj_idx])
        
    if not all_rel_logits:
        return [copy.deepcopy(p) for p in predictions]

    # Concatenate all data
    all_rel_logits = torch.cat(all_rel_logits, dim=0).to(device)
    all_sub_labels = torch.cat(all_sub_labels, dim=0).to(device).long()
    all_obj_labels = torch.cat(all_obj_labels, dim=0).to(device).long()
    all_sub_boxes = torch.cat(all_sub_boxes, dim=0).to(device)
    all_obj_boxes = torch.cat(all_obj_boxes, dim=0).to(device)
    
    total_rels = len(all_rel_logits)
    print(f"Total pairs to process: {total_rels}")
    
    corrected_logits_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, total_rels, batch_size), desc="Correcting Logits"):
            end = min(i + batch_size, total_rels)
            out = model(
                rel_logits=all_rel_logits[i:end],
                sub_labels=all_sub_labels[i:end],
                obj_labels=all_obj_labels[i:end],
                sl_boxes=all_sub_boxes[i:end],
                bl_boxes=all_obj_boxes[i:end]
            )
            # bias residual returns logits. 
            # we should apply softmax so vg_evaluation sorts by probabilities
            probs = F.softmax(out["logits"], dim=-1)
            corrected_logits_list.append(probs.cpu())
            
    corrected_probs = torch.cat(corrected_logits_list, dim=0)

    print("Re-assembling and mapping to original predictions...")
    corrected_predictions = []
    rel_ptr = 0
    
    for img_idx, pred in enumerate(predictions):
        n_rels = rel_counts[img_idx]
        
        # deepcopy structures
        new_pred = BoxList(pred.bbox, pred.size, pred.mode)
        
        for k, v in pred.extra_fields.items():
            if k not in ['pred_rel_scores', 'pred_rel_labels', 'relation_logits']:
                new_pred.add_field(k, v)
                if k in pred.triplet_extra_fields:
                    new_pred.triplet_extra_fields.append(k)
                    
        if n_rels > 0:
            probs = corrected_probs[rel_ptr:rel_ptr + n_rels]
            new_pred.add_field('pred_rel_scores', probs, is_triplet=True)
            
            # optional: save actual classes
            pred_rel_labels = probs[:, 1:].argmax(dim=1) + 1
            new_pred.add_field('pred_rel_labels', pred_rel_labels, is_triplet=True)
            rel_ptr += n_rels
            
        corrected_predictions.append(new_pred)
        
    return corrected_predictions


def main():
    config_file = "/root/sgg_project/MotifPredictor/PredCls/config.yml"
    cache_file = "/root/sgg_project/output/motif_eval_test/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch"
    model_weights = "/root/sgg_project/output/pairwise_bias_residual_v2_best.pth"
    output_dir = "/root/sgg_project/output/bias_residual_v2_eval"

    cfg.merge_from_file(config_file)
    cfg.freeze()

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Initializing offline evaluation...")

    # Load Dicts
    with open('datasets/vg/VG-SGG-dicts-with-attri.json', 'r') as f:
        vg_dict = json.load(f)
        
    ind_to_classes = ['__background__'] + [vg_dict['idx_to_label'].get(str(i), 'unknown') for i in range(1, max([int(k) for k in vg_dict['idx_to_label'].keys()]) + 1)]
    ind_to_predicates = ['__background__'] + [vg_dict['idx_to_predicate'].get(str(i), 'unknown') for i in range(1, max([int(k) for k in vg_dict['idx_to_predicate'].keys()]) + 1)]

    # Load cache
    logger.info(f"Loading cached Test set predictions: {cache_file}...")
    try:
        cache = torch.load(cache_file, map_location='cpu', weights_only=False)
    except TypeError:
        cache = torch.load(cache_file, map_location='cpu')
        
    predictions = cache['predictions']
    groundtruths = cache['groundtruths']
    dataset = DummyDataset(groundtruths, ind_to_classes, ind_to_predicates)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseBiasResidualCorrector(DummyDS(device), cfg).to(device)
    
    logger.info(f"Loading trained weights from {model_weights}")
    ckpt = torch.load(model_weights, map_location=device)
    # Check if it has state_dict wrapper or raw
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    # Apply corrector
    corrected_predictions = apply_corrector_to_predictions(predictions, model, device=device)

    # Evaluation
    iou_types = ["relations"]

    # 1. Eval Corrected
    logger.info("=" * 80)
    logger.info("Evaluating CORRECTED predictions...")
    logger.info("=" * 80)
    output_folder_corr = os.path.join(output_dir, "corrected")
    os.makedirs(output_folder_corr, exist_ok=True)
    
    with torch.no_grad():
        result_corr = vg_evaluation(
            cfg=cfg, dataset=dataset, predictions=corrected_predictions, 
            output_folder=output_folder_corr, logger=logger, iou_types=iou_types, groundtruths=groundtruths
        )

    # 2. Eval Original
    logger.info("=" * 80)
    logger.info("Evaluating ORIGINAL predictions...")
    logger.info("=" * 80)
    output_folder_orig = os.path.join(output_dir, "original")
    os.makedirs(output_folder_orig, exist_ok=True)
    
    with torch.no_grad():
        result_orig = vg_evaluation(
            cfg=cfg, dataset=dataset, predictions=predictions, 
            output_folder=output_folder_orig, logger=logger, iou_types=iou_types, groundtruths=groundtruths
        )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("      SUMMARY: ORIGINAL vs BIAS RESIDUAL CORRECTED")
    logger.info("=" * 80)
    
    orig_f = os.path.join(output_folder_orig, 'result_dict.pytorch')
    corr_f = os.path.join(output_folder_corr, 'result_dict.pytorch')
    
    if os.path.exists(orig_f) and os.path.exists(corr_f):
        try:
            r_orig = torch.load(orig_f, map_location='cpu', weights_only=False)
            r_corr = torch.load(corr_f, map_location='cpu', weights_only=False)
        except TypeError:
            r_orig = torch.load(orig_f, map_location='cpu')
            r_corr = torch.load(corr_f, map_location='cpu')
            
        metrics = ["R @ 20", "R @ 50", "R @ 100", "mR @ 20", "mR @ 50", "mR @ 100"]
        for key in metrics:
            if key in r_orig and key in r_corr:
                delta = r_corr[key] - r_orig[key]
                logger.info(f"  {key:<10}: {r_orig[key]:.4f}  ->  {r_corr[key]:.4f}  (Δ = {delta:+.4f})")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
