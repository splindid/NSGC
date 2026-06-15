# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

# 添加项目根目录到 Python 路径（无需安装项目）
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np
# Fix for numpy 1.24+ compatibility with older pycocotools
if not hasattr(np, "float"):
    np.float = float

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_relation_detector_R_101_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--quick-test",
        type=int,
        default=None,
        help="Only test on the first N images for quick validation",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    # 检查是否可以从缓存加载，以跳过模型构建
    output_dir = cfg.OUTPUT_DIR
    dataset_names = cfg.DATASETS.TEST
    if cfg.DATASETS.TO_TEST == 'train':
        dataset_names = cfg.DATASETS.TRAIN
    elif cfg.DATASETS.TO_TEST == 'val':
        dataset_names = cfg.DATASETS.VAL

    all_caches_exist = True
    if output_dir:
        for dataset_name in dataset_names:
            cache_path = os.path.join(output_dir, "inference", dataset_name, "eval_results.pytorch")
            if not os.path.exists(cache_path):
                all_caches_exist = False
                break
    else:
        all_caches_exist = False

    if cfg.TEST.ALLOW_LOAD_FROM_CACHE and all_caches_exist:
        logger.info("All caches found, skipping model building to avoid size mismatch.")
        model = None
    else:
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        
        # --- 关键：加载你训练好的后处理模型权重 ---
        hb3c_weight_path = "output/hb3c_weights_final.pth"
        if os.path.exists(hb3c_weight_path):
            logger.info(f"Loading trained HB3C weights from {hb3c_weight_path}")
            # 适配 CombinedROIHeads 结构
            if hasattr(model.roi_heads, "relation"):
                rel_head = model.roi_heads.relation
                if hasattr(rel_head, "ds") and hasattr(rel_head.ds, "corrector"):
                    rel_head.ds.corrector.load_state_dict(torch.load(hb3c_weight_path, map_location=cfg.MODEL.DEVICE))
                else:
                    logger.warning("HB3C weights found but model.roi_heads.relation.ds.corrector not found.")
            else:
                logger.warning("HB3C weights found but model.roi_heads.relation not found.")
        # ------------------------------------------

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    # amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    # --- 修改此处：取消默认的 bbox 评估以节省时间 ---
    iou_types = ()
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    # --------------------------------------------
    output_folders = [None] * len(cfg.DATASETS.TEST)

    dataset_names = cfg.DATASETS.TEST

    # This variable enables the script to run the test on any dataset split.
    if cfg.DATASETS.TO_TEST:
        assert cfg.DATASETS.TO_TEST in {'train', 'val', 'test', None}
        if cfg.DATASETS.TO_TEST == 'train':
            dataset_names = cfg.DATASETS.TRAIN
        elif cfg.DATASETS.TO_TEST == 'val':
            dataset_names = cfg.DATASETS.VAL


    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg=cfg, mode="test", is_distributed=distributed, dataset_to_test=cfg.DATASETS.TO_TEST)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        # Quick test mode: limit number of images
        if args.quick_test is not None:
            logger.info(f"Quick test mode: Only evaluating first {args.quick_test} images")
            # Wrap the data loader to limit iterations
            from itertools import islice
            limited_data = list(islice(data_loader_val, args.quick_test))
            class LimitedDataLoader:
                def __init__(self, data):
                    self.data = data
                    self.dataset = data_loader_val.dataset
                def __iter__(self):
                    return iter(self.data)
                def __len__(self):
                    return len(self.data)
            data_loader_val = LimitedDataLoader(limited_data)
        
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
