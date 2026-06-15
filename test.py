"""
Scene Graph Generation Testing Script
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.logger import setup_logger
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/my_config.yaml")
    parser.add_argument("--ckpt", default="", help="checkpoint to load")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("test", cfg.OUTPUT_DIR, 0)
    
    # 构建模型
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model.to(device)
    model.eval()

    # 加载检查点
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    ckpt_path = args.ckpt if args.ckpt else None
    checkpointer.load(ckpt_path)

    # 构建数据加载器
    data_loader = make_data_loader(
        cfg,
        mode='test',
        is_distributed=False,
    )

    # 测试循环
    results = []
    with torch.no_grad():
        for images, targets, image_ids in tqdm(data_loader, desc="Testing"):
            images = images.to(device)
            output = model(images)
            results.extend(output)
    
    logger.info(f"Tested on {len(results)} images")
    # 这里可以添加评估代码


if __name__ == "__main__":
    main()