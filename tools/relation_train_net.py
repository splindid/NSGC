# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified: Removed apex dependency, added Windows compatibility

"""
Basic training script for PyTorch - Scene Graph Generation
Modified for Windows compatibility without apex
"""

import argparse
import os
import sys
import random
import logging

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config


def set_seed(seed, rank=0):
    """设置随机种子以确保可重复性"""
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保 CUDA 确定性（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, local_rank, distributed, logger):
    """主训练函数"""
    
    # 设置随机种子
    if cfg.SEED >= 0:
        set_seed(cfg.SEED, get_rank())
        logger.info(f"Using seed: {cfg.SEED + get_rank()}")
    
    # 构建模型
    model = build_detection_model(cfg)##构建模型（*GeneralizedRCNN）模型（*RCNN*）
    '''
    输入：
        cfg: 配置对象
    输出：模型（GeneralizedRCNN）
    '''
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)##进行配置cuda加速
    
    # 构建优化器
    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))##配置优化器
    '''
    权重衰退器和权重偏置选择
    '''
    scheduler = make_lr_scheduler(cfg, optimizer, logger)##配置学习器
    '''
    WarmupMultiStepLR和WarmupReduceLROnPlateau两种学习率调整策略
    '''
    
    # 分布式训练设置
    if distributed:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True,  # 场景图生成可能需要这个
        )
    
    # 准备参数
    arguments = {}
    arguments["iteration"] = 0
    
    # 输出目录
    output_dir = cfg.OUTPUT_DIR
    
    # Checkpointer
    save_to_disk = is_main_process()
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, logger
    )
    
    # 加载预训练模型或恢复训练
    extra_checkpoint_data = checkpointer.load(
        cfg.MODEL.PRETRAINED_DETECTOR_CKPT, 
        with_optim=False,
        update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD
    )
    arguments.update(extra_checkpoint_data)
    
    # 构建数据加载器
    data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    
    # 验证数据加载器（可选）
    data_loader_val = None
    if cfg.SOLVER.TO_VAL:
        data_loader_val = make_data_loader(
            cfg, 
            mode='val', 
            is_distributed=distributed
        )
    
    # 开始训练
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    val_period = cfg.SOLVER.VAL_PERIOD if cfg.SOLVER.TO_VAL else -1
    
    model = do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        val_period,
        arguments,
        logger_step=cfg.SOLVER.PRINT_GRAD_FREQ if hasattr(cfg.SOLVER, 'PRINT_GRAD_FREQ') else 100,
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Scene Graph Generation Training")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        default="configs/gqa_mini.yaml",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # 检查是否使用分布式训练
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
        synchronize()

    # 加载配置
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 创建输出目录
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    # 设置日志
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    # 开始训练
    model = train(cfg, args.local_rank, distributed, logger)


if __name__ == "__main__":
    main()