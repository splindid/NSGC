"""
Scene Graph Generation 训练脚本
Windows 兼容版 - 无 apex 依赖
"""

import os
import sys

# 解决 Windows 问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import logging
import time
import datetime

import torch
import torch.nn as nn

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def setup_logger(output_dir):
    """设置日志"""
    logger = logging.getLogger("SGG")
    logger.setLevel(logging.DEBUG)
    
    # 控制台输出
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件输出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train_log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="Scene Graph Generation Training")
    parser.add_argument("--config-file", default="configs/vg_mini.yaml", help="配置文件路径")
    parser.add_argument("--skip-model", action="store_true", help="跳过模型构建，只测试数据加载")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ============================================================
    # 1. 基础设置
    # ============================================================
    print("=" * 60)
    print("Scene Graph Generation Training")
    print("=" * 60)
    
    # 检查 CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # ============================================================
    # 2. 加载配置
    # ============================================================
    print("\n" + "-" * 40)
    print("Loading configuration...")
    
    from maskrcnn_benchmark.config import cfg
    
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
        print(f"Loaded config from: {args.config_file}")
    else:
        print(f"WARNING: Config file not found: {args.config_file}")
        print("Using default configuration")
    
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # 确保输出目录存在
    if not cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = "./output/default"
    
    cfg.freeze()
    
    # 设置日志
    logger = setup_logger(cfg.OUTPUT_DIR)
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")

    # ============================================================
    # 3. 构建数据加载器
    # ============================================================
    logger.info("\n" + "-" * 40)
    logger.info("Building data loader...")
    
    try:
        from maskrcnn_benchmark.data import make_data_loader
        
        data_loader = make_data_loader(
            cfg,
            mode='train',
            is_distributed=False,
            start_iter=0,
        )
        logger.info(f"✓ Data loader built successfully!")
        logger.info(f"  - Number of batches: {len(data_loader)}")
        logger.info(f"  - Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
        
    except Exception as e:
        logger.error(f"✗ Failed to build data loader: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试数据加载
    logger.info("\nTesting data loading...")
    try:
        for i, (images, targets, _) in enumerate(data_loader):
            logger.info(f"  Batch {i}: images shape = {images.tensors.shape}")
            logger.info(f"  Batch {i}: num targets = {len(targets)}")
            if i >= 2:  # 只测试前几个 batch
                break
        logger.info("✓ Data loading test passed!")
    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if args.skip_model:
        logger.info("Skipping model building (--skip-model flag)")
        return

    # ============================================================
    # 4. 构建模型
    # ============================================================
    logger.info("\n" + "-" * 40)
    logger.info("Building model...")
    
    try:
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        
        device = torch.device(cfg.MODEL.DEVICE)
        model = build_detection_model(cfg)
        model.to(device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✓ Model built successfully!")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Device: {device}")
        
    except Exception as e:
        logger.error(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # 5. 构建优化器
    # ============================================================
    logger.info("\n" + "-" * 40)
    logger.info("Building optimizer...")
    
    try:
        from maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler
        
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)
        
        logger.info(f"✓ Optimizer: {type(optimizer).__name__}")
        logger.info(f"  - Learning rate: {cfg.SOLVER.BASE_LR}")
        
    except Exception as e:
        logger.error(f"✗ Failed to build optimizer: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # 6. 训练循环
    # ============================================================
    logger.info("\n" + "-" * 40)
    logger.info("Starting training...")
    logger.info(f"  - Max iterations: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"  - Checkpoint period: {cfg.SOLVER.CHECKPOINT_PERIOD}")
    
    model.train()
    
    max_iter = cfg.SOLVER.MAX_ITER
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    
    start_time = time.time()
    iteration = 0
    
    try:
        for epoch in range(100):  # 足够多的 epoch
            for images, targets, _ in data_loader:
                iteration += 1
                
                if iteration > max_iter:
                    break
                
                # 跳过空标注
                if any(len(target) < 1 for target in targets):
                    logger.warning(f"Iter {iteration}: Empty target, skipping...")
                    continue
                
                # 移动数据到 GPU
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                
                # 前向传播
                try:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                except Exception as e:
                    logger.warning(f"Iter {iteration}: Forward error - {e}")
                    continue
                
                # 检查 loss
                if not torch.isfinite(losses):
                    logger.warning(f"Iter {iteration}: Loss is {losses}, skipping...")
                    continue
                
                # 反向传播
                optimizer.zero_grad()
                losses.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                scheduler.step()
                
                # 打印日志
                if iteration % 10 == 0 or iteration == 1:
                    elapsed = time.time() - start_time
                    eta = elapsed / iteration * (max_iter - iteration)
                    eta_str = str(datetime.timedelta(seconds=int(eta)))
                    
                    loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                    lr = optimizer.param_groups[0]["lr"]
                    
                    logger.info(
                        f"Iter [{iteration}/{max_iter}] "
                        f"LR: {lr:.6f} | Loss: {losses.item():.4f} | "
                        f"ETA: {eta_str}"
                    )
                    logger.info(f"  {loss_str}")
                
                # 保存检查点
                if iteration % checkpoint_period == 0:
                    save_path = os.path.join(cfg.OUTPUT_DIR, f"model_{iteration:06d}.pth")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "iteration": iteration,
                    }, save_path)
                    logger.info(f"✓ Saved checkpoint: {save_path}")
            
            if iteration > max_iter:
                break
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    
    # ============================================================
    # 7. 保存最终模型
    # ============================================================
    save_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    torch.save({
        "model": model.state_dict(),
        "iteration": iteration,
    }, save_path)
    
    total_time = time.time() 