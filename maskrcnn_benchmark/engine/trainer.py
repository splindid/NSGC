# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified: Removed apex dependency for PyTorch 2.x compatibility
"""
Training loop for Scene Graph Generation
"""
import datetime
import logging
import time
import sys

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    logger_step=100,
):
    """
    主训练循环
    
    Args:
        cfg: 配置对象
        model: 模型
        data_loader: 训练数据加载器
        data_loader_val: 验证数据加载器 (可选)
        optimizer: 优化器
        scheduler: 学习率调度器
        checkpointer: 检查点管理器
        device: 设备 (cuda/cpu)
        checkpoint_period: 保存检查点的周期
        test_period: 验证周期 (-1 表示不验证)
        arguments: 包含 iteration 等信息的字典
        logger_step: 日志打印周期
    
    Returns:
        训练后的模型
    """
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    
    model.train()
    
    start_training_time = time.time()
    end = time.time()
    
    # 梯度裁剪的最大范数
    grad_norm_clip = cfg.SOLVER.GRAD_NORM_CLIP if hasattr(cfg.SOLVER, 'GRAD_NORM_CLIP') else 5.0
    
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        # 检查空目标
        if any(len(target) < 1 for target in targets):
            logger.warning(f"Iteration={iteration + 1} has empty targets, skipping...")
            continue
            
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # 将数据移到设备
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # 前向传播
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())

        # 检查 NaN
        if torch.isnan(losses) or torch.isinf(losses):
            logger.warning(f"Iteration {iteration}: Loss is NaN/Inf, skipping...")
            continue

        # 减少损失（用于分布式训练日志）
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
        
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        # 计算 ETA
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # 打印日志
        if iteration % logger_step == 0 or iteration == max_iter or iteration == 1:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}/{max_iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f} MB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter=max_iter,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 if torch.cuda.is_available() else 0,
                )
            )
            sys.stdout.flush()

        # 保存检查点
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        # 验证（可选）
        if test_period > 0 and iteration % test_period == 0 and data_loader_val is not None:
            # TODO: 实现验证逻辑
            pass

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max(max_iter - start_iter, 1)
        )
    )
    
    return model
