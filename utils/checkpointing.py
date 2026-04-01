# =============================================================================
# utils/checkpointing.py — 断点续训与权重保存模块
# 保存/加载: 模型权重 + 优化器状态 + 调度器状态 + AMP Scaler 状态
# =============================================================================

import os
import shutil

import torch


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    val_top1: float,
    best_top1: float,
    is_best: bool = False,
):
    """
    保存训练检查点。

    每个 epoch 保存 latest.pth (用于断点续训)。
    当 is_best=True 时,额外复制一份 best_model.pth (用于最终推理)。

    参数:
        checkpoint_dir (str):        检查点保存目录
        epoch          (int):        当前 epoch 编号
        model          (nn.Module):  模型
        optimizer      (Optimizer):  优化器
        scheduler:                   学习率调度器
        scaler:                      AMP GradScaler
        val_top1       (float):      当前 epoch 验证集 Top-1 准确率
        best_top1      (float):      历史最优 Top-1 准确率
        is_best        (bool):       当前 epoch 是否为历史最优
    """

    state = {
        "epoch": epoch + 1,          # 保存下一个要训练的 epoch 编号
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_top1": val_top1,
        "best_top1": best_top1,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存最新检查点 (每个 epoch 覆盖)
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(state, latest_path)

    # 如果是最优模型,额外保存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copy2(latest_path, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scaler=None,
    device: torch.device = None,
) -> tuple:
    """
    从检查点恢复训练状态。

    参数:
        checkpoint_path (str):        检查点文件路径
        model           (nn.Module):  模型 (会被原地加载权重)
        optimizer       (Optimizer):  优化器 (可选,推理时不需要)
        scheduler:                    调度器 (可选)
        scaler:                       AMP GradScaler (可选)
        device          (torch.device): 加载到的目标设备

    返回:
        tuple: (start_epoch, best_top1)
    """

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 1)
    best_top1 = checkpoint.get("best_top1", 0.0)

    return start_epoch, best_top1
