# =============================================================================
# train.py — ViT 训练主脚本
#
# 核心技术栈:
#   PyTorch AMP (autocast + GradScaler) — FP16 自动混合精度,激活 Tensor Cores
#   AdamW 优化器                        — Transformer 训练标准选择
#   CosineAnnealingLR + Warmup          — 学习率调度
#   Gradient Clipping                   — 防止 Transformer 训练早期梯度爆炸
#   Checkpointing                       — 断点续训 + 保存最优模型权重
# =============================================================================

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# ── 项目内部模块 ────────────────────────────────────────────────────────────────
from model import build_vit
from data_processing import build_dataloaders, load_config
from utils.logger import get_logger
from utils.metrics import AverageMeter, accuracy
from utils.checkpointing import save_checkpoint, load_checkpoint


# =============================================================================
# 第一部分: 优化器与学习率调度
# =============================================================================

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """
    构建 AdamW 优化器。

    为什么 ViT 使用 AdamW 而非 SGD?
        - Transformer 的损失面比 CNN 更不平坦,Adam 的自适应学习率有助于导航
        - AdamW 将权重衰减从梯度更新中解耦,正则化效果更好

    参数:
        model (nn.Module): 需要优化的模型
        cfg   (dict):      配置字典

    返回:
        optim.Optimizer: 配置好的 AdamW 优化器
    """
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"]["weight_decay"]
    betas = tuple(cfg["train"].get("betas", [0.9, 0.999]))

    # 对不同类型的参数分组:
    # - LayerNorm 和 bias 不应用权重衰减 (避免干扰归一化和偏置学习)
    # - 其余参数正常施加权重衰减
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or "cls_token" in name or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(param_groups, lr=lr, betas=betas)

    print(f"[Optimizer] AdamW  lr={lr}, weight_decay={weight_decay}, betas={betas}")
    print(f"  decay params: {len(decay_params)},  no-decay params: {len(no_decay_params)}")

    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: dict):
    """
    构建学习率调度器: 先 Warmup 再余弦退火。

    Phase 1 — Warmup: 学习率从 ~0 线性增长到 lr
    Phase 2 — CosineAnnealing: 从 lr 平滑降低到 min_lr

    参数:
        optimizer (Optimizer): 优化器
        cfg       (dict):      配置字典

    返回:
        scheduler: SequentialLR 调度器
    """
    total_epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"]["warmup_epochs"]
    min_lr = cfg["train"]["min_lr"]

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    print(
        f"[Scheduler] Warmup {warmup_epochs} epochs -> "
        f"CosineAnnealing {total_epochs - warmup_epochs} epochs (min_lr={min_lr})"
    )

    return scheduler


# =============================================================================
# 第二部分: 单个 epoch 的训练与验证
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger,
    cfg: dict,
) -> dict:
    """
    执行一个完整的训练 epoch。

    参数:
        model, loader, optimizer, criterion, scaler, device, epoch, logger, cfg

    返回:
        dict: {"loss", "top1", "top5", "throughput", "time"}
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top-1 Acc")
    top5_meter = AverageMeter("Top-5 Acc")
    batch_time = AverageMeter("Batch Time")

    log_interval = cfg["train"].get("log_interval", 50)
    max_grad_norm = cfg["train"].get("max_grad_norm", None)

    epoch_start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):
        batch_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP 混合精度前向传播
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # AMP 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪 (ViT 训练中非常重要,防止早期梯度爆炸)
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # 计算指标
        with torch.no_grad():
            acc1, acc5 = accuracy(logits.detach(), labels, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1, batch_size)
        top5_meter.update(acc5, batch_size)

        batch_elapsed = time.perf_counter() - batch_start
        batch_time.update(batch_elapsed)

        # 定期打印日志
        if (batch_idx + 1) % log_interval == 0:
            throughput = batch_size / batch_elapsed
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch [{epoch}] "
                f"Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Top-1: {top1_meter.avg:.2f}%  "
                f"Top-5: {top5_meter.avg:.2f}%  "
                f"LR: {current_lr:.6f}  "
                f"Throughput: {throughput:.0f} img/s"
            )

    # epoch 汇总
    epoch_elapsed = time.perf_counter() - epoch_start
    total_samples = len(loader.dataset)
    epoch_throughput = total_samples / epoch_elapsed

    logger.info(
        f"[Train Epoch {epoch} Summary] "
        f"Loss: {loss_meter.avg:.4f}  "
        f"Top-1: {top1_meter.avg:.2f}%  "
        f"Top-5: {top5_meter.avg:.2f}%  "
        f"Time: {epoch_elapsed:.1f}s  "
        f"Throughput: {epoch_throughput:.0f} img/s"
    )

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
        "throughput": epoch_throughput,
        "time": epoch_elapsed,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger,
) -> dict:
    """
    在验证集上评估模型性能。

    返回:
        dict: {"loss", "top1", "top5"}
    """
    model.eval()

    loss_meter = AverageMeter("Val Loss")
    top1_meter = AverageMeter("Val Top-1")
    top5_meter = AverageMeter("Val Top-5")

    val_start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1, batch_size)
        top5_meter.update(acc5, batch_size)

    val_elapsed = time.perf_counter() - val_start

    logger.info(
        f"[Val   Epoch {epoch} Summary] "
        f"Loss: {loss_meter.avg:.4f}  "
        f"Top-1: {top1_meter.avg:.2f}%  "
        f"Top-5: {top5_meter.avg:.2f}%  "
        f"Time: {val_elapsed:.1f}s"
    )

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }


# =============================================================================
# 第三部分: 主训练循环
# =============================================================================

def main(config_path: str):
    """
    训练主函数: 加载配置 -> 构建模型/数据/优化器 -> 训练循环 -> 保存模型
    """
    # ── 加载配置 ──────────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # ── 日志 ──────────────────────────────────────────────────────────────────
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(name="train", log_file=os.path.join(log_dir, "train.log"))
    logger.info("=" * 70)
    logger.info("ViT 图像分类训练启动")
    logger.info(f"配置文件: {config_path}")
    logger.info("=" * 70)

    # ── 设备 ──────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        logger.warning("未检测到 CUDA GPU! 将在 CPU 上训练 (速度极慢,仅供调试)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}  显存: {gpu_mem:.1f} GB")
        torch.backends.cudnn.benchmark = True

    # ── 数据 ──────────────────────────────────────────────────────────────────
    logger.info("构建数据加载器...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    logger.info(f"构建模型: {cfg['model'].get('name', 'ViT')}")
    model = build_vit(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params/1e6:.2f}M  |  可训练参数: {trainable_params/1e6:.2f}M")

    # ── 损失函数 ──────────────────────────────────────────────────────────────
    label_smoothing = cfg["train"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    logger.info(f"损失函数: CrossEntropyLoss (label_smoothing={label_smoothing})")

    # ── 优化器与调度器 ────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── AMP GradScaler ────────────────────────────────────────────────────────
    scaler = GradScaler()
    logger.info("AMP GradScaler 已初始化 (FP16 混合精度训练)")

    # ── 断点续训 ──────────────────────────────────────────────────────────────
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 1
    best_top1 = 0.0

    resume_path = cfg["train"].get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        logger.info(f"从 checkpoint 恢复训练: {resume_path}")
        start_epoch, best_top1 = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        logger.info(f"恢复成功: 从 Epoch {start_epoch} 继续, 历史最优 Top-1: {best_top1:.2f}%")
    else:
        logger.info("未指定 resume checkpoint,从头开始训练")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    total_epochs = cfg["train"]["epochs"]
    logger.info(f"\n开始训练: Epoch {start_epoch} -> {total_epochs}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch}/{total_epochs}  当前 LR: {epoch_lr:.6f}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            logger=logger,
            cfg=cfg,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger,
        )

        scheduler.step()

        # 保存 checkpoint
        is_best = val_stats["top1"] > best_top1
        if is_best:
            best_top1 = val_stats["top1"]
            logger.info(f"新最优! Val Top-1: {best_top1:.2f}%")

        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_top1=val_stats["top1"],
            best_top1=best_top1,
            is_best=is_best,
        )

    # ── 训练结束 ──────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("训练完成!")
    logger.info(f"验证集最优 Top-1 准确率: {best_top1:.2f}%")
    logger.info(f"最优权重已保存至: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    logger.info("=" * 70)


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT 图像分类训练脚本 (AMP + AdamW)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
