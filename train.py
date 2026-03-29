# =============================================================================
# train.py — 训练主脚本
# 目标: 在 RTX 4060 Ti 上,以 FP16 + Batch Size 1024,达到 ≈1667 img/s 吞吐量
#
# 核心技术栈:
#   PyTorch AMP (autocast + GradScaler) — FP16 自动混合精度,激活 Tensor Cores
#   Linear LR Scaling Rule             — 大 Batch 训练的学习率线性缩放
#   CosineAnnealingLR                  — 余弦退火,平滑降低学习率
#   Warmup                             — 前 N 个 epoch 线性升温,避免大 LR 训练早期发散
#   Checkpointing                      — 断点续训 + 保存最优模型权重
# =============================================================================

import os
import time
import yaml
import argparse                                       # 命令行参数解析

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler       # AMP 混合精度核心组件

# ── 项目内部模块 ────────────────────────────────────────────────────────────────
from model            import build_resnet34           # 模型构建工厂函数
from data_processing  import build_dataloaders, load_config  # 数据加载
from utils.logger     import get_logger               # 日志记录器
from utils.metrics    import AverageMeter, accuracy   # 性能指标工具
from utils.checkpointing import save_checkpoint, load_checkpoint  # 断点续训


# =============================================================================
# 第一部分: 学习率策略工具
# =============================================================================

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """
    构建优化器,并应用 Linear Scaling Rule 调整学习率。

    Linear Scaling Rule (Goyal et al., 2017):
        当 Batch Size 从 base_batch (通常 256) 扩大 k 倍时,
        初始学习率也应等比扩大 k 倍,以保持训练动态等价。
        公式: lr = base_lr × (batch_size / base_batch_size)

    参数:
        model (nn.Module): 需要优化的模型
        cfg   (dict):      配置字典

    返回:
        optim.Optimizer: 配置好学习率的 SGD 优化器
    """

    # ── 从配置文件读取超参数 ──────────────────────────────────────────────────
    base_lr         = cfg["train"]["base_lr"]          # 基准学习率 (对应 base_batch_size)
    base_batch_size = cfg["train"]["base_batch_size"]  # 基准 batch size,通常为 256
    batch_size      = cfg["data"]["batch_size"]        # 实际使用的 batch size (1024)
    momentum        = cfg["train"]["momentum"]         # SGD 动量,通常 0.9
    weight_decay    = cfg["train"]["weight_decay"]     # L2 正则化系数,通常 1e-4

    # ── Linear Scaling Rule 计算实际学习率 ───────────────────────────────────
    # 示例: base_lr=0.1, base_batch=256, batch=1024 → scaled_lr = 0.1 × 4 = 0.4
    scaled_lr = base_lr * (batch_size / base_batch_size)

    # ── 构建 SGD 优化器 ───────────────────────────────────────────────────────
    # 为什么选 SGD 而不是 Adam?
    #   在 ImageNet 规模的训练中,SGD + Momentum 通常比 Adam 收敛到更好的泛化性能
    #   Adam 的自适应学习率在大 batch 时可能导致泛化差距 (generalization gap)
    optimizer = optim.SGD(
        model.parameters(),    # 要优化的参数
        lr=scaled_lr,          # 线性缩放后的学习率
        momentum=momentum,     # 动量: 累积历史梯度方向,加速收敛,抑制震荡
        weight_decay=weight_decay,  # L2 正则化: 防止权重过大,抑制过拟合
        nesterov=True          # Nesterov 动量: 在动量方向上超前一步计算梯度,
                               # 相比普通动量收敛更快
    )

    print(f"[Optimizer] SGD  lr={scaled_lr:.4f} (base={base_lr} × {batch_size}/{base_batch_size}), "
          f"momentum={momentum}, weight_decay={weight_decay}, nesterov=True")

    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: dict):
    """
    构建学习率调度器:先 Warmup,再余弦退火。

    两阶段策略:
        Phase 1 — Warmup (前 warmup_epochs 个 epoch):
            学习率从 0 线性增长到 scaled_lr
            目的: 在训练初期参数随机分散时,用小 LR 稳定训练,避免早期梯度爆炸

        Phase 2 — CosineAnnealingLR (剩余 epoch):
            学习率按余弦曲线从 scaled_lr 平滑降低到 min_lr (接近 0)
            目的: 训练后期用更小的步长精细调整权重,提升最终精度

    参数:
        optimizer (Optimizer): 需要调度的优化器
        cfg       (dict):      配置字典

    返回:
        scheduler: 包含 warmup + cosine 两阶段的 SequentialLR 调度器
    """

    total_epochs  = cfg["train"]["epochs"]           # 总训练 epoch 数
    warmup_epochs = cfg["train"]["warmup_epochs"]    # Warmup 持续 epoch 数,通常 5
    min_lr        = cfg["train"]["min_lr"]           # 余弦退火的最低学习率

    # ── Warmup 调度器: 线性升温 ──────────────────────────────────────────────
    # LinearLR: 将学习率从 start_factor × lr 线性增长到 lr
    # start_factor=1e-4: 起始 LR 极小 (约为 scaled_lr 的 1/10000)
    # total_iters=warmup_epochs: Warmup 持续的 epoch 数
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,       # warmup 起始学习率倍率 (相对于 scaled_lr)
        end_factor=1.0,          # warmup 结束学习率倍率 (恢复到 scaled_lr)
        total_iters=warmup_epochs
    )

    # ── 余弦退火调度器 ────────────────────────────────────────────────────────
    # CosineAnnealingLR: 学习率按余弦曲线从当前 LR 降低到 eta_min
    # T_max: 余弦周期的半周期长度,这里设为 warmup 结束后的剩余 epoch 数
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,   # 余弦退火的周期
        eta_min=min_lr                         # 最低学习率下限
    )

    # ── SequentialLR: 串联两个调度器 ─────────────────────────────────────────
    # milestones=[warmup_epochs]: 在第 warmup_epochs 个 epoch 从 warmup 切换到 cosine
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    print(f"[Scheduler] Warmup {warmup_epochs} epochs → CosineAnnealing {total_epochs - warmup_epochs} epochs "
          f"(min_lr={min_lr})")

    return scheduler


# =============================================================================
# 第二部分: 单个 epoch 的训练与验证
# =============================================================================

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    scaler:     GradScaler,
    device:     torch.device,
    epoch:      int,
    logger,
    cfg:        dict
) -> dict:
    """
    执行一个完整的训练 epoch,返回该 epoch 的统计指标。

    AMP 工作流程:
        1. autocast()        : 在 FP16 下执行前向传播 (卷积、BN),
                               自动识别需要 FP32 的操作 (如 Softmax in CrossEntropy)
        2. scaler.scale()    : 将 loss 乘以 scale factor,防止 FP16 梯度下溢
        3. scaler.step()     : 将梯度反缩放回 FP32 精度,再更新参数
        4. scaler.update()   : 动态调整 scale factor (若出现 inf/nan 则降低)

    参数:
        model     : 训练中的模型
        loader    : 训练集 DataLoader
        optimizer : SGD 优化器
        criterion : 损失函数 (CrossEntropyLoss)
        scaler    : AMP GradScaler
        device    : 训练设备 (cuda)
        epoch     : 当前 epoch 编号 (从 1 开始)
        logger    : 日志记录器
        cfg       : 配置字典

    返回:
        dict: {"loss": float, "top1": float, "top5": float, "throughput": float}
    """

    # 切换模型到训练模式
    # 训练模式下: BN 使用当前 batch 统计量; Dropout 按概率随机丢弃神经元
    model.train()

    # AverageMeter: 滑动平均计数器,用于累积 loss、accuracy、时间等指标
    loss_meter  = AverageMeter("Loss")
    top1_meter  = AverageMeter("Top-1 Acc")
    top5_meter  = AverageMeter("Top-5 Acc")
    batch_time  = AverageMeter("Batch Time")   # 每个 batch 的处理时间

    log_interval = cfg["train"].get("log_interval", 50)  # 每隔多少 batch 打印一次日志

    epoch_start = time.perf_counter()   # 记录 epoch 开始时间,用于计算吞吐量

    for batch_idx, (images, labels) in enumerate(loader):

        batch_start = time.perf_counter()  # 记录 batch 开始时间

        # ── 数据搬运到 GPU ────────────────────────────────────────────────────
        # non_blocking=True: 异步传输,CPU 不等待 GPU 确认,立即返回继续执行
        # 配合 pin_memory=True 才能生效 (锁页内存 → DMA 异步传输)
        images = images.to(device, non_blocking=True)   # [N, 3, 224, 224] → GPU
        labels = labels.to(device, non_blocking=True)   # [N] → GPU

        # ── 清空上一个 batch 的梯度 ──────────────────────────────────────────
        # set_to_none=True: 将梯度置为 None 而非 0
        # 相比 zero_grad(),内存操作更少,速度略快
        optimizer.zero_grad(set_to_none=True)

        # ── 前向传播 (FP16 混合精度) ─────────────────────────────────────────
        # autocast() 上下文: 自动将卷积、矩阵乘法等算子切换为 FP16
        #                    保留 Loss 计算、BN 等精度敏感操作为 FP32
        # 效果: 激活 RTX 4060 Ti 的 Tensor Cores,吞吐量提升约 2-3x
        with autocast():
            logits = model(images)          # 前向传播,得到 [N, num_classes] logits
            loss   = criterion(logits, labels)  # 计算交叉熵损失

        # ── 反向传播 (梯度缩放) ──────────────────────────────────────────────
        # scaler.scale(loss): 将 loss 乘以 scale factor (初始约 65536)
        #                     防止 FP16 梯度因数值过小而下溢到 0
        scaler.scale(loss).backward()       # 反向传播,计算缩放后的梯度

        # ── 梯度裁剪 (可选,防止梯度爆炸) ────────────────────────────────────
        # 在 step 之前必须先 unscale,否则裁剪的是缩放后的梯度 (数值错误)
        max_grad_norm = cfg["train"].get("max_grad_norm", None)
        if max_grad_norm is not None:
            # unscale: 将梯度从缩放空间还原到真实 FP32 空间
            scaler.unscale_(optimizer)
            # clip_grad_norm_: 将所有参数梯度的全局 L2 范数裁剪到 max_grad_norm
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # ── 参数更新 ─────────────────────────────────────────────────────────
        # scaler.step(): 内部先反缩放梯度,检查是否有 inf/nan,
        #                若无异常则调用 optimizer.step() 更新参数
        scaler.step(optimizer)

        # scaler.update(): 根据本次是否出现 inf/nan 动态调整 scale factor
        #                  正常 → scale factor 逐渐增大 (充分利用 FP16 精度范围)
        #                  异常 → scale factor 减半 (防止溢出)
        scaler.update()

        # ── 计算指标 ─────────────────────────────────────────────────────────
        # 计算 Top-1 和 Top-5 准确率
        # 注意: 必须用 .detach() 切断计算图,防止持有不必要的引用占用显存
        with torch.no_grad():
            acc1, acc5 = accuracy(logits.detach(), labels, topk=(1, 5))

        batch_size = images.size(0)  # 当前 batch 的实际样本数

        # 更新滑动平均计数器
        loss_meter.update(loss.item(), batch_size)   # loss.item() 将 Tensor 转为 Python float
        top1_meter.update(acc1,        batch_size)
        top5_meter.update(acc5,        batch_size)

        batch_elapsed = time.perf_counter() - batch_start
        batch_time.update(batch_elapsed)

        # ── 定期打印训练日志 ─────────────────────────────────────────────────
        if (batch_idx + 1) % log_interval == 0:
            # 计算当前吞吐量 (图片/秒)
            throughput = batch_size / batch_elapsed
            current_lr = optimizer.param_groups[0]["lr"]  # 获取当前学习率

            logger.info(
                f"Epoch [{epoch}] "
                f"Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Top-1: {top1_meter.avg:.2f}%  "
                f"Top-5: {top5_meter.avg:.2f}%  "
                f"LR: {current_lr:.6f}  "
                f"Throughput: {throughput:.0f} img/s"
            )

    # ── epoch 级别汇总 ────────────────────────────────────────────────────────
    epoch_elapsed  = time.perf_counter() - epoch_start
    total_samples  = len(loader.dataset)
    # 整个 epoch 的平均吞吐量
    epoch_throughput = total_samples / epoch_elapsed

    logger.info(
        f"[Train Epoch {epoch} Summary] "
        f"Loss: {loss_meter.avg:.4f}  "
        f"Top-1: {top1_meter.avg:.2f}%  "
        f"Top-5: {top5_meter.avg:.2f}%  "
        f"Time: {epoch_elapsed:.1f}s  "
        f"Throughput: {epoch_throughput:.0f} img/s  "
        f"{'✅ 达标' if epoch_throughput >= 1667 else '⚠️ 未达标 (目标 1667 img/s)'}"
    )

    return {
        "loss":       loss_meter.avg,
        "top1":       top1_meter.avg,
        "top5":       top5_meter.avg,
        "throughput": epoch_throughput,
        "time":       epoch_elapsed,
    }


@torch.no_grad()  # 装饰器: 整个函数内关闭梯度计算,节省显存和计算
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    logger,
) -> dict:
    """
    在验证集上评估模型性能。

    与训练的关键区别:
        - model.eval(): BN 使用全局统计量 (训练集上累积的 running_mean/var)
        - torch.no_grad(): 不建立计算图,不计算梯度 → 节省约 50% 显存
        - 不进行数据增强 (val transform 是确定性的)

    参数:
        model     : 评估模式下的模型
        loader    : 验证集 DataLoader
        criterion : 损失函数
        device    : 评估设备
        epoch     : 当前 epoch 编号
        logger    : 日志记录器

    返回:
        dict: {"loss": float, "top1": float, "top5": float}
    """

    # 切换到评估模式
    # 评估模式: BN 使用 running_mean/var (而非当前 batch 统计量)
    #           Dropout 关闭 (所有神经元全部激活)
    model.eval()

    loss_meter = AverageMeter("Val Loss")
    top1_meter = AverageMeter("Val Top-1")
    top5_meter = AverageMeter("Val Top-5")

    val_start = time.perf_counter()

    for images, labels in loader:

        # 搬运到 GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 验证也使用 autocast 加速前向传播 (虽然不做梯度计算,但 FP16 前向更快)
        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        # 计算 Top-1 和 Top-5 准确率
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1,        batch_size)
        top5_meter.update(acc5,        batch_size)

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
    训练主函数: 串联所有组件,执行完整的训练流程。

    流程:
        加载配置 → 初始化设备 → 构建模型/数据/优化器/调度器
        → 循环 epoch: 训练 → 验证 → 保存检查点 → 更新 LR
        → 结束,打印最终结果

    参数:
        config_path (str): 配置文件路径
    """

    # ── 加载配置文件 ──────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # ── 初始化日志记录器 ──────────────────────────────────────────────────────
    # 日志同时输出到控制台和文件,方便训练结束后回溯
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)         # 确保日志目录存在
    logger = get_logger(
        name="train",
        log_file=os.path.join(log_dir, "train.log")
    )
    logger.info("=" * 70)
    logger.info("ResNet-34 训练启动")
    logger.info(f"配置文件: {config_path}")
    logger.info("=" * 70)

    # ── 设备选择 ──────────────────────────────────────────────────────────────
    # 优先使用 CUDA GPU,没有 GPU 时退回 CPU (速度会非常慢)
    if not torch.cuda.is_available():
        logger.warning("⚠️  未检测到 CUDA GPU!将在 CPU 上训练 (速度极慢,仅供调试)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备: {device}")

    # 打印 GPU 信息
    if device.type == "cuda":
        gpu_name  = torch.cuda.get_device_name(0)
        gpu_mem   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}  显存: {gpu_mem:.1f} GB")

        # 启用 cuDNN 自动优化:针对固定输入尺寸 (224×224) 寻找最快的卷积算法
        # 注意: 如果输入尺寸会变化,请设为 False 防止频繁重新搜索
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark: True (输入尺寸固定时加速卷积)")

    # ── 构建数据加载器 ────────────────────────────────────────────────────────
    logger.info("构建数据加载器...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    num_classes = cfg["model"]["num_classes"]   # 从配置读取类别数
    logger.info(f"构建 ResNet-34,类别数: {num_classes}")
    model = build_resnet34(num_classes=num_classes)
    model = model.to(device)                     # 将模型参数移动到 GPU

    # 统计并打印模型参数量
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params/1e6:.2f}M  |  可训练参数: {trainable_params/1e6:.2f}M")

    # ── 构建损失函数 ──────────────────────────────────────────────────────────
    # CrossEntropyLoss = Log Softmax + Negative Log Likelihood Loss
    # label_smoothing: 标签平滑,将硬标签 [0,1] 软化为 [ε/K, 1-ε+ε/K]
    #                  防止模型过度自信,提升泛化能力,对大 batch 训练尤其有效
    label_smoothing = cfg["train"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    logger.info(f"损失函数: CrossEntropyLoss (label_smoothing={label_smoothing})")

    # ── 构建优化器和学习率调度器 ──────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── 构建 AMP GradScaler ───────────────────────────────────────────────────
    # GradScaler 动态管理 FP16 梯度缩放,防止梯度下溢
    # 初始 scale 约为 65536,会根据训练状态自动调整
    scaler = GradScaler()
    logger.info("AMP GradScaler 已初始化 (FP16 混合精度训练)")

    # ── 断点续训: 从 checkpoint 恢复 ─────────────────────────────────────────
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)   # 确保 checkpoint 目录存在

    start_epoch = 1          # 默认从第 1 epoch 开始
    best_top1   = 0.0        # 记录验证集最优 Top-1 准确率

    # 如果配置文件中指定了 resume 路径,则加载该 checkpoint
    resume_path = cfg["train"].get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        logger.info(f"从 checkpoint 恢复训练: {resume_path}")
        start_epoch, best_top1 = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        logger.info(f"恢复成功: 从 Epoch {start_epoch} 继续, 历史最优 Top-1: {best_top1:.2f}%")
    else:
        logger.info("未指定 resume checkpoint,从头开始训练")

    # ── 主训练循环 ────────────────────────────────────────────────────────────
    total_epochs = cfg["train"]["epochs"]
    logger.info(f"\n开始训练: Epoch {start_epoch} → {total_epochs}")
    logger.info(f"{'='*70}")

    for epoch in range(start_epoch, total_epochs + 1):

        epoch_lr = optimizer.param_groups[0]["lr"]   # 获取当前学习率用于记录
        logger.info(f"\n{'─'*60}")
        logger.info(f"Epoch {epoch}/{total_epochs}  当前 LR: {epoch_lr:.6f}")

        # ── 训练一个 epoch ────────────────────────────────────────────────────
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

        # ── 在验证集上评估 ─────────────────────────────────────────────────────
        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger,
        )

        # ── 更新学习率调度器 ───────────────────────────────────────────────────
        # scheduler.step() 必须在 optimizer.step() 之后,每个 epoch 结束时调用一次
        scheduler.step()

        # ── 保存 checkpoint ────────────────────────────────────────────────────
        is_best = val_stats["top1"] > best_top1
        if is_best:
            best_top1 = val_stats["top1"]
            logger.info(f"🏆 新最优! Val Top-1: {best_top1:.2f}%")

        # 每个 epoch 都保存最新 checkpoint (用于断点续训)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_top1=val_stats["top1"],
            best_top1=best_top1,
            is_best=is_best,           # is_best=True 时额外保存 best_model.pth
        )

    # ── 训练结束汇总 ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("训练完成!")
    logger.info(f"验证集最优 Top-1 准确率: {best_top1:.2f}%")
    logger.info(f"最优权重已保存至: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    logger.info(f"{'='*70}")


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":

    # 构建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="ResNet-34 训练脚本 (FP16 + AMP, 目标 ≈1667 img/s)"
    )

    # --config 参数: 指定配置文件路径,默认使用 configs/default.yaml
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 启动训练主函数
    # 使用方法:
    #   python train.py                              # 使用默认配置
    #   python train.py --config configs/myexp.yaml  # 使用自定义配置
    main(config_path=args.config)
