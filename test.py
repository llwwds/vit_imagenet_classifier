# =============================================================================
# test.py — ViT 测试/推理主脚本
# 功能:
#   1. 加载训练好的模型权重 (best_model.pth)
#   2. 在验证集上评估 Top-1 / Top-5 准确率
#   3. 支持对单张图片进行推理预测
# =============================================================================

import os
import argparse

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
from PIL import Image

from model import build_vit
from data_processing import build_dataloaders, load_config
from utils.logger import get_logger
from utils.metrics import AverageMeter, accuracy


def evaluate(cfg: dict, checkpoint_path: str, logger):
    """
    在验证集上评估模型,报告 Top-1 和 Top-5 准确率。

    参数:
        cfg             (dict): 配置字典
        checkpoint_path (str):  模型权重文件路径
        logger:                 日志记录器
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"评估设备: {device}")

    # ── 构建模型并加载权重 ────────────────────────────────────────────────────
    model = build_vit(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"已加载权重: {checkpoint_path}")

    if "best_top1" in checkpoint:
        logger.info(f"该权重训练时最优 Top-1: {checkpoint['best_top1']:.2f}%")

    model.eval()

    # ── 构建验证集 DataLoader ─────────────────────────────────────────────────
    logger.info("构建验证集 DataLoader...")
    _, val_loader = build_dataloaders(cfg)

    # ── 评估循环 ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss().to(device)

    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top-1")
    top5_meter = AverageMeter("Top-5")

    logger.info("开始评估...")

    with torch.no_grad():
        for images, labels in val_loader:
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

    logger.info("=" * 50)
    logger.info("评估结果:")
    logger.info(f"  Loss:  {loss_meter.avg:.4f}")
    logger.info(f"  Top-1: {top1_meter.avg:.2f}%")
    logger.info(f"  Top-5: {top5_meter.avg:.2f}%")
    logger.info("=" * 50)


def predict_single_image(cfg: dict, checkpoint_path: str, image_path: str, logger):
    """
    对单张图片进行推理预测,输出 Top-5 预测类别和概率。

    参数:
        cfg             (dict): 配置字典
        checkpoint_path (str):  模型权重文件路径
        image_path      (str):  要预测的图片路径
        logger:                 日志记录器
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 构建模型并加载权重 ────────────────────────────────────────────────────
    model = build_vit(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── 图像预处理 (与验证集相同的 transform) ─────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载并预处理图片
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # ── 推理 ──────────────────────────────────────────────────────────────────
    with torch.no_grad():
        with autocast():
            logits = model(img_tensor)  # [1, num_classes]

    probs = torch.softmax(logits, dim=1)
    top5_probs, top5_indices = probs.topk(5, dim=1)

    # ── 加载类别名称 (如果有 synset_words.txt) ────────────────────────────────
    data_root = cfg["data"]["root"]
    synset_path = os.path.join(data_root, "synset_words.txt")
    class_names = None
    if os.path.isfile(synset_path):
        with open(synset_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    logger.info(f"\n预测结果: {image_path}")
    logger.info("-" * 40)
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item() * 100
        name = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
        logger.info(f"  Top-{i+1}: {name}  ({prob:.2f}%)")


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT 测试/推理脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="模型权重路径",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="单张图片路径 (如果指定,则进行单图推理而非验证集评估)",
    )

    args = parser.parse_args()

    logger = get_logger(name="test")
    cfg = load_config(args.config)

    if args.image:
        # 单图推理模式
        predict_single_image(cfg, args.checkpoint, args.image, logger)
    else:
        # 验证集评估模式
        evaluate(cfg, args.checkpoint, logger)
