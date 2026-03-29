# =============================================================================
# data_processing.py — 数据预处理与加载模块
# 目标: 以足够快的速度向 GPU 供给数据,保证吞吐量不受 CPU 瓶颈拖累
# 核心策略:
#   1. pin_memory=True      — 锁页内存,加速 CPU→GPU 数据传输
#   2. num_workers 多进程   — 并行解码图片,CPU 预处理不成为瓶颈
#   3. persistent_workers   — 避免每个 epoch 重建进程池的开销
#   4. prefetch_factor      — 预取 batch,隐藏 I/O 延迟
# =============================================================================

import os  # 文件路径操作

import torch  # PyTorch 核心
import yaml  # 读取 default.yaml 配置文件
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import datasets, transforms  # 标准数据集与图像变换

# =============================================================================
# 第一部分: 图像变换流水线 (Augmentation Pipeline)
# =============================================================================


def build_train_transform() -> transforms.Compose:
    """
    构建训练集数据增强流水线。

    设计原则:
        - 使用标准 ImageNet 训练增强,经过大量实验验证的最佳实践
        - 增强在 CPU DataLoader workers 中完成,不占用 GPU 算力
        - 最后 Normalize 使用 ImageNet 统计均值/方差,让输入分布接近 N(0,1)

    返回:
        transforms.Compose: 按顺序执行的变换组合
    """
    return transforms.Compose(
        [
            # ── 随机裁剪 ─────────────────────────────────────────────────────────
            # 从原图中随机截取一块区域,再 resize 到 224×224
            # - scale=(0.08, 1.0) : 裁剪区域面积占原图的 8%~100%
            #                        迫使模型学习从局部特征推断类别
            # - ratio=(3/4, 4/3)  : 裁剪框的宽高比范围,防止过度形变
            # 这是 ImageNet 训练中最重要的增强手段之一
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
            # ── 随机水平翻转 ───────────────────────────────────────────────────────
            # 50% 概率将图片左右镜像
            # 直觉: 猫向左看和向右看都是猫,标签不应改变
            transforms.RandomHorizontalFlip(p=0.5),
            # ── 颜色抖动 ───────────────────────────────────────────────────────────
            # 随机调整亮度、对比度、饱和度、色调,模拟不同光照和相机条件
            # - brightness=0.4 : 亮度随机变化 ±40%
            # - contrast=0.4   : 对比度随机变化 ±40%
            # - saturation=0.4 : 饱和度随机变化 ±40%
            # - hue=0.1        : 色调随机偏移 ±10% (过大会导致颜色失真)
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            # ── Tensor 转换 ────────────────────────────────────────────────────────
            # PIL Image (HWC, uint8, [0,255]) → Tensor (CHW, float32, [0.0, 1.0])
            # H=Height, W=Width, C=Channel
            # 自动将像素值除以 255 完成归一化到 [0, 1]
            transforms.ToTensor(),
            # ── ImageNet 标准归一化 ────────────────────────────────────────────────
            # 公式: output[c] = (input[c] - mean[c]) / std[c]
            # mean 和 std 是在整个 ImageNet 训练集上统计出的固定值
            # 效果: 将输入分布规范化到均值≈0、标准差≈1,显著加速收敛
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # R, G, B 三通道均值
                std=[0.229, 0.224, 0.225],  # R, G, B 三通道标准差
            ),
        ]
    )


def build_val_transform() -> transforms.Compose:
    """
    构建验证集 / 测试集变换流水线。

    与训练集的关键区别:
        - 不使用任何随机增强,确保评估结果可复现
        - Resize → CenterCrop 的确定性裁剪是业界标准验证方式
        - Normalize 参数与训练集完全相同,保证输入分布一致

    返回:
        transforms.Compose: 确定性变换组合
    """
    return transforms.Compose(
        [
            # ── 等比缩放 ───────────────────────────────────────────────────────────
            # 将图片的短边缩放到 256,长边等比例缩放
            # 为什么是 256 而不是直接 224?
            # 因为后续 CenterCrop(224) 需要一定的"边缘余量"来从中心裁剪
            # 直接 resize 到 224 的话,CenterCrop 相当于无操作
            transforms.Resize(256),
            # ── 中心裁剪 ───────────────────────────────────────────────────────────
            # 从图片中心截取 224×224 区域
            # 确定性操作: 每次对同一张图的结果完全相同
            transforms.CenterCrop(224),
            # ── Tensor 转换 ────────────────────────────────────────────────────────
            transforms.ToTensor(),
            # ── ImageNet 标准归一化 (与训练集参数完全相同) ─────────────────────────
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# =============================================================================
# 第二部分: 数据集构建
# =============================================================================


def build_datasets(data_root: str):
    """
    从磁盘加载训练集和验证集,经过train和val的工具流水线处理，返回处理好的datasets

    要求的目录结构 (torchvision ImageFolder 格式):
        data_root/
        ├── train/
        │   ├── 类别A/   img1.jpg, img2.jpg, ...
        │   ├── 类别B/   img3.jpg, ...
        │   └── ...
        └── val/
            ├── 类别A/   img4.jpg, ...
            ├── 类别B/   img5.jpg, ...
            └── ...

    ImageFolder 的自动行为:
        1. 扫描所有子文件夹,按字母序将文件夹名映射为整数标签 (0, 1, 2, ...)
        2. 返回 (image_tensor, label_int) 格式的样本对

    参数:
        data_root (str): 数据集根目录,例如 "dataset/imagenet"

    返回:
        tuple: (train_dataset, val_dataset)
    """

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # 提前检查目录,给出比 Python 默认报错更清晰的提示
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"训练集目录不存在: {train_dir}\n"
            f"请确认数据集已放置在正确路径,并检查 configs/default.yaml 中的 data.root。"
        )
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"验证集目录不存在: {val_dir}\n"
            f"请确认数据集已放置在正确路径,并检查 configs/default.yaml 中的 data.root。"
        )

    # 加载训练集,绑定带随机增强的 transform
    # 使用前面自己定义的build_train_transform()工具流水线
    train_dataset = datasets.ImageFolder(
        root=train_dir, transform=build_train_transform()
    )

    # 加载验证集,绑定确定性 transform
    # 使用前面自己定义的build_val_transform()工具流水线
    val_dataset = datasets.ImageFolder(root=val_dir, transform=build_val_transform())

    return train_dataset, val_dataset


# =============================================================================
# 第三部分: DataLoader 构建 (性能关键)
# =============================================================================


def build_dataloaders(cfg: dict):
    """
    根据配置字典构建高性能 DataLoader。

    性能优化要点 (针对 RTX 4060 Ti + Batch Size 1024):
    ┌─────────────────────┬───────────────────────────────────────────────────┐
    │ 参数                 │ 作用                                              │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ pin_memory=True     │ 将 Tensor 预分配在 CPU 锁页内存,GPU 通过 DMA        │
    │                     │ 直接读取,绕过操作系统内核拷贝,传输速度提升 2-3x       │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ num_workers=N       │ N 个独立子进程并行解码图片+执行 transform            │
    │                     │ 经验值: CPU 核心数 或 CPU 核心数 × 2                │
    │                     │ 设为 0 则串行加载,必然成为性能瓶颈                   │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ persistent_workers  │ epoch 结束后不销毁工作进程                          │
    │                     │ 避免每 epoch 重新 fork/初始化进程的数秒开销          │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ prefetch_factor=2   │ 每个 worker 预先加载 2 个 batch                    │
    │                     │ GPU 处理当前 batch 时,下一批已在内存就绪             │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ drop_last=True      │ 丢弃训练集最后一个不完整 batch                      │
    │                     │ 保证 BN 统计量计算稳定 (避免单样本 batch)            │
    └─────────────────────┴───────────────────────────────────────────────────┘

    参数:
        cfg (dict): 从 default.yaml 解析的配置字典

    返回:
        tuple: (train_loader, val_loader)
    """

    # ── 从配置文件读取参数 ────────────────────────────────────────────────────
    data_root = cfg["data"]["root"]
    batch_size = cfg["data"]["batch_size"]  # 训练 batch size,推荐 1024
    num_workers = cfg["data"]["num_workers"]  # 数据加载进程数
    prefetch_factor = cfg["data"].get("prefetch_factor", 2)  # 预取倍数,默认 2
    # 验证集 batch size: 无反向传播,显存占用更少,可以用更大的 batch
    val_batch_size = cfg["data"].get("val_batch_size", batch_size * 2)

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset, val_dataset = build_datasets(data_root)

    # 打印数据集信息,方便确认加载正确
    num_classes = len(train_dataset.classes)
    print(
        f"[DataLoader] 训练集: {len(train_dataset):,} 张  |  "
        f"验证集: {len(val_dataset):,} 张  |  "
        f"类别数: {num_classes}"
    )
    print(
        f"[DataLoader] train batch={batch_size}, val batch={val_batch_size}, "
        f"workers={num_workers}, prefetch={prefetch_factor}"
    )

    # ── 训练集 DataLoader ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 每个 epoch 随机打乱,防止模型记住样本顺序
        num_workers=num_workers,  # 并行加载进程数
        # 0 = 主进程串行加载 (调试用,生产环境必须 >0)
        pin_memory=True,  # 锁页内存加速,仅 CUDA 训练时生效
        persistent_workers=(num_workers > 0),
        # 保持 worker 进程存活,避免 epoch 间重建进程
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        # num_workers=0 时此参数必须为 None
        drop_last=True,  # 丢弃最后不完整的 batch
        # 保证每个 batch 大小完全一致
    )

    # ── 验证集 DataLoader ──────────────────────────────────────────────────────
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,  # 验证集不需要打乱,保证结果可复现
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,  # 验证集必须保留所有样本,确保评估完整
    )

    return train_loader, val_loader


# =============================================================================
# 第四部分: 配置文件读取工具
# =============================================================================


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """
    从 YAML 配置文件读取超参数。

    参数:
        config_path (str): 配置文件路径

    返回:
        dict: 所有配置项的字典
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n请确认 configs/default.yaml 已正确创建。"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        # yaml.safe_load: 安全解析 YAML,防止 YAML 中的任意代码执行
        cfg = yaml.safe_load(f)

    return cfg


# =============================================================================
# 快速验证入口
# =============================================================================

if __name__ == "__main__":
    import time

    cfg = load_config("configs/default.yaml")
    train_loader, val_loader = build_dataloaders(cfg)

    print("\n[验证] 加载第一个训练 batch...")
    t0 = time.perf_counter()
    images, labels = next(iter(train_loader))
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  images : {images.shape}  dtype={images.dtype}")  # [1024, 3, 224, 224]
    print(f"  labels : {labels.shape}  dtype={labels.dtype}")  # [1024]
    print(f"  耗时   : {elapsed:.1f} ms")
    print("数据加载验证完成 ✅")
