# ViT ImageNet 图像分类

基于 Vision Transformer (ViT) 的 ImageNet 图像分类项目。从零实现 ViT 架构,支持 FP16 混合精度训练。

## 项目结构

```
vit_imagenet_classifier/
├── configs/
│   └── default.yaml         # 模型与训练超参数配置
├── dataset/
│   └── imagenet/            # 数据集 (ImageFolder 格式)
│       ├── train/           # 训练集 (按类别子文件夹组织)
│       ├── val/             # 验证集 (按类别子文件夹组织)
│       └── synset_words.txt # 类别标签映射 (可选)
├── utils/
│   ├── logger.py            # 日志记录
│   ├── metrics.py           # Top-K 准确率计算
│   └── checkpointing.py     # 断点续训与权重保存
├── model.py                 # ViT 模型 (完整实现)
├── data_processing.py       # 数据预处理与加载
├── train.py                 # 训练脚本
├── test.py                  # 测试/推理脚本
└── requirements.txt         # 依赖列表
```

## 环境配置

**系统要求:** Ubuntu 24 (WSL2), Python 3.10+, CUDA GPU (建议 16GB+ 显存)

```bash
# 1. 激活 conda 环境
conda activate torch

# 2. 安装 PyTorch (根据 CUDA 版本选择, 以 CUDA 12.1 为例)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install -r requirements.txt
```

## 数据集准备

将 ImageNet 数据集组织为如下目录结构:

```
dataset/imagenet/
├── train/
│   ├── n01440764/    # 类别文件夹
│   │   ├── img1.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

如使用 ImageNet 子集 (如 ImageNet-100),修改 `configs/default.yaml` 中的 `model.num_classes`。

## 快速开始

### 训练

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置
python train.py --config configs/default.yaml
```

### 测试 (验证集评估)

```bash
python test.py --checkpoint checkpoints/best_model.pth
```

### 单图推理

```bash
python test.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

## 模型架构

默认配置为 ViT-Base/16:

| 参数 | 值 |
|------|------|
| Patch Size | 16x16 |
| Embedding Dim | 768 |
| Depth (层数) | 12 |
| Attention Heads | 12 |
| MLP Ratio | 4.0 |
| 参数量 | ~86M |

## 性能记录

| 模型 | 数据集 | Top-1 | Top-5 | 备注 |
|------|--------|-------|-------|------|
| ViT-Base/16 | ImageNet-1K | - | - | 待训练 |
