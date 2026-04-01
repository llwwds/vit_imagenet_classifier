# =============================================================================
# model.py — Vision Transformer (ViT) 图像分类模型
# 架构: ViT (Patch Embedding + Transformer Encoder + Classification Head)
# 参考: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
# =============================================================================

import torch
import torch.nn as nn


# =============================================================================
# 模块一: PatchEmbedding — 图像分块嵌入
# =============================================================================
# 将输入图像切分为固定大小的 patch,然后通过线性投影映射到 embedding 空间。
# 例如: 224×224 图像,patch_size=16 → 14×14 = 196 个 patch
# 每个 patch 被展平后投影为一个 embed_dim 维的向量
# =============================================================================


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层。

    参数:
        img_size   (int): 输入图像的边长 (假设正方形),默认 224
        patch_size (int): 每个 patch 的边长,默认 16
        in_channels(int): 输入图像通道数,默认 3 (RGB)
        embed_dim  (int): 嵌入维度,默认 768
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        # 每条边上的 patch 数量
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积实现分块 + 线性投影 (等价于手动切块再乘权重矩阵,但更高效)
        # kernel_size = stride = patch_size → 无重叠地将图像切成 patch
        # out_channels = embed_dim → 每个 patch 投影为 embed_dim 维向量
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入图像,形状 [B, C, H, W]

        返回:
            Tensor: patch embeddings,形状 [B, num_patches, embed_dim]
        """
        # [B, C, H, W] → [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # [B, embed_dim, H/P, W/P] → [B, embed_dim, num_patches]
        x = x.flatten(2)
        # [B, embed_dim, num_patches] → [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


# =============================================================================
# 模块二: Attention — 多头自注意力
# =============================================================================
# Transformer 的核心: 让每个 patch 能够"关注"所有其他 patch 的信息
# 多头机制: 将注意力分成多个独立的"头",每个头关注不同的特征子空间
# =============================================================================


class Attention(nn.Module):
    """
    多头自注意力 (Multi-Head Self-Attention)。

    参数:
        embed_dim  (int):   输入/输出的嵌入维度
        num_heads  (int):   注意力头的数量
        attn_drop  (float): 注意力权重的 Dropout 概率
        proj_drop  (float): 输出投影的 Dropout 概率
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        # 每个头的维度 = 总维度 / 头数
        self.head_dim = embed_dim // num_heads
        # 缩放因子: 1/sqrt(d_k),防止点积值过大导致 softmax 梯度消失
        self.scale = self.head_dim ** -0.5

        # Q, K, V 三个投影矩阵合并为一个线性层,提高计算效率
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影: 将多头拼接后的结果映射回 embed_dim 维
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入,形状 [B, N, D]  (B=batch, N=序列长度, D=embed_dim)

        返回:
            Tensor: 输出,形状 [B, N, D]
        """
        B, N, D = x.shape

        # 一次性计算 Q, K, V: [B, N, D] → [B, N, 3*D] → [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # 重排维度: [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 各自形状: [B, num_heads, N, head_dim]

        # 计算注意力分数: Q·K^T / sqrt(d_k)
        # [B, heads, N, head_dim] @ [B, heads, head_dim, N] → [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax 归一化: 将分数转换为概率分布 (每行和为 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权聚合 V: [B, heads, N, N] @ [B, heads, N, head_dim] → [B, heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# 模块三: MLP — 前馈网络
# =============================================================================
# Transformer Block 中的第二个子层: 两层全连接 + GELU 激活
# 先扩展维度 (embed_dim → mlp_dim),再压缩回来 (mlp_dim → embed_dim)
# =============================================================================


class MLP(nn.Module):
    """
    前馈网络 (Feed-Forward Network)。

    参数:
        embed_dim  (int):   输入/输出维度
        mlp_ratio  (float): 隐藏层维度相对于 embed_dim 的倍数,默认 4.0
        drop       (float): Dropout 概率
    """

    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()       # GELU 激活: 比 ReLU 更平滑,Transformer 标准选择
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入,形状 [B, N, D]

        返回:
            Tensor: 输出,形状 [B, N, D]
        """
        x = self.fc1(x)        # [B, N, D] → [B, N, hidden_dim]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)        # [B, N, hidden_dim] → [B, N, D]
        x = self.drop(x)
        return x


# =============================================================================
# 模块四: TransformerBlock — Transformer 编码器块
# =============================================================================
# 一个标准的 Transformer 编码器层:
#   LayerNorm → Multi-Head Attention → 残差连接
#   → LayerNorm → MLP → 残差连接
# 使用 Pre-Norm 结构 (LayerNorm 在 Attention/MLP 之前),训练更稳定
# =============================================================================


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块 (Pre-Norm 结构)。

    参数:
        embed_dim  (int):   嵌入维度
        num_heads  (int):   注意力头数
        mlp_ratio  (float): MLP 隐藏层倍数
        attn_drop  (float): 注意力 Dropout
        proj_drop  (float): 投影 Dropout
        drop_path  (float): DropPath (随机深度) 概率
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop=proj_drop,
        )

        # DropPath (Stochastic Depth): 以一定概率跳过整个 block
        # 训练时随机"关闭"某些层,类似 Dropout 但作用于整个残差路径
        # 有效防止深层 Transformer 过拟合
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入,形状 [B, N, D]

        返回:
            Tensor: 输出,形状 [B, N, D]
        """
        # Pre-Norm + Attention + 残差
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Pre-Norm + MLP + 残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# 辅助模块: DropPath (Stochastic Depth)
# =============================================================================


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth): 在训练时以概率 drop_prob 将整条残差路径置零。

    参数:
        drop_prob (float): 丢弃概率
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # 生成与 batch 维度对齐的随机 mask (每个样本独立决定是否丢弃)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        # 缩放输出以保持期望值不变
        output = x / keep_prob * random_tensor
        return output


# =============================================================================
# 模块五: VisionTransformer — ViT 主模型
# =============================================================================


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 主模型。

    架构:
        Patch Embedding + [CLS] Token + Position Embedding
        → N × TransformerBlock
        → LayerNorm → [CLS] Token → FC 分类头

    参数:
        img_size    (int):   输入图像边长,默认 224
        patch_size  (int):   patch 边长,默认 16
        in_channels (int):   图像通道数,默认 3
        num_classes (int):   分类类别数,默认 1000
        embed_dim   (int):   嵌入维度,默认 768
        depth       (int):   Transformer Block 层数,默认 12
        num_heads   (int):   注意力头数,默认 12
        mlp_ratio   (float): MLP 隐藏层倍数,默认 4.0
        attn_drop   (float): 注意力 Dropout
        proj_drop   (float): 投影/MLP Dropout
        drop_path   (float): DropPath 最大概率 (线性递增)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ── Patch Embedding ───────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ── [CLS] Token ──────────────────────────────────────────────────────
        # 一个可学习的向量,拼接在 patch 序列的最前面
        # 经过 Transformer 编码后,CLS token 的输出聚合了全局信息,用于分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Position Embedding ────────────────────────────────────────────────
        # 可学习的位置编码,为每个 token (CLS + patches) 提供位置信息
        # Transformer 本身不具备位置感知能力,必须显式注入位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(proj_drop)

        # ── Transformer Encoder ───────────────────────────────────────────────
        # DropPath 概率从 0 线性递增到 drop_path (浅层不丢弃,深层多丢弃)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        # ── 分类头 ────────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # ── 权重初始化 ────────────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """初始化权重: 遵循 ViT 原论文的初始化方案"""
        # Position embedding 和 CLS token: 截断正态分布
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 遍历所有子模块
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入图像,形状 [B, 3, H, W]

        返回:
            Tensor: 分类 logits,形状 [B, num_classes]
        """
        B = x.shape[0]

        # ── Patch Embedding ───────────────────────────────────────────────────
        # [B, 3, 224, 224] → [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        # ── 拼接 [CLS] Token ─────────────────────────────────────────────────
        # cls_token: [1, 1, D] → 扩展为 [B, 1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 拼接: [B, num_patches, D] → [B, 1 + num_patches, D]
        x = torch.cat([cls_tokens, x], dim=1)

        # ── 加上位置编码 ──────────────────────────────────────────────────────
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # ── Transformer Encoder ───────────────────────────────────────────────
        # 通过 depth 层 TransformerBlock
        x = self.blocks(x)

        # ── 分类头 ────────────────────────────────────────────────────────────
        # 取 [CLS] token 的输出 (序列第 0 个位置)
        x = self.norm(x[:, 0])
        # [B, embed_dim] → [B, num_classes]
        logits = self.head(x)

        return logits


# =============================================================================
# 工厂函数: 快速实例化模型
# =============================================================================


def build_vit(cfg: dict) -> VisionTransformer:
    """
    根据配置字典构建 ViT 模型。

    参数:
        cfg (dict): 从 default.yaml 解析的配置字典

    返回:
        VisionTransformer: 初始化完毕的模型
    """
    model_cfg = cfg["model"]
    model = VisionTransformer(
        img_size=model_cfg.get("img_size", 224),
        patch_size=model_cfg.get("patch_size", 16),
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 1000),
        embed_dim=model_cfg.get("embed_dim", 768),
        depth=model_cfg.get("depth", 12),
        num_heads=model_cfg.get("num_heads", 12),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        attn_drop=model_cfg.get("attn_drop", 0.0),
        proj_drop=model_cfg.get("proj_drop", 0.0),
        drop_path=model_cfg.get("drop_path", 0.1),
    )
    return model


# =============================================================================
# 快速验证: 直接运行此文件时,检查模型结构和输出形状
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 使用默认参数构建 ViT-Base
    cfg = {
        "model": {
            "img_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "drop_path": 0.1,
        }
    }
    model = build_vit(cfg).to(device)

    # 构造随机输入: [Batch=4, Channel=3, H=224, W=224]
    dummy_input = torch.randn(4, 3, 224, 224, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"输出形状 (logits): {output.shape}")
    # 预期: torch.Size([4, 1000])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")
    # ViT-Base 约 86M 参数
