# =============================================================================
# model.py — ResNet-34 图像分类模型
# 架构: ResNet-34 (BasicBlock × [3, 4, 6, 3])
# 精度: FP16 (AMP), 目标吞吐: ≈1667 img/s @ RTX 4060 Ti
# =============================================================================

import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块库 (Conv2d, Linear 等都在这里)

# =============================================================================
# 模块一: BasicBlock — 基础残差块
# =============================================================================
# ResNet-34 的核心计算单元。
# 每个 Block 包含两条并行路径:
#   主路径 (Main Path) : conv1 → bn1 → relu → conv2 → bn2
#   残差路径 (Shortcut): 恒等映射 或 1×1 降维卷积
# 两条路径的输出相加后,再过一次 ReLU,得到最终输出。
# =============================================================================


class BasicBlock(nn.Module):
    """
    BasicBlock: ResNet-34 的基本构建单元。

    参数:
        in_channels  (int): 输入特征图的通道数
        out_channels (int): 输出特征图的通道数
        stride       (int): 第一层卷积的步长。
                            stride=1 → 特征图尺寸不变 (Stage1)
                            stride=2 → 特征图尺寸减半,即下采样 (Stage2/3/4 的首块)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        # 必须调用父类 nn.Module 的构造函数,初始化 PyTorch 的内部状态
        super(BasicBlock, self).__init__()

        # -----------------------------------------------------------------
        # 主路径 — 第一层卷积块 (Conv → BN → ReLU)
        # -----------------------------------------------------------------

        # 第一层 3×3 卷积
        # - kernel_size=3: 感受野为 3×3,捕捉局部空间特征
        # - stride=stride: 如果 stride=2,此处完成下采样(特征图 H/W 减半)
        # - padding=1: 填充，保证卷积后,在 stride=1 时特征图尺寸不变
        # - bias=False: 后面紧跟 BN,BN 本身有偏置参数,因此卷积不需要 bias
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # 批归一化 (Batch Normalization)
        # - 对每个通道独立做归一化,稳定训练、加速收敛
        # - num_features=out_channels: BN 的参数数量等于通道数
        self.bn1 = nn.BatchNorm2d(out_channels)

        # ReLU 激活函数
        # - inplace=True: 直接在原始 Tensor 上修改,节省显存
        self.relu = nn.ReLU(inplace=True)

        # -----------------------------------------------------------------
        # 主路径 — 第二层卷积块 (Conv → BN)
        # 注意: 第二层步长恒定为 1,不做下采样
        # 注意: 此处故意不加 ReLU,因为要先和残差分支相加再激活
        # -----------------------------------------------------------------

        # 第二层 3×3 卷积,步长固定为 1
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # 恒定为 1,下采样只由 conv1 负责
            padding=1,
            bias=False,
        )

        # 第二层的批归一化
        self.bn2 = nn.BatchNorm2d(out_channels)

        # -----------------------------------------------------------------
        # 残差路径 (Shortcut Connection)
        # 目的: 让梯度可以绕过主路径直接反传,缓解梯度消失
        #
        # 关键判断: 主路径输出 x 的形状 与 输入 残差_x 的形状必须一致才能相加
        # 如果满足以下任一条件,形状就会不匹配:
        #   1. stride != 1  → 主路径下采样了,空间尺寸变小
        #   2. in_channels != out_channels → 通道数变化了
        # 不匹配时,用 1×1 卷积 + BN 将残差分支对齐到相同形状
        # -----------------------------------------------------------------

        if stride != 1 or in_channels != out_channels:
            # 维度不匹配:使用 1×1 卷积投影 (Projection Shortcut)
            # - kernel_size=1: 1×1 卷积只改变通道数,不改变空间感受野
            # - stride=stride: 同步主路径的下采样倍率,保持空间尺寸一致
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # 与 conv1 的 stride 保持一致
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),  # 1×1 卷积后也需要 BN
            )
        else:
            # 维度完全匹配:使用恒等映射 (Identity Shortcut)
            # nn.Identity() 是一个直通模块,输入即输出,不做任何计算
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x (Tensor): 输入特征图,形状 [N, in_channels, H, W]

        返回:
            Tensor: 输出特征图,形状 [N, out_channels, H', W']
                    当 stride=2 时 H'=H/2, W'=W/2;否则 H'=H, W'=W
        """

        # 先分离出残差分支,保存原始输入(或投影后的输入)
        residual = self.shortcut(x)  # 形状: [N, out_channels, H', W']

        # --- 主路径开始 ---

        # 第一层卷积: 提取特征,可能同时下采样
        x = self.conv1(x)  # [N, out_channels, H', W']

        # 第一层批归一化: 稳定激活值分布
        x = self.bn1(x)

        # 第一层 ReLU: 引入非线性,过滤负值
        x = self.relu(x)

        # 第二层卷积: 进一步提取特征,步长固定为 1
        x = self.conv2(x)  # [N, out_channels, H', W']

        # 第二层批归一化: 在加法之前再次归一化
        x = self.bn2(x)

        # --- 残差相加 + 最终激活 ---

        # 将主路径输出与残差分支输出逐元素相加
        # 这是 ResNet 的核心操作: 让网络学习"残差"而非完整映射
        x = x + residual  # [N, out_channels, H', W']

        # 相加之后再做 ReLU 激活
        # 注意: 第二层卷积后故意没有 ReLU,就是为了等到这里统一激活
        x = self.relu(x)

        return x


# =============================================================================
# 辅助函数: _make_stage — 堆叠多个 BasicBlock 构成一个 Stage
# =============================================================================


def _make_stage(
    in_channels: int, out_channels: int, stride: int, num_blocks: int
) -> nn.Sequential:
    """
    将多个 BasicBlock 串联成一个 Stage (残差阶段)。

    设计原则:
        - 只有第一个 Block 使用传入的 stride (可能为 2 进行下采样)
        - 后续 Block 的 stride 全部为 1,不再改变空间尺寸
        - 后续 Block 的 in_channels = out_channels (第一个 Block 已完成通道对齐)

    参数:
        in_channels  (int): 整个 Stage 的输入通道数
        out_channels (int): 整个 Stage 的输出通道数
        stride       (int): 第一个 Block 的步长
        num_blocks   (int): 该 Stage 包含的 BasicBlock 数量

    返回:
        nn.Sequential: 由多个 BasicBlock 串联而成的序列模块
    """

    blocks = []  # 用列表收集所有 Block,最后打包成 Sequential

    # 第一个 Block: 负责通道数变换 和/或 空间下采样
    # 使用传入的 stride 和 in_channels
    blocks.append(BasicBlock(in_channels, out_channels, stride=stride))

    # 后续 Block: 通道数和空间尺寸已经对齐,步长全为 1
    for _ in range(1, num_blocks):
        # in_channels 变为 out_channels,因为上一个 Block 已经完成了对齐
        blocks.append(BasicBlock(out_channels, out_channels, stride=1))

    # nn.Sequential 将 list 中的模块按顺序串联
    # 调用时: stage(x) 等价于逐个调用 block(x)
    return nn.Sequential(*blocks)


# =============================================================================
# 模块二: ResNet34 — 主干网络
# =============================================================================


class ResNet34(nn.Module):
    """
    ResNet-34 主干网络。

    架构:
        Stem (1层卷积 + 池化)
        → Stage1 (3个 BasicBlock, 64通道)
        → Stage2 (4个 BasicBlock, 128通道, 下采样)
        → Stage3 (6个 BasicBlock, 256通道, 下采样)
        → Stage4 (3个 BasicBlock, 512通道, 下采样)
        → 全局平均池化 + 展平 + FC分类

    参数:
        num_classes (int): 输出类别数。默认 1000 (ImageNet),
                           使用子集时请传入实际类别数,例如 num_classes=200
    """

    def __init__(self, num_classes: int = 1000):
        # 初始化父类 nn.Module
        super(ResNet34, self).__init__()

        # -----------------------------------------------------------------
        # 1. Stem 层 (茎干层) — 快速下采样,提取低层特征
        # 作用: 将高分辨率原始图像 224×224 快速缩小到 56×56,
        #       降低后续计算量,同时捕捉边缘、纹理等基础特征
        # -----------------------------------------------------------------

        # Stem 卷积: 7×7 大核,步长 2,将 224→112
        # - kernel_size=7: 大感受野,一次性捕捉较大范围的低层特征
        # - stride=2: 空间尺寸减半 224→112
        # - padding=3: 保证输出尺寸为 ceil(224/2) = 112,而不是 109
        # - in_channels=3: 输入为 RGB 三通道图像
        self.stem_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,  # 后接 BN,不需要卷积偏置
        )

        # Stem 批归一化
        self.stem_bn = nn.BatchNorm2d(64)

        # Stem ReLU 激活
        self.stem_relu = nn.ReLU(inplace=True)

        # Stem 最大池化: 进一步下采样,将 112→56
        # - kernel_size=3, stride=2, padding=1: 标准配置,尺寸减半
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # -----------------------------------------------------------------
        # 2. 四个残差 Stage
        # -----------------------------------------------------------------

        # Stage 1: 3 个 BasicBlock,通道数保持 64,不下采样 (stride=1)
        # 输入: [N, 64,  56, 56] → 输出: [N, 64,  56, 56]
        self.stage1 = _make_stage(
            in_channels=64,
            out_channels=64,
            stride=1,  # stride=1: 不下采样,特征图大小不变
            num_blocks=3,
        )

        # Stage 2: 4 个 BasicBlock,通道数 64→128,首块下采样 (stride=2)
        # 输入: [N, 64,  56, 56] → 输出: [N, 128, 28, 28]
        self.stage2 = _make_stage(
            in_channels=64,
            out_channels=128,
            stride=2,  # stride=2: 首块将 56×56 下采样为 28×28
            num_blocks=4,
        )

        # Stage 3: 6 个 BasicBlock,通道数 128→256,首块下采样 (stride=2)
        # 这是 ResNet-34 中最深的一个 Stage (6个Block = 12层卷积)
        # 输入: [N, 128, 28, 28] → 输出: [N, 256, 14, 14]
        self.stage3 = _make_stage(
            in_channels=128,
            out_channels=256,
            stride=2,  # stride=2: 首块将 28×28 下采样为 14×14
            num_blocks=6,
        )

        # Stage 4: 3 个 BasicBlock,通道数 256→512,首块下采样 (stride=2)
        # 输入: [N, 256, 14, 14] → 输出: [N, 512, 7, 7]
        self.stage4 = _make_stage(
            in_channels=256,
            out_channels=512,
            stride=2,  # stride=2: 首块将 14×14 下采样为 7×7
            num_blocks=3,
        )

        # -----------------------------------------------------------------
        # 3. 分类头 (Classification Head)
        # -----------------------------------------------------------------

        # 全局自适应平均池化: 将任意尺寸的特征图压缩为 1×1
        # - output_size=(1, 1): 无论输入是 7×7 还是其他尺寸,输出都是 1×1
        # - 效果: [N, 512, 7, 7] → [N, 512, 1, 1]
        # - 优势: 模型对输入尺寸不敏感,天然防止全连接层过拟合
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 展平层: 将 [N, 512, 1, 1] 的 4D Tensor 压缩为 [N, 512] 的 2D Tensor
        # start_dim=1: 从第 1 维开始展平(保留 batch 维度 N 不动)
        self.flatten = nn.Flatten(start_dim=1)

        # 全连接分类层
        # - in_features=512:  全局池化后每张图像的特征向量长度
        # - out_features=num_classes: 输出每个类别的原始分数 (logits)
        # 注意: 此处不加 Softmax,因为 PyTorch 的 CrossEntropyLoss 内部已包含
        self.fc_layer = nn.Linear(in_features=512, out_features=num_classes)

        # -----------------------------------------------------------------
        # 4. 权重初始化 (Weight Initialization)
        # 良好的初始化可以加快收敛速度,避免梯度爆炸/消失
        # -----------------------------------------------------------------
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化所有模块的权重
        遍历所有子模块,按类型分别初始化权重。
        这是 PyTorch 中标准的权重初始化写法。
        """
        for module in self.modules():
            # Conv2d: 使用 Kaiming (He) 正态初始化
            # - mode='fan_out': 考虑前向传播方向的神经元数量
            # - nonlinearity='relu': 针对 ReLU 激活函数校正方差
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

            # BatchNorm2d: 权重初始化为 1,偏置初始化为 0
            # 即初始时 BN 不做任何缩放/平移,相当于直通
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)  # gamma = 1
                nn.init.constant_(module.bias, 0)  # beta  = 0

            # Linear: 使用正态分布初始化权重,偏置初始化为 0
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            images (Tensor): 输入图像批次,形状 [N, 3, 224, 224]
                             N = Batch Size (推荐 1024)
                             数据类型应为 torch.float32 (AMP 会自动转 FP16)

        返回:
            Tensor: 分类 logits,形状 [N, num_classes]
                    每个值是对应类别的原始得分,未经 Softmax 归一化
        """

        # -----------------------------------------------------------------
        # FP16 精度转换
        # 目的: 激活 RTX 4060 Ti 的 Tensor Cores (专为 FP16 矩阵运算优化)
        # 注意: 实际项目中推荐在训练循环外用 torch.cuda.amp.autocast() 统一管理
        #       此处 half() 写在模型内仅作为架构完整性展示
        #       如果使用 autocast,可以删除下面这行
        # -----------------------------------------------------------------
        images = images.half()  # torch.float32 → torch.float16

        # -----------------------------------------------------------------
        # Stem 阶段: 224×224 → 56×56
        # -----------------------------------------------------------------

        # 7×7 卷积: [N, 3, 224, 224] → [N, 64, 112, 112]
        x = self.stem_conv(images)

        # 批归一化: 归一化 64 个通道的激活值
        x = self.stem_bn(x)

        # ReLU 激活: 过滤负值,引入非线性
        x = self.stem_relu(x)

        # 最大池化: [N, 64, 112, 112] → [N, 64, 56, 56]
        x = self.stem_pool(x)

        # -----------------------------------------------------------------
        # 残差 Stage 阶段: 逐步提取高层语义特征
        # -----------------------------------------------------------------

        # Stage 1: [N, 64, 56, 56] → [N, 64,  56, 56]  (不下采样)
        x = self.stage1(x)

        # Stage 2: [N, 64, 56, 56] → [N, 128, 28, 28]  (下采样 ×2)
        x = self.stage2(x)

        # Stage 3: [N, 128, 28, 28] → [N, 256, 14, 14] (下采样 ×2)
        x = self.stage3(x)

        # Stage 4: [N, 256, 14, 14] → [N, 512, 7, 7]   (下采样 ×2)
        x = self.stage4(x)

        # -----------------------------------------------------------------
        # 分类头: 特征聚合 → 映射为类别分数
        # -----------------------------------------------------------------

        # 全局平均池化: [N, 512, 7, 7] → [N, 512, 1, 1]
        # 将每个通道的 7×7=49 个空间值平均成 1 个值
        x = self.avg_pool(x)

        # 展平: [N, 512, 1, 1] → [N, 512]
        # 为全连接层准备一维特征向量
        x = self.flatten(x)

        # 全连接分类: [N, 512] → [N, num_classes]
        # 输出每个类别的 logit (原始得分)
        logits = self.fc_layer(x)

        return logits  # 返回 [N, 1000] 的 FP16 logits


# =============================================================================
# 工厂函数: 快速实例化模型
# =============================================================================


def build_resnet34(num_classes: int = 1000, pretrained: bool = False) -> ResNet34:
    """
    构建 ResNet-34 模型实例。

    参数:
        num_classes (int) : 分类类别数,默认 1000 (ImageNet)
        pretrained  (bool): 是否加载预训练权重 (此处为占位符,实际需接入权重路径)

    返回:
        ResNet34: 初始化完毕的模型
    """
    # 实例化模型
    model = ResNet34(num_classes=num_classes)

    if pretrained:
        # 实际项目中在这里加载权重,例如:
        # state_dict = torch.load("resnet34_imagenet.pth")
        # model.load_state_dict(state_dict)
        raise NotImplementedError("预训练权重加载尚未实现,请手动指定权重路径。")

    return model


# =============================================================================
# 快速验证: 直接运行此文件时,检查模型结构和输出形状
# =============================================================================

if __name__ == "__main__":
    # 检查是否有可用 GPU,有则使用 CUDA,否则退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 实例化模型并移动到目标设备
    model = build_resnet34(num_classes=1000).to(device)

    # 构造一个随机输入 Tensor,模拟推理数据
    # 形状: [Batch=4, Channel=3, H=224, W=224]
    # 注意: 验证时用小 Batch=4,实际训练时使用 Batch=1024
    dummy_input = torch.randn(4, 3, 224, 224, device=device)

    # 使用 torch.no_grad() 关闭梯度计算,节省显存 (推理/验证时使用)
    with torch.no_grad():
        output = model(dummy_input)  # 前向传播

    # 打印输出形状: 预期为 [4, 1000]
    print(f"输出形状 (logits): {output.shape}")
    # 预期输出: 输出形状 (logits): torch.Size([4, 1000])

    # 统计模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 格式化为百万 (M) 单位,ResNet-34 约 21.8M 参数
    print(f"总参数量: {total_params / 1e6:.2f}M")
    # 预期输出: 总参数量: 21.80M
