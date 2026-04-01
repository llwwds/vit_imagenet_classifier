# =============================================================================
# utils/metrics.py — 性能指标计算模块
# 包含: AverageMeter (滑动平均计数器) 和 accuracy (Top-K 准确率计算)
# =============================================================================

import torch


class AverageMeter:
    """
    滑动平均计数器,用于累积和跟踪训练/验证过程中的指标 (loss, accuracy 等)。

    用法:
        meter = AverageMeter("Loss")
        for batch in loader:
            meter.update(loss_value, batch_size)
        print(meter.avg)  # 整个 epoch 的加权平均值
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        """重置所有计数器"""
        self.val = 0.0    # 最近一次更新的值
        self.avg = 0.0    # 加权平均值
        self.sum = 0.0    # 加权总和
        self.count = 0    # 总样本数

    def update(self, val: float, n: int = 1):
        """
        更新计数器。

        参数:
            val (float): 当前 batch 的指标值
            n   (int):   当前 batch 的样本数 (用于加权平均)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """
    计算 Top-K 准确率。

    参数:
        output (Tensor): 模型输出 logits,形状 [N, num_classes]
        target (Tensor): 真实标签,形状 [N]
        topk   (tuple):  需要计算的 K 值,如 (1, 5) 表示同时计算 Top-1 和 Top-5

    返回:
        list[float]: 各 K 值对应的准确率 (百分比形式,如 75.3)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 取出每个样本的 Top-maxk 预测类别的索引
        # pred 形状: [N, maxk]
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)

        # 转置为 [maxk, N],方便与 target 逐行比较
        pred = pred.t()

        # correct[i][j] = True 表示第 j 个样本的 Top-(i+1) 预测命中了真实标签
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            # 取前 k 行,统计命中的样本数
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # 转为百分比
            acc = correct_k.mul_(100.0 / batch_size).item()
            results.append(acc)

        return results
