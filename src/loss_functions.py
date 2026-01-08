"""
Class-Balanced Loss - 基于有效样本数的重加权
Focal Loss - 难样本挖掘
Label Smoothing - 标签平滑正则化
"""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveBalancedLoss(nn.Module):
    """自适应平衡损失函数"""

    def __init__(self, samples_per_class, beta=0.99, gamma=2.0, smoothing=0.1):
        """
        参数说明：samples_per_class: 每个类别的样本数量（列表/数组）
        beta: 有效样本数的衰减因子（默认0.99）
        gamma: Focal Loss的聚焦参数（默认2.0）
        smoothing: 标签平滑的强度（默认0.1）
        """
        super().__init__()

        # effective_num计算每个类的"有效样本数"
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)  # 有效样本数权重计算，权重与有效样本数成反比，样本少的类获得更高权重
        weights = weights / weights.sum() * len(weights)  # 最后归一化使权重平均值为1
        self.register_buffer('weights', torch.FloatTensor(weights))  # 使用register_buffer将权重注册为模型的持久状态，但不参与梯度计算。

        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = len(samples_per_class)

    def forward(self, inputs, targets):
        # 标签平滑，防止模型过度自信, one-hot编码
        """
        例子：原始标签（未平滑）:[0,1,0] # 0%积极，100%消极，0%中性
        平滑后的标签： [0.033, 0.9, 0.033]   # 3.3%可能积极，90%可能消极，3.3%可能中性；现实中，人类标注也不是100%准确
        """
        confidence = 1.0 - self.smoothing
        smooth_targets = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), confidence
        )
        smooth_targets += self.smoothing / self.num_classes

        # 计算交叉熵损失
        log_probs = torch.log_softmax(inputs, dim=-1)
        # 手动计算平滑标签的交叉熵损失，而不是使用nn.CrossEntropyLoss
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        #  Focal Loss调制
        """
        Focal Loss的核心思想：
            当模型预测正确且置信度高时（p_t接近1），focal_weight接近0
            当模型预测错误时（p_t接近0），focal_weight接近1
            这让模型更关注难分类的样本
        """
        p_t = torch.exp(-ce_loss)  # 预测正确类的概率
        focal_weight = (1 - p_t) ** self.gamma

        # 类别权重
        batch_weights = self.weights[targets]

        # 组合损失
        loss = batch_weights * focal_weight * ce_loss

        return loss.mean()