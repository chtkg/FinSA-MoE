# 实现了一个用于混合专家模型（Mixture of Experts, MoE）的高效轻量级路由器

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EfficientRouter(nn.Module):
    """高效轻量级路由器"""
    # 初始化参数
    def __init__(
            self,
            input_dim: int = 4096,  # 输入特征维度
            num_experts: int = 2,   # 专家数量
            hidden_dim: int = 64,   # 路由网络隐藏层维度
            dropout: float = 0.1,   # Dropout概率
            temperature_init: float = 1.0  # 温度参数初始值
    ):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim

        # 两层MLP路由网络
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 全连接层，第一层将输入维度转换到隐藏维度
            nn.LayerNorm(hidden_dim),    # 层归一化，稳定训练
            nn.ReLU(),         # 激活函数，引入非线性
            nn.Dropout(dropout),    # 防止过拟合
            nn.Linear(hidden_dim, num_experts)   # 全连接层第二层输出专家数量的logits
        )

        # 可学习的温度参数，用于控制softmax分布的平滑程度，温度越高，分布越平滑；温度越低，分布越尖锐
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        # 专家使用统计
        self.register_buffer('expert_usage', torch.zeros(num_experts))   # 记录每个专家的使用次数
        self.register_buffer('total_routed', torch.tensor(0.0))    #  记录总的路由次数

    def forward(
            self,
            hidden_states: torch.Tensor,
            return_router_logits: bool = False,
            force_soft: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 如果输入是3维张量（batch_size, seq_len, hidden_dim），提取[CLS] token（第一个token），否则直接使用输入
        if hidden_states.dim() == 3:
            router_input = hidden_states[:, 0, :]
        else:
            router_input = hidden_states

        router_logits = self.router(router_input) / self.temperature.clamp(min=0.1)
        
        if self.training or force_soft:
            routing_weights = F.softmax(router_logits, dim=-1)
            # 注意：force_soft 时通常处于 eval()，不会更新 usage 统计也无所谓
        else:
            routing_weights = F.one_hot(
                router_logits.argmax(dim=-1),
                num_classes=self.num_experts
            ).float()

        if return_router_logits:
            return routing_weights, router_logits
        return routing_weights, None

    def compute_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        batch_usage = routing_weights.mean(dim=0)
        uniform_distribution = 1.0 / self.num_experts
        target = torch.full_like(batch_usage, uniform_distribution)
        balance_loss = F.kl_div(
            torch.log(batch_usage + 1e-8),
            target,
            reduction='batchmean'
        )
        return balance_loss

    def compute_route_loss(self, router_logits: torch.Tensor, expert_labels: torch.Tensor) -> torch.Tensor:
        """
        L_route：监督路由器根据域标签选择对应专家的损失
        router_logits: (batch_size, num_experts)
        expert_labels: (batch_size,) 取值范围 [0, num_experts-1]，例如 0=新闻, 1=股评
        """
        return F.cross_entropy(router_logits, expert_labels)