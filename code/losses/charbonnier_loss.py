"""
Charbonnier Loss 模块

Charbonnier Loss (Pseudo-Huber) 是一种比 L1/L2 更稳定的损失函数
参考论文: "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"
"""

import torch
import torch.nn as nn
from .base_loss import BaseLoss


class CharbonnierLoss(BaseLoss):
    """
    Charbonnier Loss (Pseudo-Huber Loss)
    
    loss = mean(sqrt((pred - target)^2 + eps^2))
    
    特点:
        - 比 L2 对异常值更鲁棒
        - 比 L1 在零点更平滑
        - 适合超分辨率任务
    
    Args:
        eps: 平滑参数，防止梯度消失，默认为 1e-6
    
    Example:
        >>> loss_fn = CharbonnierLoss(eps=1e-6)
        >>> pred = torch.randn(2, 1, 32, 64, 64)
        >>> target = torch.randn(2, 1, 32, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 Charbonnier 损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            Charbonnier 损失值
        """
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'charbonnier',
            'eps': self.eps
        }
