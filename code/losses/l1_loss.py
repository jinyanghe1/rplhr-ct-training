"""
L1 Loss 模块

提供标准的 L1 损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class L1Loss(BaseLoss):
    """
    L1 Loss (Mean Absolute Error)
    
    loss = mean(|pred - target|)
    
    Args:
        reduction: 损失归约方式，可选 'mean', 'sum', 'none'
    
    Example:
        >>> loss_fn = L1Loss()
        >>> pred = torch.randn(2, 1, 32, 64, 64)
        >>> target = torch.randn(2, 1, 32, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 L1 损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            L1 损失值
        """
        return F.l1_loss(pred, target, reduction=self.reduction)
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'l1',
            'reduction': self.reduction
        }


class SmoothL1Loss(BaseLoss):
    """
    Smooth L1 Loss (Huber Loss)
    
    对异常值比 L1 更鲁棒，比 L2 更稳定
    
    Args:
        beta: 转换阈值，默认为 1.0
        reduction: 损失归约方式
    
    Example:
        >>> loss_fn = SmoothL1Loss(beta=0.5)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'smooth_l1',
            'beta': self.beta,
            'reduction': self.reduction
        }
