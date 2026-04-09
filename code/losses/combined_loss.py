"""
组合 Loss 模块

提供多种 Loss 的组合，如 L1 + SSIM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss
from .ssim_loss import SSIM3D


class CombinedLoss(BaseLoss):
    """
    组合 Loss (L1 + SSIM)
    
    loss = L1(pred, target) + alpha * (1 - SSIM(pred, target))
    
    结合像素级损失和结构相似性损失，广泛用于超分辨率任务
    
    Args:
        alpha: SSIM 损失权重，默认为 0.1
        ssim_window_size: SSIM 窗口大小，默认为 7
    
    Example:
        >>> loss_fn = CombinedLoss(alpha=0.1)
        >>> pred = torch.randn(2, 1, 32, 64, 64)
        >>> target = torch.randn(2, 1, 32, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, alpha: float = 0.1, ssim_window_size: int = 7):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIM3D(window_size=ssim_window_size, channel=1)
        self.ssim_window_size = ssim_window_size
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            组合损失值
        """
        # L1 损失
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM 损失
        ssim_value = self.ssim(pred, target)
        ssim_loss = 1 - ssim_value
        
        return l1_loss + self.alpha * ssim_loss
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'l1_ssim',
            'alpha': self.alpha,
            'ssim_window_size': self.ssim_window_size
        }


class L1SSIMLoss3D(BaseLoss):
    """
    L1 + SSIM 组合 Loss (3D版本，与 CombinedLoss 相同)
    
    保留此别名以兼容现有代码
    """
    
    def __init__(self, alpha: float = 0.1, ssim_window_size: int = 7):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIM3D(window_size=ssim_window_size, channel=1)
        self.ssim_window_size = ssim_window_size
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = F.l1_loss(pred, target)
        ssim_value = self.ssim(pred, target)
        ssim_loss = 1 - ssim_value
        return l1_loss + self.alpha * ssim_loss
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'l1_ssim_3d',
            'alpha': self.alpha,
            'ssim_window_size': self.ssim_window_size
        }


class WeightedLoss(BaseLoss):
    """
    多 Loss 加权组合
    
    允许组合多个不同的 Loss 函数
    
    Args:
        losses: Loss 函数列表
        weights: 对应的权重列表
    
    Example:
        >>> loss1 = L1Loss()
        >>> loss2 = SSIMLoss()
        >>> loss_fn = WeightedLoss([loss1, loss2], [1.0, 0.5])
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, losses: list, weights: list):
        super().__init__()
        assert len(losses) == len(weights), "Loss 数量和权重数量必须相同"
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'weighted',
            'weights': self.weights,
            'sub_losses': [loss.get_config() for loss in self.losses]
        }
