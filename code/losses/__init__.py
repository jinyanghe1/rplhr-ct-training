"""
模块化 Loss 函数系统

提供统一的 Loss 函数接口，支持通过配置选择不同的 Loss 函数。

Usage:
    from losses import get_loss
    
    # 通过配置文件创建 loss
    loss_fn = get_loss('config/loss_configs/loss_l1.txt')
    
    # 或者直接通过类型创建
    from losses import LossFactory
    loss_fn = LossFactory.create_loss('l1')
    
    # 使用
    loss = loss_fn(pred, target)
"""

from .base_loss import BaseLoss
from .l1_loss import L1Loss
from .eagle3d_loss import EAGLELoss3D, Sobel3D
from .charbonnier_loss import CharbonnierLoss
from .ssim_loss import SSIMLoss, SSIM3D
from .combined_loss import CombinedLoss, L1SSIMLoss3D
from .loss_factory import LossFactory, get_loss

__all__ = [
    # 基础类
    'BaseLoss',
    # Loss 函数
    'L1Loss',
    'EAGLELoss3D',
    'Sobel3D',
    'CharbonnierLoss',
    'SSIMLoss',
    'SSIM3D',
    'CombinedLoss',
    'L1SSIMLoss3D',
    # 工厂函数
    'LossFactory',
    'get_loss',
]
