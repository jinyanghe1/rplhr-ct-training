"""
SSIM Loss 模块

提供 3D SSIM (Structural Similarity Index Measure) 损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class SSIM3D(nn.Module):
    """
    3D SSIM (Structural Similarity Index Measure)
    
    用于3D体积数据的结构相似性计算
    
    Args:
        window_size: 高斯窗口大小，默认为 7
        size_average: 是否对结果取平均，默认为 True
        channel: 输入通道数，默认为 1
    """
    
    def __init__(self, window_size: int = 7, size_average: bool = True, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """创建3D高斯核窗口"""
        _1D_window = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        _1D_window = torch.exp(-(_1D_window ** 2) / (2 * 1.5 ** 2))
        _1D_window = _1D_window / _1D_window.sum()
        
        # 3D高斯核 = 1D * 1D * 1D
        _2D_window = _1D_window.unsqueeze(1) * _1D_window.unsqueeze(0)
        _3D_window = _2D_window.unsqueeze(2) * _1D_window.unsqueeze(0).unsqueeze(0)
        _3D_window = _3D_window.unsqueeze(0).unsqueeze(0)
        
        window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        计算 SSIM 值
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
        
        Returns:
            SSIM 值，范围 [0, 1]，越接近1表示越相似
        """
        device = img1.device
        window = self.window.to(device)
        
        # 确保通道数匹配
        if img1.size(1) != self.channel:
            window = window.repeat(img1.size(1), 1, 1, 1, 1)
        
        mu1 = F.conv3d(img1, window, padding=self.window_size//2, groups=img1.size(1))
        mu2 = F.conv3d(img2, window, padding=self.window_size//2, groups=img2.size(1))
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv3d(img1 * img1, window, padding=self.window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=self.window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1).mean(1)


class SSIMLoss(BaseLoss):
    """
    SSIM Loss (基于结构相似性的损失)
    
    loss = 1 - SSIM(pred, target)
    
    值越小表示预测与目标越相似
    
    Args:
        window_size: 高斯窗口大小，默认为 7
        channel: 输入通道数，默认为 1
    
    Example:
        >>> loss_fn = SSIMLoss(window_size=7)
        >>> pred = torch.randn(2, 1, 32, 64, 64)
        >>> target = torch.randn(2, 1, 32, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, window_size: int = 7, channel: int = 1):
        super().__init__()
        self.ssim = SSIM3D(window_size=window_size, channel=channel)
        self.window_size = window_size
        self.channel = channel
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 SSIM 损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            SSIM 损失值 (1 - SSIM)
        """
        ssim_value = self.ssim(pred, target)
        return 1 - ssim_value
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'ssim',
            'window_size': self.window_size,
            'channel': self.channel
        }
