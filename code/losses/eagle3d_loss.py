"""
EAGLE3D Loss 模块

3D版本 EAGLE Loss (Edge-Aware Gradient Local Enhancement)
用于3D CT医学影像超分辨率，强调边缘保持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss


class Sobel3D(nn.Module):
    """
    3D Sobel 梯度算子
    
    用于计算3D体积数据的边缘梯度
    """
    
    def __init__(self):
        super().__init__()
        # 3D Sobel核 - x方向
        sobel_x = torch.tensor([
            [[[1,2,1], [2,4,2], [1,2,1]],
             [[0,0,0], [0,0,0], [0,0,0]],
             [[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        
        # 3D Sobel核 - y方向
        sobel_y = torch.tensor([
            [[[1,2,1], [0,0,0], [-1,-2,-1]],
             [[2,4,2], [0,0,0], [-2,-4,-2]],
             [[1,2,1], [0,0,0], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        
        # 3D Sobel核 - z方向 (层间梯度)
        sobel_z = torch.tensor([
            [[[1,0,-1], [2,0,-2], [1,0,-1]],
             [[2,0,-2], [4,0,-4], [2,0,-2]],
             [[1,0,-1], [2,0,-2], [1,0,-1]]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        
        self.register_buffer('kernel_x', sobel_x)
        self.register_buffer('kernel_y', sobel_y)
        self.register_buffer('kernel_z', sobel_z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算3D梯度幅值
        
        Args:
            x: 输入张量，形状 (B, C, D, H, W)
        
        Returns:
            梯度幅值，形状与输入相同
        """
        B, C, D, H, W = x.shape
        
        # 动态获取设备并移动kernels到正确设备
        device = x.device
        kernel_x = self.kernel_x.to(device)
        kernel_y = self.kernel_y.to(device)
        kernel_z = self.kernel_z.to(device)
        
        # 扩展到对应channel数
        kernel_x = kernel_x.repeat(C, 1, 1, 1, 1)
        kernel_y = kernel_y.repeat(C, 1, 1, 1, 1)
        kernel_z = kernel_z.repeat(C, 1, 1, 1, 1)
        
        # 3D卷积
        grad_x = F.conv3d(x, kernel_x, padding=1, groups=C)
        grad_y = F.conv3d(x, kernel_y, padding=1, groups=C)
        grad_z = F.conv3d(x, kernel_z, padding=1, groups=C)
        
        # 梯度幅值 (L2 norm)
        return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)


class EAGLELoss3D(BaseLoss):
    """
    3D版本 EAGLE Loss (Edge-Aware Gradient Local Enhancement)
    
    用于3D CT医学影像超分辨率，结合 L1 损失和边缘梯度损失
    
    loss = L1(pred, target) + alpha * L1(Sobel3D(pred), Sobel3D(target))
    
    Args:
        alpha: 边缘损失权重，默认为 0.1
    
    Example:
        >>> loss_fn = EAGLELoss3D(alpha=0.1)
        >>> pred = torch.randn(2, 1, 32, 64, 64)
        >>> target = torch.randn(2, 1, 32, 64, 64)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sobel3d = Sobel3D()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 EAGLE3D 损失
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            EAGLE3D 损失值
        """
        # L1 损失
        l1 = F.l1_loss(pred, target)
        
        # 边缘梯度损失
        pred_grad = self.sobel3d(pred)
        target_grad = self.sobel3d(target)
        edge = F.l1_loss(pred_grad, target_grad)
        
        return l1 + self.alpha * edge
    
    def get_config(self) -> dict:
        return {
            'loss_type': 'eagle3d',
            'alpha': self.alpha
        }
