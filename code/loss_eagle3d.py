import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel3D(nn.Module):
    """3D Sobel梯度算子 - 用于计算3D体积数据的边缘梯度"""
    def __init__(self):
        super().__init__()
        # 3D Sobel核 - x方向
        sobel_x = torch.tensor([
            [[[1,2,1], [2,4,2], [1,2,1]],
             [[0,0,0], [0,0,0], [0,0,0]],
             [[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3)
        # 3D Sobel核 - y方向
        sobel_y = torch.tensor([
            [[[1,2,1], [0,0,0], [-1,-2,-1]],
             [[2,4,2], [0,0,0], [-2,-4,-2]],
             [[1,2,1], [0,0,0], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3)
        # 3D Sobel核 - z方向 (层间梯度)
        sobel_z = torch.tensor([
            [[[1,0,-1], [2,0,-2], [1,0,-1]],
             [[2,0,-2], [4,0,-4], [2,0,-2]],
             [[1,0,-1], [2,0,-2], [1,0,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3)
        self.register_buffer('kernel_x', sobel_x)
        self.register_buffer('kernel_y', sobel_y)
        self.register_buffer('kernel_z', sobel_z)

    def forward(self, x):
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
        # Reflect padding to reduce boundary artifacts on shallow z-patches
        x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        # 3D卷积 (no additional padding since we pre-padded)
        grad_x = F.conv3d(x_padded, kernel_x, padding=0, groups=C)
        grad_y = F.conv3d(x_padded, kernel_y, padding=0, groups=C)
        grad_z = F.conv3d(x_padded, kernel_z, padding=0, groups=C)
        # 梯度幅值 (L2 norm)
        return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)


class EAGLELoss3D(nn.Module):
    """
    3D版本EAGLE Loss (Edge-Aware Gradient Local Enhancement)
    用于3D CT医学影像超分辨率
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.sobel3d = Sobel3D()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        pred_grad = self.sobel3d(pred)
        target_grad = self.sobel3d(target)
        edge = F.l1_loss(pred_grad, target_grad)
        return l1 + self.alpha * edge


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Pseudo-Huber)
    论文: "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"
    比L1/L2更稳定，对异常值更鲁棒
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class SSIM3D(nn.Module):
    """
    3D SSIM (Structural Similarity Index Measure)
    用于3D体积数据的结构相似性计算
    """
    def __init__(self, window_size=7, size_average=True, channel=1):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size, channel):
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

    def forward(self, img1, img2):
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


class L1SSIMLoss3D(nn.Module):
    """
    L1 + SSIM 组合 Loss (3D版本)
    loss = L1 + alpha * (1 - SSIM)
    参考: ROADMAP Phase 1 Step 1 L4
    """
    def __init__(self, alpha=0.1, ssim_window_size=7):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIM3D(window_size=ssim_window_size, channel=1)

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        ssim_value = self.ssim(pred, target)
        ssim_loss = 1 - ssim_value  # SSIM损失: 1 - SSIM (值越小越好)
        return l1_loss + self.alpha * ssim_loss


class MultiScaleL1Loss(nn.Module):
    """
    Multi-scale L1 Loss
    在不同尺度下采样后计算L1 loss并加权
    参考: ROADMAP Phase 1 Step 1 L3
    
    默认使用3个尺度: 1x, 0.5x, 0.25x
    权重: [1.0, 0.5, 0.25]
    """
    def __init__(self, scales=[1, 0.5, 0.25], weights=[1.0, 0.5, 0.25], mode='trilinear'):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.mode = mode
        assert len(scales) == len(weights), "scales和weights长度必须相同"

    def forward(self, pred, target):
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                # 原始尺度
                pred_scaled = pred
                target_scaled = target
            else:
                # 下采样
                B, C, D, H, W = pred.shape
                new_D = max(int(D * scale), 1)
                new_H = max(int(H * scale), 1)
                new_W = max(int(W * scale), 1)
                
                pred_scaled = F.interpolate(
                    pred, size=(new_D, new_H, new_W), 
                    mode=self.mode, align_corners=False
                )
                target_scaled = F.interpolate(
                    target, size=(new_D, new_H, new_W), 
                    mode=self.mode, align_corners=False
                )
            
            l1_loss = F.l1_loss(pred_scaled, target_scaled)
            total_loss += weight * l1_loss
        
        return total_loss


__all__ = ['Sobel3D', 'EAGLELoss3D', 'CharbonnierLoss', 'SSIM3D', 'L1SSIMLoss3D', 'MultiScaleL1Loss']
