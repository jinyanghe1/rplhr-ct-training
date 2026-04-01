# 3D Edge-Aware Loss 调研报告

> **调研时间**: 2026-04-01  
> **目的**: 为3D CT医学影像超分寻找合适的edge-aware损失函数

---

## 📋 背景

EXP_001尝试使用EAGLE Loss（2D实现），但CT数据是3D体积数据(B,C,D,H,W)，直接使用`F.conv2d`会导致维度不匹配错误。

**问题**: 如何将2D edge-aware loss适配到3D医学影像数据？

---

## 🧪 方案对比

### 方案1: Reshape 3D→2D (快速适配)

```python
def eagle_loss_3d_reshape(pred, target, alpha=0.1):
    """将3D数据reshape为4D处理"""
    B, C, D, H, W = pred.shape
    
    # (B,C,D,H,W) -> (B*D,C,H,W)
    pred_4d = pred.view(B*D, C, H, W)
    target_4d = target.view(B*D, C, H, W)
    
    # 2D EAGLE Loss计算
    l1 = F.l1_loss(pred_4d, target_4d)
    
    # 2D Sobel
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], 
                           dtype=torch.float32).view(1,1,3,3).to(pred.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], 
                           dtype=torch.float32).view(1,1,3,3).to(pred.device)
    
    c = pred_4d.shape[1]
    pred_grad = torch.abs(F.conv2d(pred_4d, sobel_x.repeat(c,1,1,1), padding=1, groups=c)) + \
                torch.abs(F.conv2d(pred_4d, sobel_y.repeat(c,1,1,1), padding=1, groups=c))
    target_grad = torch.abs(F.conv2d(target_4d, sobel_x.repeat(c,1,1,1), padding=1, groups=c)) + \
                  torch.abs(F.conv2d(target_4d, sobel_y.repeat(c,1,1,1), padding=1, groups=c))
    
    edge = F.l1_loss(pred_grad, target_grad)
    return l1 + alpha * edge
```

**优点**:
- 快速实现，改动最小
- 保持EAGLE Loss原算法

**缺点**:
- 丢失层间(z轴)梯度信息
- 不是真正的3D梯度

---

### 方案2: 3D Sobel Operator (推荐)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobel3D(nn.Module):
    """3D Sobel算子 - 真正的3D梯度"""
    def __init__(self):
        super().__init__()
        # 3D Sobel核: x, y, z三个方向
        self.register_buffer('kernel_x', torch.tensor([
            [[[1,2,1], [2,4,2], [1,2,1]],
             [[0,0,0], [0,0,0], [0,0,0]],
             [[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3))
        
        self.register_buffer('kernel_y', torch.tensor([
            [[[1,2,1], [0,0,0], [-1,-2,-1]],
             [[2,4,2], [0,0,0], [-2,-4,-2]],
             [[1,2,1], [0,0,0], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3))
        
        self.register_buffer('kernel_z', torch.tensor([
            [[[1,0,-1], [2,0,-2], [1,0,-1]],
             [[2,0,-2], [4,0,-4], [2,0,-2]],
             [[1,0,-1], [2,0,-2], [1,0,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3))
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            gradient_magnitude: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 扩展到对应channel数
        kernel_x = self.kernel_x.repeat(C, 1, 1, 1, 1)
        kernel_y = self.kernel_y.repeat(C, 1, 1, 1, 1)
        kernel_z = self.kernel_z.repeat(C, 1, 1, 1, 1)
        
        # 3D卷积
        grad_x = F.conv3d(x, kernel_x, padding=1, groups=C)
        grad_y = F.conv3d(x, kernel_y, padding=1, groups=C)
        grad_z = F.conv3d(x, kernel_z, padding=1, groups=C)
        
        # 梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        
        return gradient_magnitude


class EAGLELoss3D(nn.Module):
    """3D版本EAGLE Loss"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.sobel3d = Sobel3D()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W)
            target: (B, C, D, H, W)
        """
        # L1基础损失
        l1 = F.l1_loss(pred, target)
        
        # 3D梯度损失
        pred_grad = self.sobel3d(pred)
        target_grad = self.sobel3d(target)
        edge = F.l1_loss(pred_grad, target_grad)
        
        return l1 + self.alpha * edge
```

**优点**:
- 真正的3D梯度计算
- 包含层间(z轴)信息
- 更适合3D医学影像

**缺点**:
- 计算量比2D版本大
- 需要更多显存

---

### 方案3: 使用Kornia库 (最简单)

```python
# 需要: pip install kornia

import kornia
from kornia.filters import spatial_gradient3d

def gradient_loss_kornia(pred, target):
    """使用Kornia的3D空间梯度"""
    # 计算3D梯度
    pred_grad = spatial_gradient3d(pred, mode='sobel', order=1)
    target_grad = spatial_gradient3d(target, mode='sobel', order=1)
    
    # pred_grad: (B, C, 3, D, H, W) - 3个方向的梯度
    # 合并梯度幅值
    pred_grad_mag = torch.sqrt(
        pred_grad[:,:,0]**2 + pred_grad[:,:,1]**2 + pred_grad[:,:,2]**2 + 1e-8
    )
    target_grad_mag = torch.sqrt(
        target_grad[:,:,0]**2 + target_grad[:,:,1]**2 + target_grad[:,:,2]**2 + 1e-8
    )
    
    return F.l1_loss(pred_grad_mag, target_grad_mag)
```

**优点**:
- 现成实现，无需手动编写核
- 支持sobel/diff两种模式
- 支持1阶/2阶导数

**缺点**:
- 需要额外依赖kornia
- 可控性不如手动实现

---

### 方案4: Charbonnier Loss (通用替代)

```python
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
```

**优点**:
- 简单有效
- 无需适配维度
- 超分任务常用

---

## 📊 方案对比总结

| 方案 | 实现难度 | 3D感知 | 计算量 | 推荐度 |
|------|----------|--------|--------|--------|
| Reshape 3D→2D | ⭐ 低 | ❌ 否 | 低 | ⭐⭐ |
| **3D Sobel** | ⭐⭐ 中 | ✅ 是 | 中 | ⭐⭐⭐⭐⭐ |
| Kornia | ⭐ 低 | ✅ 是 | 中 | ⭐⭐⭐⭐ |
| Charbonnier | ⭐ 低 | ❌ 否 | 低 | ⭐⭐⭐ |

---

## 🎯 推荐实施方案

### EXP_001 修复建议

```python
# code/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel3D(nn.Module):
    """3D Sobel梯度算子"""
    def __init__(self):
        super().__init__()
        self.register_buffer('kernel_x', torch.tensor([[
            [[1,2,1], [2,4,2], [1,2,1]],
            [[0,0,0], [0,0,0], [0,0,0]],
            [[-1,-2,-1], [-2,-4,-2], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3))
        
        self.register_buffer('kernel_y', torch.tensor([[
            [[1,2,1], [0,0,0], [-1,-2,-1]],
            [[2,4,2], [0,0,0], [-2,-4,-2]],
            [[1,2,1], [0,0,0], [-1,-2,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3))
        
        self.register_buffer('kernel_z', torch.tensor([[
            [[1,0,-1], [2,0,-2], [1,0,-1]],
            [[2,0,-2], [4,0,-4], [2,0,-2]],
            [[1,0,-1], [2,0,-2], [1,0,-1]]]
        ], dtype=torch.float32).view(1,1,3,3,3))
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        grad_x = F.conv3d(x, self.kernel_x.repeat(C,1,1,1,1), padding=1, groups=C)
        grad_y = F.conv3d(x, self.kernel_y.repeat(C,1,1,1,1), padding=1, groups=C)
        grad_z = F.conv3d(x, self.kernel_z.repeat(C,1,1,1,1), padding=1, groups=C)
        return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)


class EAGLELoss3D(nn.Module):
    """
    EAGLE Loss 3D版本
    Edge-Aware Gradient Local Enhancement for 3D CT
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.sobel3d = Sobel3D()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) - 预测3D体积
            target: (B, C, D, H, W) - 目标3D体积
        """
        l1 = F.l1_loss(pred, target)
        pred_grad = self.sobel3d(pred)
        target_grad = self.sobel3d(target)
        edge = F.l1_loss(pred_grad, target_grad)
        return l1 + self.alpha * edge
```

---

## 📚 参考文献

1. **EAGLE Loss原始论文**: 
   - "EAGLE: an edge-aware gradient localization enhanced loss for CT image reconstruction" - SPIE JMI 2024

2. **3D Sobel Operator**:
   - Kornia Documentation: `kornia.filters.spatial_gradient3d`

3. **Charbonnier Loss**:
   - "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution" - CVPR 2017

4. **3D梯度在医学影像中的应用**:
   - "A 3D CNN with Gradient Guidance for Super-Resolution of LGE Cardiac MRI" - PMC 2023

---

## ✅ 后续行动

1. [ ] 更新 `code/loss.py` 添加 `EAGLELoss3D`
2. [ ] 更新 `code/trainxuanwu.py` 使用新的3D版本
3. [ ] 重新开始 EXP_001 训练
4. [ ] 对比 3D vs 2D reshape 的效果

---

*报告生成: 2026-04-01*  
*调研来源: WebSearch + Code Analysis*
