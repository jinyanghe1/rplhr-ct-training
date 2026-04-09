# 模块化 Loss 函数系统

本目录包含了一个模块化的 Loss 函数系统，支持通过配置选择不同的 Loss 函数。

## 目录结构

```
losses/
├── __init__.py          # 统一接口，导出所有 Loss 类
├── base_loss.py         # 基础 Loss 抽象类
├── l1_loss.py           # L1 Loss
├── eagle3d_loss.py      # EAGLE3D Loss
├── charbonnier_loss.py  # Charbonnier Loss
├── ssim_loss.py         # SSIM Loss
├── combined_loss.py     # 组合 Loss (L1+SSIM)
└── loss_factory.py      # Loss 工厂类
```

## 支持的 Loss 类型

| Loss 类型 | 类名 | 说明 |
|-----------|------|------|
| `l1` | L1Loss | 标准 L1 损失 |
| `eagle3d` | EAGLELoss3D | 边缘感知梯度增强损失 (3D) |
| `charbonnier` | CharbonnierLoss | Pseudo-Huber 损失，更稳定 |
| `ssim` | SSIMLoss | 结构相似性损失 |
| `l1_ssim` | CombinedLoss | L1 + SSIM 组合损失 |
| `smooth_l1` | SmoothL1Loss | Smooth L1 (Huber) 损失 |

## 使用示例

### 1. 直接导入使用

```python
from losses import L1Loss, EAGLELoss3D, CharbonnierLoss
import torch

# 创建 Loss 函数
loss_fn = EAGLELoss3D(alpha=0.1)

# 使用
pred = torch.randn(2, 1, 32, 64, 64)
target = torch.randn(2, 1, 32, 64, 64)
loss = loss_fn(pred, target)
```

### 2. 使用 LossFactory

```python
from losses import LossFactory

# 通过类型创建
loss_fn = LossFactory.create_loss('l1')
loss_fn = LossFactory.create_loss('eagle3d', alpha=0.2)
loss_fn = LossFactory.create_loss('charbonnier', eps=1e-6)

# 通过配置字典创建
config = {'loss_type': 'l1_ssim', 'alpha': 0.1}
loss_fn = LossFactory.create_loss_from_config(config)

# 通过配置文件创建
loss_fn = LossFactory.create_loss_from_file('config/loss_configs/loss_l1.txt')
```

### 3. 使用便捷函数 get_loss

```python
from losses import get_loss

# 支持多种配置方式
loss_fn = get_loss('l1')  # 类型名称
loss_fn = get_loss('config/loss_configs/loss_eagle3d.txt')  # 配置文件
loss_fn = get_loss({'loss_type': 'ssim', 'window_size': 7})  # 配置字典
```

### 4. 在训练脚本中使用

```python
from losses import get_loss

# 从配置文件中读取 loss 类型
def get_loss_from_args(args):
    if hasattr(args, 'loss_config') and args.loss_config:
        return get_loss(args.loss_config)
    elif hasattr(args, 'loss_type'):
        return get_loss(args.loss_type)
    else:
        return get_loss('l1')  # 默认使用 L1

# 在训练循环中
loss_fn = get_loss_from_args(args)
for batch in dataloader:
    pred = model(batch['input'])
    loss = loss_fn(pred, batch['target'])
    loss.backward()
```

## 配置文件格式

配置文件使用简单的键值对格式：

```
# 注释以 # 或 // 开头
loss_type=eagle3d
alpha=0.1
```

示例配置文件见 `config/loss_configs/` 目录。

## 扩展新的 Loss

1. 在 `base_loss.py` 中继承 `BaseLoss`
2. 在 `loss_factory.py` 中注册新的 Loss 类型

```python
# my_loss.py
from losses.base_loss import BaseLoss

class MyLoss(BaseLoss):
    def __init__(self, param=1.0):
        super().__init__()
        self.param = param
    
    def forward(self, pred, target):
        return self.param * torch.mean((pred - target) ** 2)
    
    def get_config(self):
        return {'loss_type': 'my_loss', 'param': self.param}

# 注册
from losses.loss_factory import LossFactory
LossFactory.register_loss('my_loss', MyLoss)
```

## 与现有代码兼容

本模块完全兼容现有的 `loss_eagle3d.py`，可以作为其替代品使用，也可以同时使用。

```python
# 新旧代码兼容
from loss_eagle3d import EAGLELoss3D as OldEAGLELoss  # 原有导入方式
from losses import EAGLELoss3D as NewEAGLELoss         # 新的导入方式

# 两者功能完全相同
```
