# 模块化训练策略系统

本目录包含一个灵活、可配置的模块化训练策略系统，支持优化器、学习率调度器、EMA、梯度裁剪等组件的自由组合。

## 目录结构

```
training/
├── __init__.py              # 统一接口，导出所有组件
├── optimizer_factory.py     # 优化器工厂 (Adam, AdamW, SGD)
├── scheduler_factory.py     # 学习率调度器工厂 (Cosine, Plateau, Step, Warmup)
├── ema.py                   # EMA (指数移动平均) 实现
├── grad_clip.py             # 梯度裁剪工具
├── trainer_base.py          # 基础训练器类
├── trainer_advanced.py      # 高级训练器 (支持 EMA 和 GradClip)
└── README.md                # 本文档
```

## 快速开始

### 方式一：使用高级训练器（推荐）

```python
from training import TrainerAdvanced

# 创建训练器
trainer = TrainerAdvanced(
    model=model,
    config=config,  # 配置对象
    device=device,
    checkpoint_root='../checkpoints'
)

# 开始训练
trainer.train(train_loader, val_loader, num_epochs=200)
```

### 方式二：手动组合组件

```python
from training import (
    build_optimizer,
    build_scheduler,
    create_ema,
    create_grad_clipper
)

# 创建优化器
optimizer = build_optimizer(model, config)

# 创建学习率调度器
scheduler = build_scheduler(optimizer, config)

# 创建 EMA
ema = create_ema(model, config, device=device)

# 创建梯度裁剪器
grad_clipper = create_grad_clipper(config)

# 在训练循环中使用
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向/反向传播
        loss.backward()
        
        # 梯度裁剪
        if grad_clipper:
            grad_clipper(model.parameters())
        
        optimizer.step()
        
        # 更新 EMA
        if ema:
            ema.update(model)
```

## 配置参数

### 优化器配置

```python
optim_type = 'AdamW'      # 优化器类型: 'Adam', 'AdamW', 'SGD'
lr = 0.0002               # 学习率
weight_decay = 0.0001     # 权重衰减

# Adam/AdamW 特有参数
betas = (0.9, 0.999)      # Adam betas

# SGD 特有参数
momentum = 0.9            # SGD 动量
nesterov = False          # 是否使用 Nesterov 动量
```

### 学习率调度器配置

```python
scheduler_type = 'cosine'     # 类型: 'cosine', 'plateau', 'step', 'multistep', 'exponential', 'none'
warmup_epochs = 5             # Warmup 轮数 (0 表示不使用)

# Cosine 特有参数
Tmax = 100                    # 周期长度
lr_gap = 1000                 # 最小学习率 = lr / lr_gap

# Plateau 特有参数
patience = 15                 # 耐心值
mode = 'max'                  # 'max' (最大化指标) 或 'min' (最小化损失)

# Step 特有参数
step_size = 30                # 步长
gamma = 0.1                   # 衰减系数

# MultiStep 特有参数
milestones = [30, 60, 90]     # 里程碑 epoch
```

### EMA 配置

```python
use_ema = True            # 是否启用 EMA
ema_decay = 0.999         # EMA 衰减系数 (越接近 1，历史权重越大)
use_ema_for_val = True    # 是否使用 EMA 模型进行验证
```

### 梯度裁剪配置

```python
use_grad_clip = True          # 是否启用梯度裁剪
grad_clip_norm = 1.0          # 梯度 L2 范数上限
grad_clip_type = 'norm'       # 裁剪类型: 'norm', 'value', 'adaptive'

# Value 裁剪特有参数
grad_clip_value = 1.0         # 梯度值上限

# Norm 裁剪特有参数
grad_clip_norm_type = 2.0     # 范数类型 (2.0 表示 L2)
```

## 配置文件示例

配置文件位于 `code/config/training_configs/` 目录：

### 1. 基线配置 (Adam + Cosine)

```python
# train_adam_cosine.txt
optim_type = 'Adam'
lr = 0.0002
weight_decay = 0.0
scheduler_type = 'cosine'
Tmax = 100
use_ema = False
use_grad_clip = False
```

### 2. AdamW + EMA（推荐）

```python
# train_adamw_ema.txt
optim_type = 'AdamW'
lr = 0.0002
weight_decay = 0.0001
scheduler_type = 'cosine'
Tmax = 100
use_ema = True
ema_decay = 0.999
use_grad_clip = False
```

### 3. 高级配置（完整功能）

```python
# train_advanced.txt
optim_type = 'AdamW'
lr = 0.0002
weight_decay = 0.0001
scheduler_type = 'cosine'
Tmax = 100
warmup_epochs = 5
use_ema = True
ema_decay = 0.999
use_grad_clip = True
grad_clip_norm = 1.0
```

## 使用示例

### 示例 1：加载配置文件并训练

```python
import sys
sys.path.insert(0, './code')

from config import opt
from net import model_TransSR
from training import TrainerAdvanced

# 加载默认配置
opt.load_config('../config/default.txt')

# 加载训练策略配置（覆盖相关参数）
training_config = './code/config/training_configs/train_advanced.txt'
with open(training_config, 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.split('=', 1)
            key = key.strip()
            value = eval(value.split('#')[0].strip())
            setattr(opt, key, value)

# 创建模型和训练器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_TransSR.TVSRN().to(device)

trainer = TrainerAdvanced(model, opt, device)

# 开始训练（需要准备 train_loader 和 val_loader）
# trainer.train(train_loader, val_loader, opt.epoch)
```

### 示例 2：对比不同训练策略

```python
from training import create_trainer

strategies = [
    'train_adam_cosine.txt',
    'train_adamw_ema.txt', 
    'train_advanced.txt'
]

for strategy_file in strategies:
    # 加载配置
    config = load_config(strategy_file)
    
    # 创建训练器
    trainer = create_trainer(model, config, trainer_type='advanced')
    
    # 训练
    trainer.train(train_loader, val_loader, num_epochs)
```

## 组件详解

### OptimizerFactory

```python
from training import OptimizerFactory

optimizer = OptimizerFactory.create_optimizer(
    parameters=model.parameters(),
    optim_type='AdamW',
    lr=0.0002,
    weight_decay=0.0001
)
```

### SchedulerFactory

```python
from training import SchedulerFactory, WarmupScheduler

# 基础调度器
scheduler = SchedulerFactory.create_scheduler(
    optimizer,
    scheduler_type='cosine',
    T_max=100
)

# 带 Warmup 的调度器
warmup_scheduler = WarmupScheduler(
    optimizer,
    warmup_epochs=5,
    base_scheduler=scheduler
)
```

### EMA

```python
from training import EMA

# 创建 EMA
ema = EMA(model, decay=0.999, device=device)

# 训练时更新
ema.update(model)

# 验证时使用 EMA 参数
ema.apply_shadow(model)
validate(model)
ema.restore(model)
```

### GradClipper

```python
from training import GradClipper, AdaptiveGradClipper

# 固定阈值裁剪
clipper = GradClipper(max_norm=1.0)

# 自适应裁剪
clipper = AdaptiveGradClipper(initial_max_norm=1.0)

# 使用
loss.backward()
grad_norm = clipper(model.parameters())  # 返回裁剪前的梯度范数
optimizer.step()
```

## 向后兼容性

本模块设计与现有代码兼容，支持从旧的配置参数自动推断：

```python
# 旧配置格式（仍然支持）
opt.optim = 'AdamW'       # 映射到 optim_type
opt.wd = 0.0001           # 映射到 weight_decay
opt.cos_lr = True         # 映射到 scheduler_type='cosine'
opt.Tmin = True           # 映射到 scheduler_type='plateau', mode='min'
```

## 注意事项

1. **不要修改现有的 `train.py` 或 `trainxuanwu.py`**
2. 新代码完全放在 `training/` 目录下
3. 通过配置文件选择训练策略，无需修改代码
4. EMA 模型在验证时通常效果更好，建议启用 `use_ema_for_val=True`
5. 梯度裁剪可以防止梯度爆炸，但过大的裁剪阈值可能效果不明显

## 扩展开发

如需添加新的优化器或调度器：

1. 在 `optimizer_factory.py` 的 `OptimizerFactory.create_optimizer()` 中添加新类型
2. 在 `scheduler_factory.py` 的 `SchedulerFactory.create_scheduler()` 中添加新类型
3. 更新配置文件示例

## 参考

- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
- [Exponential Moving Average](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
