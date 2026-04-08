# RPLHR-CT 模块化系统设计规范

> **版本**: v1.0  
> **日期**: 2026-04-03  
> **状态**: 设计中

---

## 1. 设计原则

### 1.1 核心原则

```
┌─────────────────────────────────────────────────────────────────┐
│                    模块化设计三大原则                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. 配置驱动  - 所有策略通过配置选择，不硬编码                     │
│ 2. 向后兼容  - 保留所有历史实现，不删除或覆盖                      │
│ 3. 即插即用  - 各模块独立开发，通过统一接口集成                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 模块划分

```
RPLHR-CT/
└── code/
    ├── losses/           # Loss 函数模块
    ├── augmentation/     # 数据增强模块
    ├── training/         # 训练策略模块
    ├── models/           # 模型架构模块 (预留)
    └── inference/        # 推理策略模块 (预留)
```

---

## 2. Loss 函数模块 (losses/)

### 2.1 设计目标
支持通过配置灵活切换不同的 Loss 函数，支持组合 Loss。

### 2.2 模块结构

```
code/losses/
├── __init__.py
├── base.py                 # 基础 Loss 抽象类
├── registry.py             # Loss 注册表
├── l1_loss.py
├── eagle3d_loss.py
├── charbonnier_loss.py
├── ssim_loss.py
└── combined_loss.py        # 组合 Loss (L1+SSIM等)
```

### 2.3 配置接口

```ini
# config/modules/loss/eagle3d.txt
loss_type = 'eagle3d'
loss_alpha = 0.1
loss_beta = 0.01
```

```ini
# config/modules/loss/l1_ssim.txt
loss_type = 'combined'
loss_components = ['l1', 'ssim']
loss_weights = [1.0, 0.1]
```

### 2.4 使用方式

```python
from losses import build_loss

# 通过配置创建 Loss
config = load_config('config/modules/loss/eagle3d.txt')
criterion = build_loss(config)

# 使用
loss = criterion(pred, target)
```

---

## 3. 数据增强模块 (augmentation/)

### 3.1 设计目标
支持通过配置选择多种数据增强策略，支持组合和概率控制。

### 3.2 模块结构

```
code/augmentation/
├── __init__.py
├── base.py                 # 基础增强抽象类
├── registry.py             # 增强器注册表
├── flip.py                 # 翻转增强
├── noise.py                # 噪声增强
├── elastic.py              # 弹性形变
├── intensity.py            # 强度变换 (可选)
└── pipeline.py             # 增强流水线
```

### 3.3 配置接口

```ini
# config/modules/augment/flip_noise.txt
use_augmentation = True
augment_types = ['flip', 'noise']
augment_probability = 0.5

# Flip 参数
flip_axis = ['horizontal', 'vertical']  # 可选 'depth'

# Noise 参数
noise_type = 'gaussian'      # 'gaussian', 'poisson', 'both'
noise_sigma = 0.01
noise_poisson_scale = 1.0
```

### 3.4 使用方式

```python
from augmentation import build_augmentation

# 创建增强器
config = load_config('config/modules/augment/flip_noise.txt')
augmenter = build_augmentation(config)

# 在 Dataset 中使用
class TrainDataset(Dataset):
    def __getitem__(self, idx):
        data = load_data(idx)
        if self.training:
            data = augmenter(data)
        return data
```

---

## 4. 训练策略模块 (training/)

### 4.1 设计目标
支持通过配置选择优化器、学习率调度器、EMA、梯度裁剪等策略。

### 4.2 模块结构

```
code/training/
├── __init__.py
├── optimizer.py            # 优化器工厂
├── scheduler.py            # 学习率调度器工厂
├── ema.py                  # EMA 实现
├── grad_clip.py            # 梯度裁剪
└── trainer.py              # 训练器基类
```

### 4.3 配置接口

```ini
# config/modules/training/advanced.txt

# 优化器
optim_type = 'AdamW'
lr = 0.0002
weight_decay = 0.0001

# 学习率调度
scheduler_type = 'cosine'
Tmax = 100
warmup_epochs = 5

# EMA
use_ema = True
ema_decay = 0.999

# 梯度裁剪
use_grad_clip = True
grad_clip_norm = 1.0

# 其他
use_amp = False             # 混合精度 (预留)
gradient_accumulation = 1   # 梯度累积 (预留)
```

### 4.4 使用方式

```python
from training import build_optimizer, build_scheduler, EMA, GradClip

# 构建训练组件
optimizer = build_optimizer(model_params, config)
scheduler = build_scheduler(optimizer, config)
ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
grad_clip = GradClip(max_norm=config.grad_clip_norm) if config.use_grad_clip else None

# 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        loss = criterion(model(batch), target)
        loss.backward()
        
        if grad_clip:
            grad_clip(model.parameters())
        
        optimizer.step()
        
        if ema:
            ema.update(model)
```

---

## 5. 配置系统架构

### 5.1 配置层次

```
config/
├── modules/                    # 模块配置模板
│   ├── loss/
│   │   ├── l1.txt
│   │   ├── eagle3d.txt
│   │   ├── charbonnier.txt
│   │   └── l1_ssim.txt
│   ├── augment/
│   │   ├── none.txt
│   │   ├── flip.txt
│   │   ├── noise.txt
│   │   └── combined.txt
│   └── training/
│       ├── baseline.txt
│       ├── adamw_ema.txt
│       └── advanced.txt
└── experiments/                # 具体实验配置 (运行生成)
    ├── EXP_001_baseline/
    │   └── config.txt
    └── EXP_002_eagle3d/
        └── config.txt
```

### 5.2 配置继承与组合

```python
# 基础配置 + 模块配置组合
base_config = load_config('config/xuanwu_ratio4.txt')
loss_config = load_config('config/modules/loss/eagle3d.txt')
aug_config = load_config('config/modules/augment/flip.txt')
train_config = load_config('config/modules/training/advanced.txt')

# 合并配置
config = merge_configs(base_config, loss_config, aug_config, train_config)
```

### 5.3 配置验证

```python
from config_system import validate_config

# 验证配置完整性
errors = validate_config(config)
if errors:
    raise ConfigError(f"Invalid config: {errors}")
```

---

## 6. 实验管理规范

### 6.1 实验命名规则

```
EXP_[序号]_[模块]_[描述]

示例:
- EXP_001_baseline_l1              # 基线实验
- EXP_002_loss_eagle3d             # EAGLE3D Loss
- EXP_003_aug_flip_noise           # Flip+Noise 增强
- EXP_004_train_adamw_ema          # AdamW + EMA
- EXP_005_combined_best            # 最优组合
```

### 6.2 实验目录结构

```
experiments/
└── EXP_002_loss_eagle3d/
    ├── config.txt              # 完整配置
    ├── checkpoints/
    │   └── best_model.pth
    ├── logs/
    │   └── training.log
    ├── results/
    │   ├── validation_output/  # 验证集输出
    │   └── visualizations/     # 可视化结果
    └── report.md               # 实验报告
```

### 6.3 实验记录模板

```markdown
## EXP-[编号]: [实验描述]

### 配置
- **Loss**: [类型]
- **Augmentation**: [类型]
- **Training**: [策略]

### 结果
| Epoch | PSNR | SSIM | vs基线 |
|-------|------|------|--------|
| 10    |      |      |        |
| 20    |      |      |        |
| 50    |      |      |        |

### 结论
- [采纳/保留/放弃] + 原因

### 下一步
- [具体行动]
```

---

## 7. 开发规范

### 7.1 模块开发 checklist

- [ ] 实现基础接口 (BaseClass)
- [ ] 注册到注册表
- [ ] 创建配置模板
- [ ] 编写单元测试
- [ ] 更新文档

### 7.2 代码规范

```python
# 1. 所有模块必须继承基类
class MyLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, pred, target):
        # 实现
        pass

# 2. 使用装饰器注册
@LOSSES.register('my_loss')
class MyLoss(BaseLoss):
    ...

# 3. 配置参数必须有默认值
class MyLoss(BaseLoss):
    def __init__(self, config):
        self.alpha = config.get('alpha', 0.1)  # 有默认值
```

### 7.3 向后兼容保证

```
规则:
1. 现有文件 (train.py, loss_eagle3d.py等) 不修改
2. 新功能在新目录实现 (losses/, augmentation/等)
3. 通过配置切换，保持默认行为不变
```

---

## 8. 集成路线图

### Phase 1: 基础模块 (当前)
- [x] Loss 模块设计
- [x] 数据增强模块设计
- [x] 训练策略模块设计

### Phase 2: 配置系统 (下一步)
- [ ] 配置解析器实现
- [ ] 模块注册系统
- [ ] 配置验证器

### Phase 3: 工具集成 (下一步)
- [ ] 实验创建脚本更新
- [ ] 自动配置组合
- [ ] 实验对比工具

### Phase 4: 新模块扩展 (预留)
- [ ] 模型架构模块
- [ ] 推理策略模块
- [ ] 评估指标模块

---

## 9. 附录

### 9.1 配置参数速查表

| 模块 | 参数名 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| Loss | loss_type | str | 'l1' | Loss 类型 |
| Loss | loss_alpha | float | 0.1 | 组合权重 |
| Augment | use_augmentation | bool | False | 是否启用 |
| Augment | augment_types | list | [] | 增强类型列表 |
| Augment | augment_probability | float | 0.5 | 应用概率 |
| Training | optim_type | str | 'Adam' | 优化器类型 |
| Training | use_ema | bool | False | 是否使用 EMA |
| Training | ema_decay | float | 0.999 | EMA 衰减率 |
| Training | use_grad_clip | bool | False | 是否梯度裁剪 |
| Training | grad_clip_norm | float | 1.0 | 裁剪阈值 |

### 9.2 快速开始

```bash
# 1. 查看可用配置
./scripts/autodl_skill/list_configs.sh

# 2. 创建新实验
./scripts/autodl_skill/quick_experiment.sh \
    EXP_002_eagle3d \
    loss/eagle3d \
    augment/none \
    training/baseline

# 3. 运行实验
./scripts/autodl_skill/run_training.sh EXP_002_eagle3d

# 4. 对比实验
./scripts/autodl_skill/compare_experiments.sh EXP_001_baseline EXP_002_eagle3d
```

---

*文档版本: v1.0*  
*最后更新: 2026-04-03*
