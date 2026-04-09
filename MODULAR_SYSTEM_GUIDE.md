# RPLHR-CT 模块化配置系统使用指南

## 概述

模块化配置系统将训练配置分解为三个独立模块：
- **Loss 模块**: 损失函数配置
- **Augmentation 模块**: 数据增强配置
- **Training 模块**: 优化器、学习率调度等训练参数

这种设计允许灵活组合不同配置，快速进行实验对比。

## 目录结构

```
config/
├── modules/
│   ├── loss/
│   │   ├── l1.txt           # L1 Loss
│   │   ├── eagle3d.txt      # EAGLE3D Edge-Aware Loss
│   │   ├── charbonnier.txt  # Charbonnier Loss
│   │   ├── l1_ssim.txt      # L1 + SSIM Loss
│   │   └── multiscale.txt   # Multi-Scale L1 Loss
│   ├── augment/
│   │   ├── none.txt         # 无增强
│   │   ├── flip.txt         # 随机翻转
│   │   ├── noise.txt        # 随机噪声
│   │   ├── combined.txt     # 组合增强
│   │   └── light.txt        # 轻量级增强
│   └── training/
│       ├── baseline.txt     # 基线配置 (Adam)
│       ├── adamw_ema.txt    # AdamW + EMA
│       ├── advanced.txt     # 高级配置 (Warmup + Cosine)
│       ├── sgd_cosine.txt   # SGD + Cosine
│       └── fast.txt         # 快速训练
└── experiments/             # 实验配置保存目录
    └── *.json               # 实验配置文件
```

## 快速开始

### 1. 列出可用配置

```bash
cd scripts/autodl_skill

# 列出所有配置
./list_configs.sh

# 只列出 Loss 配置
./list_configs.sh --loss

# JSON 格式输出
./list_configs.sh --json
```

### 2. 快速启动实验

```bash
# 格式: ./quick_experiment.sh <exp_name> <loss_cfg> <aug_cfg> <train_cfg> [epochs]

# 示例1: 使用 L1 loss + 翻转增强 + AdamW/EMA, 训练 50 轮
./quick_experiment.sh exp01 l1 flip adamw_ema 50

# 示例2: 使用 EAGLE3D loss + 组合增强 + 高级配置, 训练 100 轮
./quick_experiment.sh exp02 eagle3d combined advanced 100

# 示例3: 使用 Charbonnier loss + 无增强 + 快速配置
./quick_experiment.sh exp03 charbonnier none fast 30
```

### 3. 使用模块化配置创建实验

```bash
# 创建带模块化配置的实验记录
./create_experiment.sh \
    --loss eagle3d \
    --augment flip \
    --training adamw_ema \
    --name "eagle_adamw_test"

# 创建并立即启动
./create_experiment.sh \
    --loss l1_ssim \
    --augment combined \
    --training advanced \
    --name "ssim_advanced" \
    --epochs 100
```

### 4. 使用配置文件运行训练

```bash
# 使用预定义的实验配置
./run_training.sh --config ../config/experiments/exp01.json 50

# 使用模块组合
./run_training.sh --loss eagle3d --augment flip --training adamw_ema 100

# 基础用法 (使用默认配置)
./run_training.sh 50
```

### 5. 对比实验配置

```bash
# 对比两个实验的模块配置差异
./compare_experiments.sh --compare-modules --exp1 exp01 --exp2 exp02

# 生成完整对比报告
./compare_experiments.sh --report
```

## 配置模块详解

### Loss 模块

| 配置 | 说明 | 适用场景 |
|------|------|----------|
| `l1` | 标准 L1 损失 | 基线实验 |
| `eagle3d` | 3D 边缘感知损失 | 需要保留边缘细节 |
| `charbonnier` | Charbonnier/Pseudo-Huber | 对异常值更鲁棒 |
| `l1_ssim` | L1 + SSIM 组合 | 兼顾像素级和结构级 |
| `multiscale` | 多尺度 L1 | 捕获不同尺度特征 |

**配置示例** (`eagle3d.txt`):
```python
type = 'EAGLE3D'
alpha = 0.1  # 边缘损失权重
```

### Augmentation 模块

| 配置 | 说明 | 增强策略 |
|------|------|----------|
| `none` | 无增强 | - |
| `flip` | 随机翻转 | 水平/垂直翻转 |
| `noise` | 随机噪声 | 高斯噪声 |
| `combined` | 组合增强 | 翻转+噪声+旋转 |
| `light` | 轻量级 | 低强度翻转+噪声 |

**配置示例** (`combined.txt`):
```python
enabled = True
flip_prob = 0.5
noise_prob = 0.2
noise_std = 0.01
rotation_prob = 0.1
max_rotation_angle = 15
```

### Training 模块

| 配置 | 优化器 | 调度器 | 特性 |
|------|--------|--------|------|
| `baseline` | Adam | Plateau | 基线配置 |
| `adamw_ema` | AdamW | Plateau | EMA 模型平均 |
| `advanced` | AdamW | Cosine | Warmup + 梯度裁剪 |
| `sgd_cosine` | SGD | Cosine | 大规模数据集 |
| `fast` | AdamW | StepLR | 快速验证 |

**配置示例** (`advanced.txt`):
```python
optimizer = 'AdamW'
lr = 0.0003
wd = 0.0001
scheduler = 'CosineAnnealingLR'
Tmax = 200
use_warmup = True
warmup_epochs = 10
use_grad_clip = True
grad_clip_norm = 1.0
use_ema = True
ema_decay = 0.999
```

## Python API 使用

### 基础用法

```python
from code.config_system import ModularConfig, quick_build_config

# 方式1: 从模块配置文件加载
config = ModularConfig.from_module_configs(
    loss_module='eagle3d',
    augment_module='flip', 
    training_module='adamw_ema'
)

# 方式2: 快速构建
config = quick_build_config(
    loss_type='EAGLE3D',
    augment_type='flip',
    training_type='adamw_ema'
)

# 查看配置
config.print_config()
```

### 在训练脚本中使用

```python
from code.config_system import ModularConfig
from code.augmentation import Augmentor

# 加载配置
config = ModularConfig.from_module_configs('eagle3d', 'flip', 'adamw_ema')

# 构建损失函数
criterion = config.build_loss(device=device)

# 构建优化器
optimizer = config.build_optimizer(model.parameters())

# 构建学习率调度器
scheduler = config.build_scheduler(optimizer)

# 构建数据增强器
augmentor = Augmentor.from_config(config.get_augment_config())

# 在训练循环中使用
for epoch in range(epochs):
    for x, y in dataloader:
        # 应用数据增强
        x, y = augmentor(x, y)
        
        # 前向传播
        output = model(x)
        loss = criterion(output, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 保存和加载实验配置

```python
# 保存实验配置
config.save_experiment_config('my_experiment', output_dir='../config/experiments')

# 加载实验配置
config = ModularConfig.load_experiment_config('../config/experiments/my_experiment.json')
```

### 配置管理器

```python
from code.config_system import ConfigManager

# 创建管理器
manager = ConfigManager()

# 列出可用模块
loss_modules = manager.list_available_modules('loss')
augment_modules = manager.list_available_modules('augment')
training_modules = manager.list_available_modules('training')

# 列出实验
experiments = manager.list_experiments()

# 对比配置差异
diff = manager.compare_configs('exp01', 'exp02')
print(diff)
```

## 创建自定义配置模块

### 1. 创建 Loss 配置

创建文件 `config/modules/loss/my_loss.txt`:

```python
# 我的自定义 Loss
# 说明: 这里是配置说明

type = 'EAGLE3D'  # 必须是已注册的 loss 类型
alpha = 0.2       # 自定义参数
```

### 2. 创建 Augmentation 配置

创建文件 `config/modules/augment/my_augment.txt`:

```python
# 我的自定义增强配置

enabled = True
flip_prob = 0.3
noise_prob = 0.1
noise_std = 0.005
```

### 3. 创建 Training 配置

创建文件 `config/modules/training/my_training.txt`:

```python
# 我的自定义训练配置

optimizer = 'AdamW'
lr = 0.0005
wd = 0.0001
use_ema = True
ema_decay = 0.9999
```

## 实验工作流

### 标准实验流程

```bash
# 1. 查看可用配置
./list_configs.sh

# 2. 创建实验
./create_experiment.sh \
    --loss eagle3d \
    --augment flip \
    --training adamw_ema \
    --name "baseline_eagle"

# 3. 启动训练
./quick_experiment.sh baseline_eagle eagle3d flip adamw_ema 100

# 4. 监控训练
./check_training.sh --watch

# 5. 收集结果
./collect_results.sh

# 6. 对比实验
./compare_experiments.sh --report
```

### A/B 测试流程

```bash
# 实验 A: L1 Loss
./quick_experiment.sh exp_l1 l1 flip adamw_ema 50

# 实验 B: EAGLE3D Loss (其他配置相同)
./quick_experiment.sh exp_eagle eagle3d flip adamw_ema 50

# 对比结果
./compare_experiments.sh --compare-modules --exp1 exp_l1 --exp2 exp_eagle
```

## 故障排除

### 配置文件未找到

```bash
# 检查模块目录
ls -la ../../config/modules/loss/

# 确保文件名正确 (区分大小写)
./list_configs.sh --loss
```

### 实验配置加载失败

```bash
# 验证 JSON 格式
python3 -m json.tool ../../config/experiments/my_exp.json

# 检查必要字段
python3 -c "import json; d=json.load(open('../../config/experiments/my_exp.json')); print(d.keys())"
```

### 模块配置不生效

确保在 `run_training.sh` 或 `quick_experiment.sh` 中正确指定了模块:

```bash
# 正确用法
./quick_experiment.sh exp01 l1 flip adamw_ema

# 错误用法 (顺序错误)
./quick_experiment.sh exp01 flip l1 adamw_ema
```

## 最佳实践

1. **命名规范**: 实验名称使用小写字母和下划线,如 `eagle3d_flip_adamw`
2. **版本控制**: 自定义模块配置也应提交到 Git
3. **文档记录**: 在 `EXPERIMENTS.md` 中记录每个实验的目的和结果
4. **基线对比**: 每次修改一个模块,保持其他模块不变,便于归因
5. **配置备份**: 重要实验的配置文件应单独备份

## 参考

- [EXPERIMENTS.md](./scripts/autodl_skill/EXPERIMENTS.md) - 实验记录
- [ROADMAP.md](./scripts/autodl_skill/ROADMAP.md) - 开发路线图
- [AUTODL_WORKFLOW.md](./AUTODL_WORKFLOW.md) - 完整工作流程
