# 模块化数据增强系统 (Modular Data Augmentation System)

针对3D CT影像超分辨率任务的模块化数据增强系统，支持通过配置灵活选择不同的增强策略。

## 目录结构

```
augmentation/
├── __init__.py           # 模块入口，统一接口
├── base_augment.py       # 基础增强抽象类
├── flip_augment.py       # 翻转增强（H/V/D 三轴）
├── noise_augment.py      # 随机噪声（泊松+高斯）
├── elastic_augment.py    # 3D弹性形变
├── intensity_augment.py  # 强度变换
├── augment_factory.py    # 增强器工厂类
├── augment_pipeline.py   # 多增强组合管道
└── README.md             # 本文件
```

## 快速开始

### 1. 基本用法

```python
from augmentation import AugmentFactory

# 定义配置
config = {
    'use_augmentation': True,
    'augment_types': ['flip', 'noise'],
    'augment_probability': 0.5,
    'flip_axis': ['horizontal', 'vertical'],
    'noise_type': 'gaussian',
    'noise_sigma': 0.01,
}

# 创建增强器
augmenter = AugmentFactory.create(config)

# 应用增强（训练时）
lr_aug, hr_aug = augmenter(lr_img, hr_img, is_training=True)

# 验证时不应用增强
lr_val, hr_val = augmenter(lr_img, hr_img, is_training=False)
```

### 2. 从配置文件加载

```python
from augmentation import AugmentFactory

# 从配置文件创建增强器
augmenter = AugmentFactory.from_config_file('config/augment_configs/aug_combined.txt')

# 应用增强
lr_aug, hr_aug = augmenter(lr_img, hr_img, is_training=True)
```

### 3. 创建单个增强器

```python
from augmentation import FlipAugment, NoiseAugment, ElasticAugment

# 翻转增强器
flip_aug = FlipAugment(prob=0.5, axes=['horizontal', 'vertical'])
lr, hr = flip_aug(lr_img, hr_img, is_training=True)

# 噪声增强器
noise_aug = NoiseAugment(prob=0.5, noise_type='gaussian', sigma=0.01)
lr, hr = noise_aug(lr_img, hr_img, is_training=True)
```

## 配置参数说明

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_augmentation` | bool | False | 是否启用数据增强 |
| `augment_types` | list | [] | 增强类型列表，如['flip', 'noise'] |
| `augment_probability` | float | 0.5 | 基础应用概率 |

### 翻转增强参数 (flip)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `flip_axis` | list | ['horizontal', 'vertical'] | 翻转轴，可选：'horizontal'/'h', 'vertical'/'v', 'depth'/'d' |

### 噪声增强参数 (noise)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `noise_type` | str | 'gaussian' | 噪声类型：'gaussian', 'poisson', 'both' |
| `noise_sigma` | float | 0.01 | 高斯噪声标准差（相对图像最大值的比例） |
| `noise_scale` | float | 1.0 | 泊松噪声缩放因子 |

### 弹性形变参数 (elastic)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `elastic_alpha` | float | 10.0 | 变形幅度 |
| `elastic_sigma` | float | 3.0 | 平滑程度 |

> 注意：弹性形变计算量大，建议`prob`设置在0.1-0.3之间

### 强度变换参数 (intensity)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `intensity_operations` | list | ['scale', 'shift'] | 变换操作列表 |
| `intensity_scale_range` | tuple | (0.9, 1.1) | 缩放范围 |
| `intensity_shift_range` | tuple | (-50, 50) | 偏移范围(HU) |
| `intensity_gamma_range` | tuple | (0.9, 1.1) | Gamma校正范围 |
| `intensity_contrast_range` | tuple | (0.9, 1.1) | 对比度调整范围 |

## 预置配置文件

位于 `config/augment_configs/`：

- `aug_flip_only.txt` - 仅翻转增强
- `aug_noise_only.txt` - 仅噪声增强
- `aug_combined.txt` - 组合增强（推荐）
- `aug_aggressive.txt` - 激进增强

## 集成到Dataset

### 方式1：继承原有Dataset类（推荐）

```python
from augmentation import AugmentFactory
from make_dataset import train_Dataset as BaseTrainDataset

class TrainDatasetWithAugment(BaseTrainDataset):
    def __init__(self, img_list, augment_config=None):
        super().__init__(img_list)
        self.augmenter = AugmentFactory.create(augment_config or {})
    
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        case_name, img, mask = result
        
        # 处理维度
        if img.ndim == 4:
            img_squeeze = img[0]
            mask_squeeze = mask[0] if mask.ndim == 4 else mask
        else:
            img_squeeze, mask_squeeze = img, mask
        
        # 应用增强
        img_aug, mask_aug = self.augmenter(
            img_squeeze, mask_squeeze, is_training=True
        )
        
        # 恢复维度
        if img.ndim == 4:
            img_aug = img_aug[np.newaxis]
            if mask.ndim == 4:
                mask_aug = mask_aug[np.newaxis]
        
        return [case_name, img_aug, mask_aug]
```

### 方式2：在训练脚本中动态添加

```python
from augmentation import AugmentFactory

# 加载配置
config = opt.__dict__  # 从主配置获取
augmenter = AugmentFactory.create(config)

# 在数据加载后应用增强
for batch in train_loader:
    case_name, img, mask = batch
    
    # 对每个样本应用增强
    img_aug, mask_aug = augmenter(img, mask, is_training=True)
    
    # 继续训练...
```

## 设计原则

1. **不修改现有代码** - 所有增强功能通过新模块实现
2. **配置驱动** - 通过配置灵活选择增强策略
3. **训练/验证分离** - `is_training`参数控制是否应用增强
4. **LR/HR同步** - 空间变换时LR和HR同步变换，强度变换仅影响LR
5. **模块化设计** - 每个增强策略独立实现，易于扩展

## 注意事项

1. **CT影像特性**：Z轴（深度）通常不应翻转，保持层间顺序
2. **计算资源**：弹性形变计算量大，建议适当降低概率或仅在后期训练使用
3. **强度变换**：CT值有物理意义（HU），变换幅度不宜过大
4. **随机种子**：设置`random_state`可确保结果可复现
