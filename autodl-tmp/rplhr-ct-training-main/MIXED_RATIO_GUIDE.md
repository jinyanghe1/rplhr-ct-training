# 混合Ratio训练指南 (Mixed Ratio Training Guide)

## 概述

本指南介绍如何使用支持**混合超分比例**的训练代码，允许在同一份训练数据中包含不同ratio的数据（如4x和5x混合训练）。

## 核心问题与解决方案

### 原始代码的限制

原始代码存在以下硬编码问题：
1. 数据加载模块 (`in_model.py`) 中硬编码了 `* 5` 计算HR位置
2. 训练脚本 (`train.py`) 中硬编码了 `5 * tmp_pos_z + 3` 拼接预测结果
3. 路径查找固定为 `5mm/1mm` 命名

### 解决方案

创建了以下新模块：

| 模块 | 功能 |
|------|------|
| `utils/in_model_mixed.py` | 支持动态ratio的数据加载 |
| `train_mixed.py` | 支持混合ratio的训练脚本 |
| `case_ratio_config.json` | 每个case的ratio配置 |

## 快速开始

### 1. 自动检测Ratio

```bash
cd code/
python utils/in_model_mixed.py ../data/cleaned_final/
```

这会自动：
- 读取所有case的LR和HR数据
- 计算每个case的实际ratio
- 生成 `case_ratio_config.json` 配置文件

### 2. 训练（使用混合Ratio）

#### 使用 thick/thin 命名（cleaned_final数据集）

```bash
python train_mixed.py train \
    --path_key cleaned_final \
    --net_idx TVSRN_mixed \
    --gpu_idx 0 \
    --lr_dir thick \
    --hr_dir thin
```

#### 使用 5mm/1mm 命名（RPLHR-CT-tiny数据集）

```bash
python train_mixed.py train \
    --path_key RPLHR-CT-tiny \
    --net_idx TVSRN_mixed \
    --gpu_idx 0 \
    --lr_dir 5mm \
    --hr_dir 1mm
```

## 配置文件详解

### case_ratio_config.json

```json
{
  "case_001": 4,
  "case_002": 5,
  "case_003": 4,
  "case_004": 5
}
```

- **key**: case名称（与文件名一致，不含.nii.gz）
- **value**: 该case的超分ratio（整数）

### 配置优先级

1. **case_ratio_config.json**（如果存在且包含该case）
2. **全局 opt.ratio**（从default.txt读取）
3. **默认值 5**

## 支持的Ratio

### 已测试的Ratio

| 输入 | 输出 | Ratio | 应用场景 |
|------|------|-------|----------|
| 5mm | 1mm | 5x | 标准超分 |
| 4mm | 1mm | 4x | 较少使用 |
| 5mm | 1.25mm | 4x | 如果需要1.25mm分辨率 |

### Ratio与模型输出的关系

模型输出维度计算公式：
```
output_z = (input_z - 1) * ratio + 1
```

例如：
- input_z=4, ratio=5 → output_z=16
- input_z=4, ratio=4 → output_z=13

## 数据目录结构

### 方案1: thick/thin 命名（推荐，更通用）

```
cleaned_final/
├── train/
│   ├── thick/          # 低分辨率（4mm或5mm）
│   │   ├── case_001.nii.gz
│   │   └── case_002.nii.gz
│   └── thin/           # 高分辨率（1mm）
│       ├── case_001.nii.gz
│       └── case_002.nii.gz
├── val/
│   ├── thick/
│   └── thin/
└── case_ratio_config.json
```

### 方案2: 5mm/1mm 命名（原始命名）

```
RPLHR-CT-tiny/
├── train/
│   ├── 5mm/
│   └── 1mm/
├── val/
│   ├── 5mm/
│   └── 1mm/
└── case_ratio_config.json
```

## 关键代码修改说明

### 1. 数据加载 (in_model_mixed.py)

```python
# 原代码（硬编码5x）
mask_z_s = z_s * 5 + 3
mask_z_e = (z_e - 1) * 5 - 2

# 新代码（动态ratio）
def calculate_output_slices(z_s, z_e, ratio, offset=3):
    out_z_s = z_s * ratio + offset
    out_z_e = (z_e - 1) * ratio - (ratio - offset - 1)
    return out_z_s, out_z_e
```

### 2. 验证拼接 (train_mixed.py)

```python
# 原代码（硬编码5x）
pos_z_s = 5 * tmp_pos_z + 3

# 新代码（动态ratio）
case_ratio = in_model.get_case_ratio(case_name, opt.path_img, 'val')
offset = case_ratio // 2
pos_z_s = calculate_val_position(tmp_pos_z, case_ratio, offset)
```

## 注意事项

### 1. 模型架构限制

虽然数据加载支持混合ratio，但**模型架构本身** (`model_TransSR.py`) 仍然使用全局 `opt.ratio`：

```python
self.out_z = (opt.c_z - 1) * opt.ratio + 1
```

**解决方案：**
- 如果所有数据的ratio相近（如都是4x或5x），使用多数ratio作为全局配置
- 如果ratio差异大，建议分别训练不同ratio的模型

### 2. 训练建议

| 场景 | 建议 |
|------|------|
| 大部分5x，少量4x | 使用 `ratio=5` 全局配置，4x数据也能正常训练 |
| 一半4x，一半5x | 分别训练两个模型，或尝试ratio=4.5（实验性）|
| 统一ratio | 使用原始代码即可 |

### 3. 验证指标计算

混合ratio训练时，验证指标（PSNR/SSIM）的计算：
- 每个case使用自己的ratio计算拼接位置
- 边界裁剪根据ratio动态调整（`border = max(case_ratio, 5)`）

## 故障排除

### 问题1: "找不到LR数据"

**原因**: 目录命名不匹配

**解决**: 检查 `--lr_dir` 和 `--hr_dir` 参数

```bash
# 查看可用目录
ls ../data/cleaned_final/train/
# 输出: thick  thin

# 然后使用
--lr_dir thick --hr_dir thin
```

### 问题2: "Shape mismatch"

**原因**: ratio配置与实际数据不匹配

**解决**: 运行自动检测脚本重新生成配置

```bash
python utils/in_model_mixed.py ../data/cleaned_final/
```

### 问题3: 拼接出现artifact

**原因**: offset计算不正确

**解决**: 检查 `calculate_output_slices` 函数中的offset参数

## 进阶：自定义Ratio检测逻辑

如果需要更复杂的ratio检测（如从DICOM header读取slice thickness），可以修改 `in_model_mixed.py` 中的 `auto_detect_ratios` 函数：

```python
def auto_detect_ratios(img_path, subset='train', lr_dir='thick', hr_dir='thin'):
    # 自定义逻辑
    # 例如：读取DICOM的SliceThickness标签
    # 计算 ratio = HR_thickness / LR_thickness
    pass
```

## 总结

混合ratio训练的关键修改：
1. ✅ 数据加载支持动态ratio
2. ✅ 验证拼接支持动态ratio
3. ✅ 灵活的路径命名（thick/thin 或 5mm/1mm）
4. ⚠️ 模型架构仍使用全局ratio（需根据主要数据设置）

通过这些修改，你可以在同一份训练中混合使用不同超分比例的数据！
