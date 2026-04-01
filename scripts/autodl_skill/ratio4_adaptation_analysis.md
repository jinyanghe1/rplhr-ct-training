# Ratio=4 架构适配分析

## 问题概述

宣武数据集 (1.25mm:5mm) 实际 ratio=4，但原模型设计为 ratio=5。

当 ratio=4 时，模型 Z 维度计算发生变化，导致 Swin Transformer 的 window_partition 失败。

## 核心计算公式

### 1. 输出 Z 维度计算
```python
# model_TransSR.py line 61
self.out_z = (opt.c_z - 1) * opt.ratio + 1
```

| ratio | c_z | out_z 计算 | out_z 值 |
|-------|-----|-----------|---------|
| 5 | 4 | (4-1)×5+1 | **16** |
| 4 | 4 | (4-1)×4+1 | **13** |

### 2. 问题原因
- Decoder_T 使用 `img_size=(out_z, img_size)`
- Swin Transformer 的 `window_partition` 要求输入维度能被 `window_size` 整除
- 当 TD_Tw=4，out_z=13 时：`13 % 4 = 1`，无法整除 → RuntimeError

## 需要修改的参数

### config/default.txt

| 参数 | ratio=5 时 | ratio=4 时 | 修改 |
|------|------------|-----------|------|
| `ratio` | 5 | **4** | ✅ 必需 |
| `c_z` | 4 | 4 | ❌ 不变（输入 LR 的 Z 维度） |
| `TD_Tw` | 4 | **1** | ✅ 必需（避免 window_size 不整除） |
| `TD_Tl` | 1 | 1 | ❌ 不变 |
| `TD_Td` | 4 | 4 | ❌ 不变 |

### 计算验证

```python
# ratio=4, c_z=4, TD_Tw=1
out_z = (4-1)*4+1 = 13

# Decoder_T 使用:
img_size = (13, 256)
window_size = 1  # TD_Tw=1

# window_partition 要求: H % window_size == 0, W % window_size == 0
# 13 % 1 = 0 ✅ OK
# 256 % 1 = 0 ✅ OK
```

## 受影响的文件

### 1. `config/default.txt`
- **必须修改**: `ratio = 4`
- **必须修改**: `TD_Tw = 1`
- **可选**: 如果 c_z 需要调整

### 2. `code/net/model_TransSR.py`
- **不需修改**: 计算逻辑正确，会自动适应新的 ratio
- line 61: `self.out_z = (opt.c_z - 1) * opt.ratio + 1` - 使用配置中的 ratio

### 3. `code/utils/in_model.py` / `code/utils/in_model_dicom.py`
- **不需修改**: 使用 `opt.c_z` 进行裁剪，与 ratio 无关
- Z 维度裁剪是针对输入 LR 图像，不涉及上采样比例

### 4. `code/utils/augmentation.py`
- **不需修改**: `z_ratio` 是从实际数据计算的，不是从配置读取
- line 131: `z_ratio = hr_img.shape[0] // lr_img.shape[0]`

## 完整修改清单

### 只需修改一个文件: `config/default.txt`

```diff
### data
dim = 1
-ratio = 5
+ratio = 4

### Decoder config
TD_p = 8
TD_s = 1

TD_Tw = 4
+TD_Tw = 1
TD_Tl = 1
TD_Td = 4
```

## 验证方法

训练启动后应看到：
```
Training started with ratio=4, out_z=13
```

不再出现：
```
RuntimeError: shape '[1, 3, 4, 64, 4, 1]' is invalid for input of size 3328
```

## 指标预期

修复后训练指标应恢复正常范围：
- Val PSNR: 20-40 dB（正值，不是 -61dB）
- Val SSIM: 0.5-0.9
- Val MSE: <100

## 相关文档

- 模型代码: `code/net/model_TransSR.py`
- Swin 实现: `code/net/swin_utils.py`
- 训练脚本: `code/trainxuanwu.py`
- 数据裁剪: `code/utils/in_model.py`
