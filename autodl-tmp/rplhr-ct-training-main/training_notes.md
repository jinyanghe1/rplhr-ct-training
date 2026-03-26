# 宣武数据集训练记录

> ⚠️ **CRITICAL MEMORY**: 每次对训练代码进行修改或运行训练后，**必须**同步更新此文档！
> 
> **路径**: `/root/autodl-tmp/rplhr-ct-training-main/training_notes.md`
>
> **更新时机**:
> - 修改 `trainxuanwu.py` / `in_model_xuanwu.py` / `augmentation.py` 等训练相关代码
> - 运行任何训练（包括快速验证和完整训练）
> - 修改数据增强配置
>
> **记录内容**: 修改原因、修改内容、训练结果、验证指标、遇到的问题、下一步计划

---

## 训练会话 #1 - 2026-03-26

### 基本信息
- **日期**: 2026-03-26
- **数据集**: 宣武数据集 (Xuanwu Dataset)
- **数据路径**: `/root/autodl-tmp/rplhr-ct-training-main/data/dataset01_xuanwu`
- **配置**: 保守数据增强 + 插值适配 (4:1 -> 5:1)

### 训练配置
```python
epoch = 50 (实际运行38后提前终止)
ratio = 5 (模型期望) / 实际 4 (数据本身)
use_augmentation = True
aug_prob = 0.5
clip_ct = True
min_hu = -1024
max_hu = 3071
normalize_ct = False  # 关键问题！
optim = 'AdamW'
lr = 0.0003
```

### 关键修改
1. `augmentation.py`: 添加 `_ensure_shape` 尺寸保护
2. `in_model_xuanwu.py`: 宣武数据集专用加载器，验证集禁用增强
3. `trainxuanwu.py`: 训练脚本，含插值适配逻辑

### 训练结果

#### 训练损失
| Epoch | Loss | 趋势 |
|-------|------|------|
| 1 | 626.49 | 初始 |
| 10 | 177.25 | 快速下降 |
| 20 | 154.97 | 平台期 |
| 29 | 149.68 | 最低 |
| 38 | 155.62 | 结束 |

#### 验证指标 (异常！)
| 指标 | 数值 | 状态 |
|------|------|------|
| PSNR | -60.71 dB | ❌ 应为正值 |
| MSE | 1,192,103 | ❌ 过高 |
| SSIM | 0.00002 | ❌ 接近0 |

### 问题诊断

#### 根因 #1: 数据尺度不匹配 (Critical)
- **症状**: PSNR为负，MSE达10^6
- **原因**: 
  - 模型输出被 `torch.clamp(..., 0, 1)` 限制在[0,1]
  - 但标签(HR)保持原始HU值 [-1024, 3071]
  - 尺度差异导致巨大误差

#### 根因 #2: 比例适配
- 宣武数据集比例: thick:thin = 4:1
- 模型期望: 5:1
- 通过插值适配，但可能影响解剖结构

#### 根因 #3: 增强策略
- 强度变换可能破坏HU值物理意义
- 建议仅保留几何增强

### 输出文件
```
train_log/dataset01_xuanwu/xuanwu_50epoch/
├── DATASET_INFO.txt       # 数据集标记
└── training_history.csv   # 训练历史
```

### 下一步行动计划
- [x] 最小化修复 (归一化 + 禁用强度增强)
- [x] 10 epoch快速验证
- [x] 复盘修复效果
- [ ] 决定继续50 epoch或进一步调整

---

## 修复 #1 验证结果 - 2026-03-26

### 修复内容
1. ✅ 启用数据归一化 (`normalize_ct=True`, window_center=40, window_width=400)
2. ✅ 使用仅几何增强 (`GEOMETRY_ONLY_AUG`)
3. ✅ 禁用所有强度变换

### 训练结果 (5 Epochs)

#### 训练损失
| Epoch | Loss | 趋势 |
|-------|------|------|
| 1 | 0.4813 | 初始 |
| 2 | 0.1363 | ↓ 71.6% |
| 3 | 0.1090 | ↓ 20.0% |
| 4 | 0.0983 | ↓ 9.8% |
| 5 | 0.1031 | ↑ 4.9% (轻微反弹) |

#### 验证指标
| Epoch | PSNR | MSE | SSIM |
|-------|------|-----|------|
| 1-5 | -60.70 dB | ~1,192,000 | ~0.00002 |

**关键发现**: 训练损失大幅下降，但验证指标依然异常！

### 问题诊断

#### 根因 #1: 训练与验证数据尺度不一致 (新发现)
- **训练时**: 数据经过归一化到 [0, 1] 范围
- **验证时**: `get_val_img` 返回的是原始未归一化的 mask
- **结果**: 模型在归一化数据上训练，但在原始尺度上验证

#### 根因 #2: 验证集HR未插值匹配
- 训练时对HR进行了插值 (16层 -> 10层)
- 验证时返回完整的HR，尺寸不匹配

### 修复效果评估与决策
**训练**: ✅ 损失下降明显，归一化有效  
**验证**: ❌ 指标依然异常，需要修复验证流程  
**决策**: ⚠️ 效果不达标，需进一步修复验证流程后再进行完整训练

---

## 下一步修复方案 - 2026-03-26

### 问题定位
验证流程存在两个关键问题：
1. **尺度不一致**: 训练时归一化，验证时未归一化
2. **尺寸不匹配**: 验证时HR未插值到与模型输出相同的尺寸

### 修复 #2 计划

#### 修改点 #1: 验证集数据加载器
文件: `in_model_xuanwu.py` - `get_val_img()`
- 对返回的 `tmp_mask` 进行归一化
- 或者：在训练脚本中对验证HR进行归一化

#### 修改点 #2: 验证集HR插值
文件: `trainxuanwu.py` - 验证循环
- 对验证时的HR进行插值，匹配模型输出尺寸
- 确保插值前后数值范围一致

#### 修改点 #3: 归一化参数一致性
- 训练和验证使用相同的窗宽窗位参数
- 建议: window_center=40, window_width=400 (软组织窗)

### 验证计划
- [ ] 修复验证流程
- [ ] 5 epoch快速验证
- [ ] 确认PSNR为正值且持续改善
- [ ] 效果良好 → 继续50 epoch完整训练

---

## 修复 #2 - 验证集归一化

### 执行摘要 - 2026-03-26

#### 决策结果
| 条件 | 结果 | 行动 |
|------|------|------|
| 训练损失下降 | ✅ 满足 | 继续 |
| 验证指标正常 | ❌ 不满足 | **需修复 #2** |
| 数据一致性 | ❌ 不满足 | **需修复 #2** |

**最终决策**: 先执行修复 #2（验证集归一化），5 epoch验证通过后再进行50 epoch完整训练。

#### 修复内容
文件: `trainxuanwu.py` - 验证循环
```python
# 修复 #2: 对HR进行归一化，与训练时保持一致
if getattr(opt, 'normalize_ct', False):
    window_center = getattr(opt, 'window_center', 40)
    window_width = getattr(opt, 'window_width', 400)
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    y = np.clip((y - min_val) / (max_val - min_val + 1e-8), 0, 1)
```

#### 建议验证流程
```bash
# 1. 运行5 epoch快速验证
python trainxuanwu.py train \
    --net_idx="xuanwu_fix2_5epoch" \
    --path_key="dataset01_xuanwu" \
    --epoch=5 \
    --use_augmentation=True \
    --normalize_ct=True \
    --window_center=40 \
    --window_width=400

# 2. 检查验证指标
# 预期: PSNR > 0 dB 且持续改善
```

### 关键文件清单
```
code/
├── trainxuanwu.py             # 已修复验证归一化
├── utils/
│   ├── augmentation.py        # 已添加_ensure_shape
│   ├── augmentation_config.py # 已添加GEOMETRY_ONLY_AUG
│   └── in_model_xuanwu.py     # 已修复训练归一化
└── ...
```

---

### 修复 #2 验证结果

#### 5 Epoch 验证数据
| Epoch | Train Loss | Val PSNR | Val SSIM | 趋势 |
|-------|------------|----------|----------|------|
| 1 | 0.4813 | 9.84 dB | 0.480 | 基线 ✅ |
| 2 | 0.1363 | 11.38 dB | 0.695 | ↑ 1.54 dB |
| 3 | 0.1089 | 11.72 dB | 0.761 | ↑ 0.34 dB |
| 4 | 0.0982 | 11.82 dB | 0.774 | ↑ 0.10 dB |
| 5 | 0.1031 | 11.96 dB | 0.778 | ↑ 0.14 dB |

#### 结论
- ✅ **PSNR为正值**: 9.84 → 11.96 dB (改善2.12 dB)
- ✅ **SSIM持续改善**: 0.48 → 0.78
- ✅ **训练损失下降**: 0.48 → 0.10
- ✅ **验证指标稳定**: 无明显过拟合

**决策**: 修复 #2 成功！可以进行50 epoch完整训练。

---

*📝 动态更新要求: 每次对训练代码进行修改或运行训练后，需同步更新此文档*
*📝 Memory: 训练代码修改/训练运行 → 必须更新 training_notes.md*

---

## 修复记录

### 修复 #1 - 数据归一化
**时间**: 待执行
**修改**:
1. 启用 `normalize_ct = True`
2. 或使用窗宽窗位将数据归一化到[0,1]

### 修复 #2 - 增强策略调整
**时间**: 待执行
**修改**:
1. 禁用 `intensity_scale_prob`, `intensity_shift_prob`, `contrast_prob`
2. 仅保留几何增强 (flip, rotate, shift)

---

*最后更新: 2026-03-26*
