# 方案A代码审查报告

## 一、当前代码问题分析

### 1. model_TransSR.py - 裁剪逻辑写死
**Line 174**
```python
return x_out[:, :, 3:-3]  # 固定裁剪6层，输出7层
```
**问题**：无法适配不同c_z配置。当c_z=6, ratio=4时，需要输出16层（裁剪5层），但代码硬编码为3:-3。

**建议修改**：
```python
# 动态裁剪：根据c_z和ratio决定裁剪层数
output_depth = x_out.shape[2]
expected_out = (opt.c_z - 1) * opt.ratio + 1  # 21层
if expected_out == 21:  # c_z=6, ratio=4
    return x_out[:, :, 2:-3]  # 裁剪5层，输出16层
else:
    return x_out[:, :, 3:-3]  # 默认裁剪6层
```

### 2. in_model_xuanwu.py - 多处硬编码问题

**问题A：ratio硬编码为5（Line 91）**
```python
ratio = 5  # ❌ 应该使用 opt.ratio
actual_ratio = 4
```
**影响**：即使config设置ratio=4，这里强制使用5，导致尺寸计算错误。

**问题B：插值逻辑（Line 113-116）**
```python
hr_target_z = _get_hr_target_shape(opt.c_z, ratio)  # 使用ratio=5
if crop_mask.shape[0] != hr_target_z:
    crop_mask = _interpolate_to_shape(crop_mask, target_shape, order=3)  # 16层→10层
```
**影响**：HR从真实16层插值到10层，丢失6层信息。

**问题C：HR裁剪逻辑（Line 106-108）**
```python
thin_z_s = z_s * actual_ratio  # z_s * 4
thin_z_e = z_e * actual_ratio  # (z_s+4) * 4 = 16层
```
当c_z=6时，应该是z_e * actual_ratio = (z_s+6)*4 = 24层，但这里还是按c_z=4计算。

### 3. trainxuanwu.py - 验证时插值

**问题A：硬编码model_ratio=5（Line 239）**
```python
actual_ratio = 4
model_ratio = 5   # ❌ 应该使用 opt.ratio
z_zoom_factor = model_ratio / actual_ratio  # = 1.25
target_z = int(y.shape[0] * z_zoom_factor)  # 16→20层（错误）
```

**问题B：验证时zoom插值（Line 268-272）**
```python
if y_pre.shape[0] != y.shape[0]:
    y = zoom(y, zoom_factors, order=3)  # 真实HR被插值，PSNR失真
```

**问题C：pos_z_s计算（Line 259）**
```python
ratio = getattr(opt, 'ratio', 5)  # 这里用了opt.ratio，但上面初始化y_pre时用了硬编码5
pos_z_s = ratio * tmp_pos_z + 3
```
不一致！初始化y_pre用model_ratio=5，但填充时用opt.ratio。

## 二、实施架构适配的修改清单

| 文件 | 行号 | 当前代码 | 建议修改 |
|------|------|---------|---------|
| config/default.txt | 75 | c_z = 4 | c_z = 6（新配置xuanwu_ratio4.txt） |
| config/default.txt | 79-82 | vc_z = 4 | vc_z = 6 |
| model_TransSR.py | 174 | return x_out[:,:,3:-3] | 动态裁剪逻辑 |
| in_model_xuanwu.py | 91 | ratio = 5 | 删除，使用opt.ratio |
| in_model_xuanwu.py | 110-116 | 插值HR到目标尺寸 | 删除插值逻辑 |
| in_model_xuanwu.py | 106-108 | thin_z_e = z_e * 4 | 确保与c_z一致 |
| trainxuanwu.py | 238-244 | 硬编码model_ratio=5 | 使用opt.ratio |
| trainxuanwu.py | 267-272 | zoom插值y | 删除插值 |
| trainxuanwu.py | 259 | pos_z_s计算 | 确保一致 |

## 三、验证方案A的关键测试

在修改代码后，应验证以下维度：

**测试1：模型输出维度**
```
config: c_z=6, ratio=4
input: [1, 1, 6, 256, 256]
expected out_z: (6-1)*4+1 = 21
expected output: 21 - 5 = 16层（裁剪2:-3）
✅ 输出应为 [1, 1, 16, 256, 256]
```

**测试2：数据加载**
```
train: thick裁剪6层 → thin应裁剪24层（不插值）
label shape: [6, 256, 256]  # thick
target shape: [24, 256, 256]  # thin（真实HR）
```

**测试3：训练时对齐**
```
y_pre: [1, 1, 16, 256, 256]  # 模型输出
y: [1, 1, 24, 256, 256]  # 真实HR（24层）
对齐：取y中间16层（offset=4）→ y[:,:,4:20,:,:]
```

**测试4：验证时不插值**
```
y_pre_full: 模型输出拼接后的体积（应与真实HR同尺寸或对齐后比较）
y: 原始HR（不插值）
比较：重叠区域直接计算PSNR
```

## 四、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| c_z=6时显存OOM | 中 | 训练失败 | 减小batch_size或vc_z |
| 裁剪后边界效应 | 低 | PSNR下降 | 调整裁剪范围（2:-3 vs 3:-3） |
| 数据对齐错误 | 中 | 训练崩溃 | 添加assert检查shape |
| 1:5模型兼容性 | 低 | 旧配置失效 | 保留原配置，新建xuanwu_ratio4 |

## 五、审查结论

当前代码无法直接运行方案A，必须进行以下修改：

1. 删除所有硬编码的ratio=5
2. 删除训练和验证时的zoom插值
3. 修改模型裁剪逻辑支持动态裁剪
4. 创建新的配置文件（c_z=6, ratio=4）

**修改后预期**：
- Val PSNR从9.9 dB提升到25-30 dB
- 模型直接学习真实 thick↔thin 映射
- 无插值信息损失
