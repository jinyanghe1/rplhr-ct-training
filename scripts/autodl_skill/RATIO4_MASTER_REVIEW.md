# RPLHR-CT Ratio=4 架构适配 - 主审查报告

> **审查时间**: 2026-04-01 14:30  
> **下次检查**: 2026-04-01 14:40 (每10分钟)  
> **目标**: Val PSNR > 25 dB (理想 > 30 dB)  
> **当前**: Best PSNR = 11.96 dB ❌

---

## 一、当前训练情况

### 1.1 训练状态
| 项目 | 状态 | 详情 |
|------|------|------|
| **训练进程** | ❌ 停止 | 无训练运行中 |
| **最后错误** | 🔴 `AttributeError` | `non_model` 模块缺少 `cal_mse` 函数 |
| **GPU状态** | ✅ 可用 | RTX PRO 6000 Blackwell (95GB) |
| **TensorBoard** | ✅ 运行 | 端口6007 |

### 1.2 历史Benchmark

| 实验 | Epochs | Best PSNR | 状态 |
|------|--------|-----------|------|
| xuanwu_fix2_5epoch | 5 | **11.96 dB** ⭐ | 历史最佳 |
| xuanwu_50epoch | 30 | 9.92 dB | 饱和 |
| xuanwu_fix1_10epoch | 10 | -60.7 dB | 数据异常 |

**趋势分析**:
```
Epoch  1-5:   8.66 → 9.80  (+1.14dB, 快速上升)
Epoch  6-10:  9.80 → 9.92  (+0.12dB, 峰值@epoch9)
Epoch 11-30:  9.87 → 9.78  (完全饱和，无提升)
```

**结论**: 训练在第9epoch后饱和，PSNR远未达到25dB目标（差距13dB）。

---

## 二、代码版本状态

### 2.1 版本对比
| 位置 | 最新提交 | 状态 |
|------|----------|------|
| **本地** | `6cdbb3d` | 领先 |
| **远程** | `e672899` | 落后3个提交 |

### 2.2 关键差异
本地包含**架构适配方案A**的实现（commit `acff28b`）:
- ✅ `config/xuanwu_ratio4.txt` (c_z=6, ratio=4)
- ✅ `model_TransSR.py` 动态裁剪
- ✅ `trainxuanwu.py` 尺寸对齐
- ✅ `in_model_xuanwu.py` 无插值

**⚠️ 需要同步**: 远程服务器需要 `git pull` 获取最新代码。

---

## 三、架构核心问题

### 3.1 问题分类

#### 🔴 P0 - 阻断性问题（立即修复）

| # | 文件 | 函数 | 行号 | 问题 | 影响 |
|---|------|------|------|------|------|
| 1 | `utils/non_model.py` | `cal_mse` | - | **函数不存在** | 训练崩溃 |
| 2 | `utils/in_model_xuanwu.py` | `get_train_img` | ~92 | `actual_ratio = opt.ratio` 应为4 | HR裁剪深度错误 |
| 3 | `net/model_TransSR.py` | `forward` | 174 | 硬编码 `return x_out[:,:,3:-3]` | 输出层数错误 |
| 4 | `config/default.txt` | - | - | `ratio=4` 与代码期望冲突 | 配置不一致 |

#### 🟡 P1 - 高优先级

| # | 文件 | 函数 | 行号 | 问题 |
|---|------|------|------|------|
| 5 | `trainxuanwu.py` | `train` | ~239 | `model_ratio = 5` 硬编码 |
| 6 | `trainxuanwu.py` | `train` | ~267-273 | 验证时对HR进行zoom插值 |
| 7 | `in_model_xuanwu.py` | `_get_hr_target_shape` | 50 | 默认参数 `ratio=5` |
| 8 | `trainxuanwu.py` | `train` | ~259 | `pos_z_s = ratio * tmp_pos_z + 3` 硬编码+3 |

### 3.2 核心矛盾

```
配置层面: ratio = 4 (上采样比例)
代码层面: 多处硬编码 ratio = 5 (模型期望比例)
数据集:   actual_ratio = 4 (宣武数据集真实比例 1.25mm:5mm)

这导致三层不一致！
```

**正确的理解应该是**:
- `opt.ratio` = 模型设计上采样比例 = 5（保持与原模型一致）
- `actual_ratio` = 数据集真实比例 = 4（宣武数据集）
- 宣武数据集需要插值或架构适配来弥合 4→5 的差异

---

## 四、修复路线图

### 阶段1: 紧急修复（立即执行）

#### 4.1 修复 `cal_mse` 函数缺失
```python
# utils/non_model.py - 添加以下函数

def cal_mse(img1, img2):
    """计算MSE"""
    mse = np.mean((img1 - img2) ** 2)
    return mse
```

#### 4.2 修复 `in_model_xuanwu.py` 的ratio混淆
```python
# Line 92: 修改前
actual_ratio = opt.ratio  # BUG: 当opt.ratio=5时，裁剪20层而非16层

# Line 92: 修改后  
actual_ratio = 4  # 宣武数据集真实比例
```

#### 4.3 修复 `model_TransSR.py` 硬编码裁剪
```python
# Line 174: 修改前
return x_out[:, :, 3:-3]

# Line 174: 修改后 - 动态裁剪
expected_out = (opt.c_z - 1) * opt.ratio - 5  # (4-1)*5-5 = 10
if x_out.shape[2] > expected_out:
    # 裁剪边界
    trim = (x_out.shape[2] - expected_out) // 2
    return x_out[:, :, trim:-trim]
return x_out[:, :, 3:-3]
```

### 阶段2: 配置统一（重要）

#### 4.4 明确ratio语义
方案A: **保持ratio=5，通过插值适配**（当前方案）
- 优点: 1:5模型兼容性好
- 缺点: 有插值损失

方案B: **改为ratio=4，修改架构**（架构适配）
- 优点: 无插值损失
- 缺点: 需要创建新网络配置，与原模型不兼容

**建议**: 如果目标是>25dB，建议实施方案B（无插值）。

### 阶段3: 验证逻辑修复

#### 4.5 修复验证时插值
```python
# trainxuanwu.py Line 267-273: 删除或修改

# 当前（有问题）:
if y_pre.shape[0] != y.shape[0]:
    y = zoom(y, zoom_factors, order=3)  # 插值HR

# 应该:
# 对齐到重叠区域，不插值HR
min_z = min(y_pre.shape[0], y.shape[0])
y_pre_eval = y_pre[:min_z]
y_eval = y[:min_z]
```

---

## 五、预期验收结果

### 5.1 修复后验证清单

| 检查项 | 验证方法 | 通过标准 |
|--------|----------|----------|
| 训练不崩溃 | `python trainxuanwu.py train --epoch=1` | 完成1个epoch无错误 |
| 输出尺寸正确 | DEBUG打印 `y_pre.shape` | [1,1,10,256,256] (c_z=4,ratio=5) |
| 无插值训练 | 检查 `in_model_xuanwu.py` | 无 `_interpolate_to_shape` 调用 |
| PSNR提升 | Epoch 5 benchmark | > 15 dB |
| 最终目标 | Epoch 50 benchmark | **> 25 dB** |

### 5.2 时间线估算

| 阶段 | 任务 | 预估时间 | 负责人 |
|------|------|----------|--------|
| 1 | 修复cal_mse + ratio混淆 | 30min | 开发者 |
| 2 | 同步代码到远程 + 测试 | 20min | Agent |
| 3 | 启动训练 + 监控前5epoch | 2h | Agent |
| 4 | 评估Epoch 5结果 | 10min | Agent |
| 5 | 如达标，继续训练到50epoch | 10h | Auto |
| 6 | 最终验收 | 10min | Agent |

---

## 六、Action Items

### 立即执行（下次检查前）
- [ ] 修复 `utils/non_model.py` 添加 `cal_mse` 函数
- [ ] 修复 `in_model_xuanwu.py` Line 92: `actual_ratio = 4`
- [ ] 提交并push到GitHub
- [ ] 远程执行 `git pull`

### 下次检查（10分钟后）验证
- [ ] 训练能否启动不崩溃
- [ ] DEBUG输出尺寸是否正确
- [ ] 前3个epoch的loss下降情况

---

## 七、参考文档

| 文档 | 用途 | 状态 |
|------|------|------|
| `RATIO4_ARCHITECTURE_ADAPTATION.md` | 详细适配方案 | 参考 |
| `README.md` | 基础使用说明 | 保留 |
| **本文件** | 主审查报告（最新） | 每次检查更新 |

---

*最后更新: 2026-04-01 14:30*  
*下次检查: 2026-04-01 14:40*  
*检查频率: 每10分钟*
