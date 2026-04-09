# RPLHR-CT 项目 Roadmap v4.0

> **维护时间**: 2026-04-08
> **策略更新**: Phase A 预训练完成 (PSNR=31.55@E65)，进入 Phase B Finetune
> **目标**: PSNR > 30 dB (预训练 + finetune)
> **当前基线**: PSNR=31.55 dB (pretrain_ratio5, Epoch 65)
> **当前任务**: Phase B Finetune 设计与实施

---

## ⚠️ 重要更新 (2026-04-08)

**Phase A 预训练已完成**:
- pretrain_ratio5 训练至 Epoch 65, PSNR=31.55, SSIM=0.744
- 使用32GiB公开数据集(rplhr_ct_32G), ratio=5, cosine LR
- ✅ 超过30dB目标

**参数分布关键发现**:
- x_patch_mask = 6.29M (72.5%) — 可学习 mask token，是大参数大头
- LP + Encoder = 584K (6.7%) — 冻结此部分几乎无效果
- Decoder_T + Decoder_I + Output = 1.80M (20.8%) — 核心可训练模块
- **结论**: 只冻 LP+Encoder (7%) 无意义，需要更激进的冻结策略

**c_z 不可变**:
- c_z=4 是所有配置的统一值（default.txt, xuanwu_50epoch.txt）
- c_z 影响LP输入通道、Encoder输入维度、Decoder_I embed_dim → 改c_z预训练权重全部作废
- Ratio-Aware架构已解决ratio差异：c_z=4不变，ratio在forward中动态计算

**预训练→Finetune SSIM分析**:
- 从头训练早期SSIM~0.50 → finetune SSIM~0.75 → 证明预训练有效(+50%)
- 但从头训练30epoch SSIM=0.819 > finetune 0.73 → **finetune可能不充分**

---

## 🎯 三大优化方向

### 方向一：冻结策略实验 🔒

**背景**: 50个训练样本，8.68M总参数，需要减少可训练参数量防止过拟合。

**参数分布**:
```
┌──────────────────────────────────────────────────┐
│  模型参数分布                                      │
├────────────────────┬───────────┬──────┬───────────┤
│  模块              │ 参数量     │ 占比  │ ratio依赖 │
├────────────────────┼───────────┼──────┼───────────┤
│  LP                │ 526,336   │ 6.1% │ ❌ 无      │
│  Encoder (Swin)    │ 58,144    │ 0.7% │ ❌ 无      │
│  x_patch_mask      │ 6,291,456 │ 72.5%│ ✅ 强      │
│  Decoder_T1        │ 201,760   │ 2.3% │ 间接      │
│  Decoder_I1        │ 1,601,088 │ 18.4%│ 间接      │
│  Output (conv)     │ 161       │ 0.0% │ ❌ 无      │
├────────────────────┼───────────┼──────┼───────────┤
│  TOTAL             │ 8,678,945 │ 100% │           │
└────────────────────┴───────────┴──────┴───────────┘
```

**实验设计**: 三组冻结策略对比

| 实验ID | 冻结模块 | 冻结参数量 | 可训练参数量 | 可训练% | 说明 |
|--------|----------|-----------|-------------|---------|------|
| **Freeze-A** | LP + Encoder | 584K | 8.09M | 93.3% | 基线冻结（之前方案，几乎无效） |
| **Freeze-B** | LP + Encoder + x_patch_mask | 6.88M | 1.80M | 20.8% | ✅ 推荐方案：保留Decoder学习能力 |
| **Freeze-C** | LP + Encoder + x_patch_mask + Decoder_I | 8.48M | 0.20M | 2.3% | 激进方案：只训练Decoder_T+Output |

**Freeze-B 推荐理由**:
- x_patch_mask 是空间大查找表 [12,8,256,256]，预训练已学到通用填充模式
- ratio=4 时只用前9个mask，预训练的12个mask中前9个完全可以复用
- Decoder_T/Decoder_I 需要适应 ratio=4 的z维度，必须可训练
- 1.80M 可训练参数 / 50样本 ≈ 36K params/sample，合理范围

**Freeze-C 适用场景**:
- 如果 Freeze-B 仍然过拟合
- 极快速验证预训练权重迁移效果
- 20万参数 / 50样本 = 4K params/sample，几乎不可能过拟合

**冻结实现**:
```python
# Freeze-B: 推荐
net = model_TransSR.TVSRN().to(device)
load_dict = torch.load(pretrain_path)['net'].state_dict()
net.load_state_dict(load_dict, strict=False)

for name, param in net.named_parameters():
    if name.startswith('Encoder.') or name.startswith('LP') or name == 'x_patch_mask':
        param.requires_grad = False

# 验证
trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
print(f'Trainable: {trainable:,} ({trainable/total*100:.1f}%)')
print(f'Frozen: {frozen:,} ({frozen/total*100:.1f}%)')
```

### 方向二：训练策略优化 📈

**背景**: 0403实验已验证 EAGLELoss3D 和 EMA 对SSIM提升最有效。

| 策略 | 已验证SSIM提升 | 已验证PSNR提升 | 来源 | 风险 |
|------|---------------|---------------|------|------|
| **EAGLELoss3D** | **+0.026** | +0.10 | EXP_002 | 低 |
| **EMA (0.995)** | **+0.022** | +0.08 | EXP_005 | 低 |
| Charbonnier Loss | +0.014 | +0.10 | EXP_003 | 低 |
| 随机Flip增强 | +0.010 | +0.07 | EXP_004 | 低 |
| ~~感知损失/LPIPS~~ | - | - | - | ❌ CT幻觉 |
| ~~GAN损失~~ | - | - | - | ❌ CT幻觉 |
| ~~强弹性形变~~ | - | - | - | ❌ 破坏解剖结构 |

**推荐 Finetune 配置**:
```ini
# xuanwu_finetune_ratio4.txt
[模型]
ratio = 4                  # finetune用ratio=4
c_z = 4                    # 保持不变！c_z改了预训练权重作废
max_ratio = 5              # 兼容预训练权重
crop_margin = 3

[训练]
epoch = 80
batch_size = 1
optim = AdamW
lr = 0.00003               # 预训练lr的1/10
weight_decay = 0.0001
T_max = 80                 # cosine annealing

[损失函数]
loss_type = eagle3d         # EAGLELoss3D → SSIM最优
eagle_alpha = 0.1           # L1 + 0.1*EAGLE

[EMA]
use_ema = True
ema_decay = 0.995
ema_warmup_epochs = 10

[数据增强]
use_augmentation = True
flip_prob = 0.5             # 随机翻转
# 无弹性形变、无强度变换

[学习率策略]
warmup_epochs = 5           # 前5 epoch线性warmup
gradient_clip = 1.0         # 梯度裁剪

[归一化]
normalize_ct = True
window_center = 40
window_width = 400
```

**学习率策略详解**:
- Warmup (5 epochs): 从 0 → 3e-5 线性增长，保护预训练权重
- CosineAnnealing (T_max=80): 从 3e-5 → 3e-7 平滑衰减
- Gradient Clipping (1.0): 防止梯度爆炸，小数据集常见

### 方向三：Backbone 扩展探索 🔬

**背景**: 在TVSRN基础上探索架构改进，提升模型表达能力。

| 方向 | 预期收益 | 实现难度 | 论文创新性 | 优先级 |
|------|----------|----------|-----------|--------|
| **轴向注意力增强** | z方向长距依赖 ↑ | 中 | ★★★★ | P1 |
| **分块重建+高斯融合** | 去块效应 | 中 | ★★★ | P2 |
| **残差密集连接** | 特征复用 ↑ | 低 | ★★ | P2 |
| **多尺度特征融合** | 细节+结构 ↑ | 中 | ★★★ | P2 |
| **条件归一化(AdaIN)** | 领域适配 ↑ | 中 | ★★★★ | P3 |

#### P1: 轴向注意力增强

当前模型 z 方向只通过 MToken 的交错排列隐式建模，Decoder_T 做的是 (z,y) 平面的2D注意力。改进方向：

```
方案A: 3D Swin Transformer Block
  - 将 Decoder_T 替换为 3D 窗口注意力
  - 窗口大小: (2, 8, 8) 或 (4, 4, 4)
  - 计算/显存开销大，但 z 方向建模最充分

方案B: 分离式轴向注意力
  - z-axis attention + xy-axis attention 分离计算
  - 类似 Axial-DeepLab 的思路
  - 计算量可控，z 方向全局感受野

方案C: 交叉注意力 (Cross-Attention)
  - 已知slice (c_z个) 作为 Query
  - mask token 作为 Key/Value
  - 让已知信息显式指导未知位置的生成
  - 创新性最高，最契合 MAE 思路
```

#### P2: 分块重建 + 高斯融合

当前模型是整图推理，如果推理大体积CT需要分块。但 finetune 阶段先不做，因为 c_y=c_x=256 已足够。

#### P3: 条件归一化 (AdaIN)

在 Decoder 中加入条件归一化，让模型感知当前处理的是哪个解剖区域。这对跨部位迁移最有价值，但实现复杂度高。

---

## 📊 当前状态

### 训练历史全景

| 会话 | 数据集 | 配置 | 最佳PSNR | 最佳SSIM | 备注 |
|------|--------|------|----------|----------|------|
| #1 (0326) | xuanwu, ratio=5(插值) | 无归一化 | -60.7 | - | 验证尺度错误 |
| #2 (0326) | xuanwu, 归一化+几何增强 | normalize_ct=True | 11.96 | 0.527 | 5epoch快速验证 |
| #3 (0401) | xuanwu, ratio=4架构适配 | c_z=6, 16层输出 | 19.98 | 0.819 | 30epoch |
| #4 (0403) | xuanwu, EAGLE Loss | 3D_eagle_loss | 20.11 | 0.873 | 提升有限 |
| #5 (0408) | rplhr_ct_32G, ratio=5 | Ratio-Aware预训练 | **31.55** | 0.744 | ✅ 超过30dB目标 |

### 差距分析

```
当前: 31.55 dB (预训练完成, 腹部CT)
目标: 30+ dB (宣武头部CT finetune)

预期路径:
├── Phase A 预训练: ✅ 完成 → 31.55 dB
├── Phase B 基础finetune: ⏳ → 预期 ~25 dB (头部CT PSNR天然偏低)
├── Phase B + 冻结策略: ⏳ → SSIM 提升关键
├── Phase B + EAGLELoss3D+EMA: ⏳ → SSIM +0.03~0.05
└── Phase B + backbone扩展: ⏳ → 潜在PSNR +1~3 dB
```

### PSNR 跨部位差异说明

| 数据集 | 部位 | PSNR | SSIM | 说明 |
|--------|------|------|------|------|
| RPLHR-CT-32G | 腹部 | 31.55 | 0.744 | 预训练结果 |
| 宣武 | 头部 | ~25 (预期) | ~0.75+ | finetune后 |
| 宣武(从头训练30ep) | 头部 | 19.98 | 0.819 | 历史对比 |

**关键**: PSNR ~25在头部CT上正常，文献支持头部CT SR PSNR比腹部低2-5 dB。SSIM是更可靠的跨部位指标。

---

## 🚀 执行计划

### Phase A: 公开数据集预训练 ✅ 已完成

| Step | 任务 | 目标 | 状态 |
|------|------|------|------|
| A1 | 更新 Roadmap | 文档化 Ratio-Aware 方案 | ✅ 完成 |
| A2 | 修改 model_TransSR.py | Decoder_I max_out_z + cal_z零填充 + crop_margin | ✅ 完成 |
| A3 | 修改 train.py | 传入 ratio=opt.ratio + 参数化重建 | ✅ 完成 |
| A4 | 修改 val.py | 固定 ratio=4 + 参数化重建 | ✅ 完成 |
| A5 | 修改 test.py | ratio=opt.ratio + 参数化重建 | ✅ 完成 |
| A6 | 更新 config/default.txt | 新增 max_ratio=5, crop_margin=3 | ✅ 完成 |
| A7 | AutoDL 预训练 | PSNR > 30 dB | ✅ 完成 (31.55@E65) |
| A8 | 训练崩溃诊断 | 分析终止原因+守护方案 | ✅ 完成 |
| A9 | 安全启动脚本 | systemd守护 | ✅ 完成 |

### Phase B: 自用数据集 Finetune (当前阶段)

#### B0: 准备工作

| Step | 任务 | 目标 | 状态 |
|------|------|------|------|
| B0.1 | 创建 finetune 配置 | xuanwu + ratio=4 + c_z=4 | ⏳ 待实施 |
| B0.2 | 加载预训练权重 | 从 Phase A 迁移 | ⏳ 待实施 |
| B0.3 | 实现冻结策略代码 | Freeze-A/B/C 可切换 | ⏳ 待实施 |
| B0.4 | 实现 EAGLELoss3D | 集成到 train 脚本 | ⏳ 待实施 |
| B0.5 | 实现 EMA | 集成到 train 脚本 | ⏳ 待实施 |
| B0.6 | 实现 warmup + cosine | 集成到 train 脚本 | ⏳ 待实施 |

#### B1: 冻结策略对比实验

| Step | 任务 | 冻结策略 | 可训练参数 | 预期 | 状态 |
|------|------|----------|-----------|------|------|
| B1.1 | Freeze-A 基线 | LP+Encoder (7%) | 8.09M | 建立基线 | ⏳ |
| B1.2 | Freeze-B 推荐 | +x_patch_mask (79%) | 1.80M | SSIM↑ | ⏳ |
| B1.3 | Freeze-C 激进 | +Decoder_I (98%) | 0.20M | 防过拟合 | ⏳ |
| B1.4 | 选择最优冻结策略 | 对比PSNR/SSIM | - | - | ⏳ |

**评估标准**: 
- 主要看 SSIM（跨部位可比）
- PSNR 作为参考（头部CT天然偏低）
- 训练稳定性（loss 曲线平滑度）

#### B2: 训练策略优化

| Step | 任务 | 基于策略 | 预期收益 | 状态 |
|------|------|---------|----------|------|
| B2.1 | 加入 EAGLELoss3D | 最优冻结策略 | SSIM +0.02~0.03 | ⏳ |
| B2.2 | 加入 EMA | B2.1 | SSIM 稳定 +0.02 | ⏳ |
| B2.3 | 加入数据增强(Flip) | B2.2 | SSIM +0.01 | ⏳ |
| B2.4 | 延长训练至 80-100ep | B2.3 | SSIM 超过0.819 | ⏳ |

#### B3: Backbone 扩展实验

| Step | 任务 | 说明 | 优先级 | 状态 |
|------|------|------|--------|------|
| B3.1 | 轴向注意力增强 | z方向建模改进 | P1 | ⏳ |
| B3.2 | 分块重建+高斯融合 | 去块效应 | P2 | ⏳ |
| B3.3 | 残差密集连接 | 特征复用 | P2 | ⏳ |
| B3.4 | 条件归一化(AdaIN) | 领域适配 | P3 | ⏳ |

#### B4: 论文数据收集

| Step | 任务 | 说明 | 状态 |
|------|------|------|------|
| B4.1 | Bicubic baseline 计算 | 论文消融对比 | ⏳ |
| B4.2 | 无预训练从头训练 baseline | 论文消融对比 | ⏳ |
| B4.3 | ROI分析 (颅骨/脑实质/脑脊液) | PSNR解释力 | ⏳ |
| B4.4 | 可视化结果对比 | 定性对比图 | ⏳ |

---

## 📁 文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| **Roadmap** | `scripts/autodl_skill/docs/roadmap/ROADMAP.md` | 项目路线图 |
| **训练笔记** | `autodl-tmp/rplhr-ct-training-main/training_notes.md` | 训练记录 |
| **100 Epoch报告** | `autodl-tmp/rplhr-ct-training-main/TRAINING_REPORT_100EPOCH.md` | 100ep训练报告 |
| **崩溃诊断** | `autodl-tmp/rplhr-ct-training-main/TRAINING_CRASH_DIAGNOSIS.md` | 训练崩溃分析 |
| **PSNR差异分析** | `autodl-tmp/rplhr-ct-training-main/PSNR_ANALYSIS_PRETRAIN_VS_FINETUNE.md` | 跨部位指标分析 |
| **AutoDL指南** | `AUTODL_TRAINING_GUIDE.md` | AutoDL使用指南 |
| **周志** | `周志/周志_0326.md`, `周志_0401.md`, `周志_0408.md` | 周报 |

---

## 🐛 已知问题

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| out_z/slice_sequence 硬编码 | ✅ | 移到 forward 动态计算 |
| Loss 计算域不一致 | ✅ | 统一裁剪逻辑 |
| 数据量不足 | ✅ | 32GiB 公开数据集预训练 |
| c_z=6 方案错误 | ✅ | c_z=4 不变，用 Ratio-Aware |
| 冻结策略过于保守 (7%) | 🔴 | 修正为 Freeze-B (79%) 或 Freeze-C (98%) |
| Finetune SSIM 低于从头训练 | 🔴 | 延长训练 + EAGLELoss3D + EMA |

---

*最后更新: 2026-04-08*
*策略版本: v4.0 - Phase A完成，进入Phase B Finetune (冻结策略+训练优化+backbone扩展)*
*下次更新: Freeze-A/B/C 对比实验完成后*
