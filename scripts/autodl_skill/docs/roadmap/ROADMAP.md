# RPLHR-CT 项目 Roadmap v3.0

> **维护时间**: 2026-04-08
> **策略更新**: 从Phase 2 Backbone微调转向Ratio-Aware架构重构
> **目标**: PSNR > 30 dB (预训练 + finetune)
> **当前基线**: PSNR=20.11 dB (EAGLELoss3D, Epoch 21)
> **当前任务**: Ratio-Aware CT 超分模型改进

---

## ⚠️ 重要更新 (2026-04-08)

**架构重构 - Ratio-Aware 方案**:
- 原模型 `out_z`, `slice_sequence`, `positional_encoding` 在 `__init__` 中硬编码为固定 ratio
- 现改为 `forward` 中根据传入 `ratio` 动态计算，支持任意超分倍率
- Loss 计算域统一：GT 和输出必须在同一物理分辨率
- 训练策略：先公开数据集预训练 → 再自用数据集 finetune

---

## 🎯 Ratio-Aware 架构改进方案

### 核心改动

```
┌─────────────────────────────────────────────────────────────┐
│  Ratio-Aware TVSRN 改动清单                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. model_TransSR.py                                        │
│     ├── MToken: out_z 从 __init__ 移到 forward 动态计算      │
│     │   out_z = (c_z - 1) * ratio + 1                      │
│     ├── slice_sequence: 从 __init__ 移到 forward 动态生成     │
│     │   根据 ratio 和 c_z 实时构建交错序列                    │
│     ├── positional encoding: 从 __init__ 移到 forward        │
│     │   根据 out_z 动态生成 PE                                │
│     ├── x_patch_mask: 从 __init__ 移到 forward               │
│     │   根据 out_z-c_z 动态生成 mask tokens                  │
│     ├── forward(self, x, ratio): 新增 ratio 参数              │
│     └── Decoder img_size 根据 out_z 动态调整                  │
│                                                             │
│  2. train.py                                                │
│     └── net(x) → net(x, ratio=opt.ratio)                   │
│                                                             │
│  3. val.py                                                  │
│     └── net(tmp_x) → net(tmp_x, ratio=4)  # 验证固定 ratio=4 │
│                                                             │
│  4. Loss 计算域统一                                          │
│     ├── 确保 GT 和输出处于同一物理分辨率                       │
│     └── 对输出进行裁剪使其与 GT 尺寸一致                      │
│                                                             │
│  5. 训练策略                                                │
│     ├── Phase A: 公开数据集预训练 (rplhr_ct_32G, ratio=5)    │
│     └── Phase B: 自用数据集 finetune (xuanwu, ratio=4)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Ratio-Aware 动态计算细节

#### out_z 动态计算
```python
# 旧: __init__ 中固定
self.out_z = (opt.c_z - 1) * opt.ratio + 1

# 新: forward 中动态
def forward(self, x, ratio=None):
    ratio = ratio if ratio is not None else opt.ratio
    out_z = (opt.c_z - 1) * ratio + 1
```

#### slice_sequence 动态生成
```python
# 旧: __init__ 中固定
self.register_buffer('slice_sequence', ...)

# 新: forward 中动态
def _build_slice_sequence(self, c_z, out_z, ratio):
    slice_list = list(range(out_z))
    vis_list = slice_list[:c_z]
    mask_list = slice_list[c_z:]
    re_list = []
    while len(vis_list) != 0:
        if len(re_list) % ratio == 0:
            re_list.append(vis_list.pop(0))
        else:
            re_list.append(mask_list.pop(0))
    return torch.tensor(re_list, dtype=torch.long, device=self.x_patch_mask.device)
```

#### Positional Encoding 动态生成
```python
# 旧: __init__ 中固定
self.register_buffer('positions_z', ...)

# 新: forward 中动态
positions_z = positionalencoding1d(self.c, out_z, 1).unsqueeze(2).unsqueeze(2)
positions_z = positions_z.to(x.device)
```

#### x_patch_mask 动态生成
```python
# 旧: __init__ 中固定
self.x_patch_mask = torch.nn.Parameter(
    torch.zeros(self.out_z - opt.c_z, self.c, opt.c_y, opt.c_x))

# 新: forward 中动态 - 但仍保留为可学习参数
# 方案: 保留最大尺寸的参数，forward 时截取前 N 个
max_out_z = (opt.c_z - 1) * max_ratio + 1  # max_ratio=5
self.x_patch_mask = nn.Parameter(torch.zeros(max_out_z - opt.c_z, self.c, opt.c_y, opt.c_x))
# forward 时:
num_masks = out_z - opt.c_z
x_patch_mask = self.x_patch_mask[:num_masks]
```

#### Loss 计算域统一
```python
# 确保输出和GT在同一物理分辨率
# 输出形状: [1, 1, out_z, c_y, c_x]
# GT形状: [1, 1, target_z, c_y, c_x]
# 需要裁剪输出使其与GT z维度一致

target_z = label.shape[2]
if y_pre.shape[2] > target_z:
    # 中心裁剪
    diff = y_pre.shape[2] - target_z
    start = diff // 2
    y_pre = y_pre[:, :, start:start+target_z]
elif y_pre.shape[2] < target_z:
    # 对GT进行裁剪 (不应发生，但做防御)
    diff = target_z - y_pre.shape[2]
    start = diff // 2
    label = label[:, :, start:start+y_pre.shape[2]]
```

---

## 📊 当前状态

### 差距分析
```
当前: 20.11 dB (Phase 1基线)
目标: 30 dB (Ratio-Aware预训练+finetune)

预期路径:
├── Ratio-Aware 架构改进: +5-8 dB (架构正确性修复)
├── 大数据集预训练 (32GiB): +3-5 dB (数据量提升)
├── Xuanwu finetune: +1-2 dB (领域适配)
└── 总预期: 29-35 dB
```

---

## 🚀 执行计划

### Phase A: 公开数据集预训练 (当前阶段)

| Step | 任务 | 目标 | 状态 |
|------|------|------|------|
| A1 | 更新 Roadmap | 文档化 Ratio-Aware 方案 | ✅ 完成 |
| A2 | 修改 model_TransSR.py | ratio-aware 动态计算 | ⏳ 待实施 |
| A3 | 修改 train.py | 传入 ratio 参数 | ⏳ 待实施 |
| A4 | 修改 val.py | 固定 ratio=4 | ⏳ 待实施 |
| A5 | 统一 Loss 计算域 | GT/输出同分辨率 | ⏳ 待实施 |
| A6 | 创建预训练配置 | ratio=5 配置文件 | ⏳ 待实施 |
| A7 | AutoDL 预训练 | PSNR > 30 dB | ⏳ 待实施 |

### Phase B: 自用数据集 Finetune (Phase A完成后)

| Step | 任务 | 目标 | 状态 |
|------|------|------|------|
| B1 | 创建 finetune 配置 | xuanwu + ratio=4 | ⏳ 待实施 |
| B2 | 加载预训练权重 | 从 Phase A 迁移 | ⏳ 待实施 |
| B3 | Xuanwu finetune | PSNR > 30 dB | ⏳ 待实施 |

---

## 📁 文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| **模块化设计规范** | `docs/architecture/MODULAR_DESIGN_SPEC.md` | 系统架构设计 |
| **EMA 问题排查** | `docs/guides/EMA_TROUBLESHOOTING.md` | EMA 故障修复指南 |
| **快速参考** | `docs/guides/QUICK_REFERENCE.md` | 常用命令速查 |
| **开发 SOP** | `docs/guides/DEVELOPMENT_SOP.md` | 开发规范 |

---

## 🐛 已知问题

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| out_z/slice_sequence 硬编码 | 🔴 → ✅ | 移到 forward 动态计算 |
| Loss 计算域不一致 | 🔴 → ✅ | 统一裁剪逻辑 |
| 数据量不足 | 🔴 → ✅ | 32GiB 公开数据集预训练 |

---

*最后更新: 2026-04-08*
*策略版本: v3.0 - Ratio-Aware 架构重构*
*下次更新: 预训练启动后*
