# autodl_skill 文档中心

> RPLHR-CT 项目 AutoDL 训练优化技能文档

---

## 文档目录结构

```
docs/
├── README.md                    # 本文档 - 入口指南
├── architecture/                # 架构设计文档
│   └── MODULAR_DESIGN_SPEC.md   # 模块化系统设计规范
├── research/                    # 技术调研文档
│   ├── 3D_EDGE_AWARE_LOSS_RESEARCH.md
│   ├── RATIO4_ARCHITECTURE_ADAPTATION.md
│   └── RATIO4_MASTER_REVIEW.md
├── roadmap/                     # 路线图与规划
│   └── ROADMAP.md               # 项目优化路线图 v2.0
├── experiments/                 # 实验记录与分析
│   ├── EXPERIMENTS.md           # 系统化实验记录
│   ├── TRAINING_LOG.md          # 训练日志
│   └── EXP_001_ANALYSIS_REPORT.md
├── guides/                      # 操作指南与规范
│   ├── DEVELOPMENT_SOP.md       # 开发标准作业程序
│   ├── SESSION_SUMMARY.md       # 会话总结
│   └── SOP_COMPLIANCE_AUDIT.md  # SOP 合规审计
└── scripts/                     # 脚本使用文档 (待创建)
    └── README.md
```

---

## 快速导航

### 🎯 如果你是第一次使用
1. 阅读 [`guides/DEVELOPMENT_SOP.md`](guides/DEVELOPMENT_SOP.md) - 了解开发规范
2. 查看 [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md) - 了解当前优化阶段
3. 阅读本文档下方的【快速开始】

### 🔬 如果你想了解技术调研
- [3D Edge Aware Loss 调研](research/3D_EDGE_AWARE_LOSS_RESEARCH.md)
- [Ratio4 架构适配分析](research/RATIO4_ARCHITECTURE_ADAPTATION.md)
- [Master 代码审查](research/RATIO4_MASTER_REVIEW.md)

### 📊 如果你想查看实验结果
- [系统化实验记录](experiments/EXPERIMENTS.md)
- [训练日志详情](experiments/TRAINING_LOG.md)
- [EXP_001 分析报告](experiments/EXP_001_ANALYSIS_REPORT.md)

### 🏗️ 如果你想了解系统架构
- [模块化系统设计规范](architecture/MODULAR_DESIGN_SPEC.md) ⭐ **新**

---

## 快速开始

### 1. 查看当前训练状态
```bash
./check_status.sh
```

### 2. 启动新实验
```bash
# 方法1: 使用实验创建脚本
./create_experiment.sh EXP_002_eagle3d "EAGLELoss3D实验" A0

# 方法2: 使用模块化配置 (即将支持)
# ./quick_experiment.sh EXP_002 loss/eagle3d augment/none training/baseline
```

### 3. 监控训练
```bash
./monitor_training.sh
# 或启动守护进程
./monitor_daemon.sh start
```

### 4. 收集结果
```bash
./collect_results.sh
```

### 5. 对比实验
```bash
./compare_experiments.sh
```

---

## 当前状态

### 最新进展 (2026-04-01)

| 实验ID | 优化策略 | PSNR | SSIM | 决策 |
|--------|----------|------|------|------|
| A0 (基线) | L1Loss | 20.01 dB | 0.847 | ✅ 基线 |
| EXP_002 | EAGLELoss3D | 20.11 dB | 0.873 | ✅ +0.10 dB |
| EXP_003 | CharbonnierLoss | 20.11 dB | 0.861 | ⚠️ 提升有限 |
| EXP_004 | Flip增强 | 20.08 dB | 0.857 | ⚠️ 提升有限 |

**当前最佳**: PSNR = 20.11 dB (EAGLELoss3D, Epoch 21)

### 目标差距
```
当前:  20.11 dB
目标:  27.00 dB
差距:   6.89 dB
```

---

## 模块化系统 (开发中)

我们正在开发模块化配置系统，支持：

- ✅ **Loss 模块**: L1, EAGLE3D, Charbonnier, SSIM, Combined
- ✅ **数据增强模块**: Flip, Noise, Elastic
- ✅ **训练策略模块**: AdamW, EMA, Gradient Clipping
- ⏳ **配置系统**: 统一配置管理
- ⏳ **工具集成**: 自动化实验管理

详细设计请参考 [MODULAR_DESIGN_SPEC.md](architecture/MODULAR_DESIGN_SPEC.md)

---

## 脚本清单

| 脚本 | 用途 | 状态 |
|------|------|------|
| `autonomous_loop.sh` | 自动化训练循环 | ✅ |
| `create_experiment.sh` | 创建实验 | ✅ |
| `run_training.sh` | 运行训练 | ✅ |
| `monitor_training.sh` | 监控训练 | ✅ |
| `monitor_daemon.sh` | 守护进程 | ✅ |
| `check_status.sh` | 检查状态 | ✅ |
| `check_training.sh` | 检查训练 | ✅ |
| `check_metrics.sh` | 检查指标 | ✅ |
| `collect_results.sh` | 收集结果 | ✅ |
| `compare_experiments.sh` | 对比实验 | ✅ |
| `download_logs.sh` | 下载日志 | ✅ |
| `optimization_tracker.sh` | 优化追踪 | ✅ |
| `fix_json_serialize.sh` | JSON修复 | ✅ |
| `quick_check.sh` | 快速检查 | ✅ |
| `training_daemon.sh` | 训练守护 | ✅ |

---

## 相关链接

- [项目根目录 README](../README.md)
- [AutoDL 训练指南](../../AUTODL_TRAINING_GUIDE.md)
- [数据流程文档](../../DATA_FLOW_VISUAL.md)

---

*最后更新: 2026-04-03*
