# 快速参考卡

## 常用命令

```bash
# 状态检查
./check_status.sh              # 查看整体状态
./quick_check.sh               # 快速检查
./check_training.sh            # 检查训练进程
./check_metrics.sh             # 查看最新指标

# 训练管理
./run_training.sh              # 启动训练
./monitor_training.sh          # 监控训练
./monitor_daemon.sh start      # 启动守护进程
./monitor_daemon.sh stop       # 停止守护进程

# 实验管理
./create_experiment.sh EXP_xxx "描述" A0    # 创建实验
./collect_results.sh           # 收集结果
./compare_experiments.sh       # 对比实验

# 自动化
./autonomous_loop.sh           # 启动自动化循环
./optimization_tracker.sh      # 追踪优化进度
```

## 配置参数

### Loss 选择
```ini
loss_type = 'l1'           # L1Loss
loss_type = 'eagle3d'      # EAGLELoss3D
loss_type = 'charbonnier'  # CharbonnierLoss
```

### 关键超参数
```ini
# 优化器
optim = 'AdamW'
lr = 0.0002
wd = 0.0001

# 学习率调度
cos_lr = True
Tmax = 100

# EMA
use_ema = True
ema_decay = 0.999

# 梯度裁剪
use_grad_clip = True
grad_clip_norm = 1.0
```

## 指标参考

| 指标 | 基线 | 目标 |
|------|------|------|
| PSNR | 20.01 dB | 27 dB |
| SSIM | 0.847 | > 0.9 |

## 决策规则

| 提升 | 决策 |
|------|------|
| ≥ 0.5 dB | ✅ 采纳 |
| 0.2-0.5 dB | ⚠️ 保留 |
| < 0.2 dB | ❌ 放弃 |
