# 训练进程意外终止 - 诊断与修复报告

> **生成时间**: 2026-04-08
> **硬件**: RTX Pro 6000 (96GB VRAM), AutoDL云实例
> **现象**: 训练进程在第15 epoch左右意外终止

---

## 1. 问题诊断

### 1.1 守护进程检查结果

经检查所有训练启动脚本，发现以下关键问题：

| 脚本文件 | 守护方式 | 问题 |
|----------|----------|------|
| `start_train_xuanwu.sh` | ❌ **无守护** | 直接运行 `python trainxuanwu.py` |
| `start_train_xuanwu_fix1_10epoch.sh` | ⚠️ **半守护** | 使用 `tee` 但无 nohup，SSH断开仍会终止 |
| `start_train_xuanwu_background.sh` | ⚠️ **nohup** | 使用了 nohup 但 PID 写入 /tmp（重启丢失） |
| `start_training.sh` (AUTODL_TRAINING_GUIDE) | ⚠️ **无守护** | 直接运行 python |

**核心问题**: 大部分训练脚本没有使用可靠的守护进程机制！

### 1.2 最可能的终止原因（按概率排序）

#### 🔴 原因1: SSH连接断开 (概率 70%)

**证据**:
- `start_train_xuanwu.sh` 和 `start_train_xuanwu_fix1_10epoch.sh` 都在**前台**运行
- 没有使用 `tmux`/`screen`/`systemd`
- 即使 `start_train_xuanwu_background.sh` 使用了 nohup，但只把PID写入 /tmp（重启清空）
- AutoDL实例的SSH连接容易因网络波动断开

**机制**: SSH断开 → Shell收到SIGHUP → 前台进程组所有进程被终止

#### 🟡 原因2: 系统内存OOM Killer (概率 20%)

**证据**:
- 默认 `num_workers=4`，每个worker会复制数据集到内存
- RPLHR-CT数据是3D体数据，单个样本可能数百MB
- 4个worker × 数百MB = 可能消耗数GB系统RAM

**检查方法**: `dmesg -T | grep -i "killed process"`

#### 🟢 原因3: 磁盘空间不足 (概率 5%)

**证据**:
- checkpoint保存为完整模型（包含网络权重+配置+历史），可能很大
- AutoDL系统盘默认50GB，数据+代码+checkpoint可能占满

**检查方法**: `df -h`

#### 🟢 原因4: 数据异常/代码Bug (概率 5%)

**证据**:
- 第15个epoch恰好遇到某异常数据
- 但训练loss一直在下降，不太像数据问题

---

## 2. 修复方案

### 2.1 推荐方案: systemd 用户级服务

这是最可靠的方案，进程与SSH会话完全解耦，即使SSH断开、终端关闭也不影响训练。

#### 服务配置文件

```ini
# ~/.config/systemd/user/train-rplhr.service
[Unit]
Description=RPLHR-CT Super-Resolution Training
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/autodl-tmp/rplhr-ct-training-main/code
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/root/miniconda3/bin/python trainxuanwu.py train \
    --net_idx="xuanwu_50epoch" \
    --path_key="dataset01_xuanwu" \
    --epoch=200 \
    --use_augmentation=True \
    --normalize_ct=True \
    --window_center=40 \
    --window_width=400 \
    --num_workers=2 \
    --test_num_workers=2
Restart=on-failure
RestartSec=30
StandardOutput=append:/root/autodl-tmp/rplhr-ct-training-main/train.log
StandardError=append:/root/autodl-tmp/rplhr-ct-training-main/train.log

[Install]
WantedBy=default.target
```

#### 启动命令

```bash
# 创建服务目录
mkdir -p ~/.config/systemd/user/

# 复制服务文件后
systemctl --user daemon-reload
systemctl --user start train-rplhr
systemctl --user status train-rplhr    # 查看状态
journalctl --user -u train-rplhr -f    # 实时日志

# 开机自启（可选）
systemctl --user enable train-rplhr

# 停止训练
systemctl --user stop train-rplhr
```

**关键优势**:
- `Restart=on-failure`: 如果进程因非正常原因终止，30秒后自动重启
- 完全与SSH会话解耦
- 日志持久化到文件
- 可随时查看状态

### 2.2 备选方案: tmux + 训练脚本

```bash
# 创建新tmux会话
tmux new -s train

# 在tmux中运行训练
cd /root/autodl-tmp/rplhr-ct-training-main/code
python trainxuanwu.py train \
    --net_idx="xuanwu_50epoch" \
    --path_key="dataset01_xuanwu" \
    --epoch=200 \
    --use_augmentation=True \
    --normalize_ct=True \
    --window_center=40 \
    --window_width=400 \
    --num_workers=2 \
    --test_num_workers=2 \
    2>&1 | tee /root/autodl-tmp/train.log

# 分离: Ctrl+B, D
# 重新连接: tmux attach -t train
```

### 2.3 其他优化建议

#### 减少内存占用
```python
# 在训练脚本中降低 num_workers
num_workers = 2       # 原来是4，降到2
test_num_workers = 2  # 原来是4，降到2
```

#### 增加checkpoint保存频率
当前代码只在PSNR创新高时保存best_model，每10个epoch保存checkpoint。建议：
- 每个epoch都保存最新模型（覆盖式）
- 每10个epoch保存归档checkpoint
- best_model单独保存不覆盖

#### 添加进程存活标记
```python
# 在训练循环中添加心跳标记
import os
heartbeat_file = '../train_log/heartbeat.txt'
with open(heartbeat_file, 'w') as f:
    f.write(f"Epoch {tmp_epoch} running at {datetime.now()}\n")
```

---

## 3. 训练权重保存状态

### 已有权重文件

| 路径 | 描述 | 状态 |
|------|------|------|
| `checkpoints/SRM/TVSRN_TINY_E20/` | Tiny数据集20 epoch训练 | ✅ 已保存 |
| `model/SRM/TVSRN/best_model.pkl` | 公开数据集100 epoch最优 | ✅ 已保存（在AutoDL上） |
| 宣武数据集训练权重 | 30 epoch训练 | ⚠️ 需确认是否在AutoDL上 |

### 权重保护建议

1. **训练完成后立即下载**: `scp -P 端口 root@地址:/root/autodl-tmp/rplhr-ct-training-main/model/ ./
2. **上传到阿里云盘**: `aliyunpan upload model_results.tar.gz /RPLHR-CT-Results/`
3. **本地备份**: 复制到 `checkpoints/` 目录

---

## 4. 重启训练前检查清单

- [ ] 检查AutoDL实例是否仍在运行
- [ ] `df -h` 检查磁盘空间
- [ ] `free -h` 检查系统内存
- [ ] `dmesg -T | grep -i "killed process"` 检查OOM记录
- [ ] 确认 `model/` 目录下的已有权重
- [ ] 配置 systemd 用户服务或 tmux
- [ ] 降低 `num_workers` 到2
- [ ] 启动训练

---

*最后更新: 2026-04-08*
