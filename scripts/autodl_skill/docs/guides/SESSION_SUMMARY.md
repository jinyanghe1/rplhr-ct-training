# RPLHR-CT Session Summary - 2026-04-01

## 当前最佳结果

| 实验 | 方法 | Best PSNR | Best SSIM | 备注 |
|------|------|-----------|-----------|------|
| **基线** | L1Loss | 20.01 | 0.847 | Epoch 16 |
| EXP_002 | EAGLELoss3D | **20.11** | **0.873** | +0.10 dB, SSIM提升显著 |
| EXP_003 | CharbonnierLoss | 20.11 | 0.861 | +0.10 dB |
| EXP_004 | Flip增强 | 20.08 | 0.857 | +0.07 dB |

**当前最佳**: PSNR=20.11, SSIM=0.873 (EAGLELoss3D)
**目标**: PSNR > 27 dB (差距 7 dB)

---

## 关键发现

1. **所有优化方法提升都 ~0.1 dB** - 当前 TVSRN 架构可能已接近极限
2. **EAGLELoss3D 的 SSIM 提升最显著** (+0.026) - 适合需要高边缘质量的场景
3. **训练在 Epoch 21 达峰后自然进入平台期** - 不是 bug，是正常行为
4. **距目标 PSNR > 27 dB 还有 7 dB 差距** - 需要更大的架构改动 (如 SwinIR)

---

## 已验证的优化 (效果有限)

| 类别 | 方法 | 结果 | 提升 |
|------|------|------|------|
| Loss | EAGLELoss3D | ✅ 有效 | +0.10 dB |
| Loss | Charbonnier | ✅ 有效 | +0.10 dB |
| Loss | L1 + SSIM | ⏸️ 未测试 | - |
| 增强 | Flip | ✅ 有效 | +0.07 dB |
| 增强 | 随机噪声 | ⏸️ 未测试 | - |
| 增强 | 3D弹性形变 | ⏸️ 未测试 | - |

---

## 下一步TODO (按优先级)

### 高优先级 - 尚未测试
1. **EMA (指数移动平均)** - 预期 +0.1-0.3 dB, 减少抖动
2. **Gradient Clipping** - 稳定训练
3. **AdamW** - 可能更稳定

### 中优先级 - 架构升级
4. **SwinIR 3D 适配** - 预期 +3-5 dB (需要较大改动)
5. **增加 TE_c** (16→32) - 预期 +0.5-1 dB

### 待解锁优化
- 数据增强: 随机噪声, 3D弹性形变 (预期 +0.3-1.5 dB)
- 训练策略: Warmup, Tmax调整, 更长训练

---

## 代码位置

| 文件 | 本地 | 远程 |
|------|------|------|
| 主训练脚本 | `code/trainxuanwu.py` | `/root/autodl-tmp/rplhr-ct-training-main/code/` |
| Loss函数 | `code/loss_eagle3d.py` | `/root/autodl-tmp/rplhr-ct-training-main/code/loss.py` |
| 模型架构 | `code/net/model_TransSR.py` | `/root/autodl-tmp/rplhr-ct-training-main/code/net/` |
| 数据集 | - | `/root/autodl-tmp/rplhr-ct-training-main/data/dataset01_xuanwu/` |

---

## AutoDL 连接

```bash
Host: connect.westd.seetacloud.com
Port: 23086
User: root
远程conda: /root/miniconda3/bin/python
```

---

## 快速恢复命令

```bash
# 1. SSH 到 AutoDL
ssh -p 23086 root@connect.westd.seetacloud.com

# 2. 激活环境并进入代码目录
source /root/miniconda3/bin/activate base
cd /root/autodl-tmp/rplhr-ct-training-main/code

# 3. 查看训练状态
ps aux | grep trainxuanwu
tail -50 /root/autodl-tmp/rplhr-ct-training-main/train_*.log

# 4. 拉取最新代码
cd /root/autodl-tmp/rplhr-ct-training-main
git pull origin main

# 5. 开始新训练
cd code
python trainxuanwu.py train --net_idx=xuanwu_aug --path_key=dataset01_xuanwu --config=../config/xuanwu_ratio4.txt --epoch=25 --use_augmentation=True --aug_prob=0.5 --normalize_ct=True --clip_ct=True --min_hu=-1024 --max_hu=3071 --use_tta=True --num_workers=4 --test_num_workers=2
```

---

## 训练配置

```ini
ratio = 4
c_z = 6
TE_c = 16
TD_Tw = 1
cos_lr = True
Tmax = 50
gap_epoch = 50
use_tta = True
```

---

## 文档索引

| 文档 | 用途 |
|------|------|
| `TRAINING_LOG.md` | 详细实验记录 |
| `ROADMAP.md` | 优化路线图和TODO |
| `3D_EDGE_AWARE_LOSS_RESEARCH.md` | Loss函数调研 |
| `SOP_COMPLIANCE_AUDIT.md` | 开发规范审计 |
| `SESSION_SUMMARY.md` | 本文档 - 快速恢复 |

---

*最后更新: 2026-04-01*
