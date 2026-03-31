# AutoDL MLOps Skill - RPLHR-CT Training

## 概述

本 Skill 实现 **本地开发 → GitHub → AutoDL 训练 → 结果回收** 的自动化闭环。

```
┌──────────────────────────────────────────────────────────────────┐
│                        本地开发环境                                │
│  code/                    │  git commit + push                   │
│  ├── trainxuanwu.py      └──────────────────────────────────→    │
│  ├── model_TransSR.py              GitHub                        │
│  └── ...                          (jinyanghe1/rplhr-ct-training)│
└──────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ SSH + git clone/pull
┌────────────────────────────────────┴──────────────────────────────┐
│                      AutoDL 实例 (GPU)                            │
│  1. source /etc/network_turbo     ← 开启 GitHub 加速            │
│  2. git clone/pull                ← 同步最新代码                 │
│  3. python trainxuanwu.py         ← 使用宣武 1:4 数据集训练      │
│  4. 评估 PSNR/SSIM/MSE           ← Benchmark 结果               │
│  5. SOTA 模型存档                 ← 保存在 /root/model/          │
│  6. 输出训练日志                  ← 下载到本地 logs/             │
└──────────────────────────────────────────────────────────────────┘
```

## 数据集信息

- **宣武数据集**: `/root/autodl-tmp/rplhr-ct-training-main/data/dataset01_xuanwu/`
- **比例**: 1mm : 4mm (厚层 : 薄层)
- **训练样本**: 39
- **验证样本**: 5

## 快速开始

### 1. 手动运行一次训练

```bash
./scripts/autodl_skill/run_training.sh [epochs]
# 示例: ./scripts/autodl_skill/run_training.sh 50
```

### 2. 查看训练状态

```bash
./scripts/autodl_skill/check_status.sh
```

### 3. 收集结果

```bash
./scripts/autodl_skill/collect_results.sh
```

### 4. 配置定时任务

```bash
./scripts/autodl_skill/setup_crontab.sh
```

## 目录结构

```
scripts/autodl_skill/
├── README.md              # 本文档
├── run_training.sh         # 主训练脚本 (git sync + 启动训练)
├── check_status.sh         # 检查训练状态
├── collect_results.sh      # 收集训练结果到本地
├── download_logs.sh        # 下载训练日志
├── setup_crontab.sh        # 配置 crontab
├── config.sh               # 配置文件 (凭证等)
├── logs/                   # 本地日志存放
│   ├── run_20260331_203000/
│   │   ├── training.log
│   │   ├── metrics.json
│   │   └── insights.md
│   └── ...
└── .gitkeep
```

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `--path_key` | `dataset01_xuanwu` | 宣武数据集 |
| `--net_idx` | `xuanwu_50epoch` | 网络配置 |
| `--epoch` | 用户指定 | 训练轮数 |
| `--use_augmentation` | `True` | 启用数据增强 |
| `--aug_prob` | `0.5` | 增强概率 |
| `--clip_ct` | `True` | CT 裁剪 |
| `--min_hu` | `-1024` | HU 最小值 |
| `--max_hu` | `3071` | HU 最大值 |

## 指标定义

训练完成后评估以下指标:

- **PSNR**: 峰值信噪比 (越高越好)
- **SSIM**: 结构相似性 (越高越好)
- **MSE**: 均方误差 (越低越好)

如果当前指标优于历史 SOTA，自动存档模型到 `/root/model/sota/`。

## Crontab 配置

默认每小时运行一次，检测代码是否有更新:

```bash
# 每小时第 30 分钟运行
30 * * * * cd /path/to/RPLHR-CT-main && ./scripts/autodl_skill/run_training.sh 10
```

## 安全注意

- SSH 凭证存储在 `config.sh` (已加入 .gitignore)
- 不要将 `config.sh` 提交到 GitHub
- 定期轮换密码

## 故障排查

### SSH 连接失败
```bash
# 检查实例是否开机
ssh -p 23086 root@connect.westd.seetacloud.com

# 检查网络加速
source /etc/network_turbo
```

### 训练失败
```bash
# 查看详细日志
tail -100 /root/autodl-tmp/rplhr-ct-training-main/train_autodl.log

# 检查 GPU 状态
nvidia-smi
```

### Git 同步失败
```bash
# 手动同步
ssh -p 23086 root@connect.westd.seetacloud.com
source /etc/network_turbo && cd /root && git pull origin main
```
