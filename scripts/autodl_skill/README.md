# RPLHR-CT AutoDL 工具脚本

本目录包含用于管理 AutoDL 云端训练的工具脚本。

---

## 📁 AutoDL 主机目录架构

### 根目录结构

```
/root/
├── autodl-tmp/                    # 数据和代码目录
│   └── rplhr-ct-training-main/
│       ├── code/                  # 训练代码
│       │   ├── trainxuanwu.py     # 主训练脚本
│       │   ├── model_TransSR.py   # 模型架构
│       │   ├── loss*.py           # Loss函数
│       │   ├── make_dataset*.py   # 数据集加载
│       │   ├── net/               # 网络模块
│       │   ├── utils/             # 工具函数
│       │   └── config.py          # 配置
│       ├── config/                 # 配置文件
│       │   ├── xuanwu_ratio4.txt  # 宣武1:4配置
│       │   └── ...
│       ├── data/                  # 训练数据
│       │   └── dataset01_xuanwu/  # 宣武数据集
│       │       ├── train/          # 训练集
│       │       └── val/           # 验证集
│       ├── model/                 # 保存的模型
│       │   └── dataset01_xuanwu/  # 按数据集分
│       │       └── xuanwu_ema_fixed/  # EMA实验模型
│       ├── checkpoints/            # 检查点
│       ├── train_*.log           # 训练日志
│       └── *.log                 # 各种日志文件
├── miniconda3/                   # Conda环境
│   └── bin/activate              # 环境激活脚本
└── ... (系统文件)
```

### 关键数据位置

| 数据类型 | 路径 |
|----------|------|
| **训练代码** | `/root/autodl-tmp/rplhr-ct-training-main/code/` |
| **训练数据** | `/root/autodl-tmp/rplhr-ct-training-main/data/dataset01_xuanwu/` |
| **训练日志** | `/root/autodl-tmp/rplhr-ct-training-main/train_*.log` |
| **验证结果** | `/root/autodl-tmp/rplhr-ct-training-main/model/dataset01_xuanwu/` |
| **检查点** | `/root/autodl-tmp/rplhr-ct-training-main/checkpoints/` |

### 配置文件

| 配置 | 路径 |
|------|------|
| **宣武1:4** | `/root/autodl-tmp/rplhr-ct-training-main/config/xuanwu_ratio4.txt` |

### 快速访问命令

```bash
# 连接AutoDL
ssh -p 23086 root@connect.westd.seetacloud.com

# 激活conda环境
source /root/miniconda3/bin/activate base

# 进入代码目录
cd /root/autodl-tmp/rplhr-ct-training-main/code

# 查看训练日志
tail -100 /root/autodl-tmp/rplhr-ct-training-main/train_ema_fixed_50epoch.log

# 查看最新模型
ls -lt /root/autodl-tmp/rplhr-ct-training-main/model/dataset01_xuanwu/
```

---

## 📁 文件结构

```
scripts/autodl_skill/
├── lib_ssh.sh              # SSH连接工具库
├── lib_monitor.sh          # 训练监控工具库
├── lib_viz.sh              # 可视化工具库
├── config.sh               # 配置文件（含敏感信息）
├── autonomous_loop.sh      # 自主训练循环（增强版）
├── collect_results.sh      # 结果收集工具（增强版）
├── check_training.sh       # 训练状态检查（增强版）
├── compare_experiments.sh  # 实验对比报告（增强版）
├── monitor_daemon.sh       # 监控守护进程（增强版）
├── monitor_training.sh     # 实时监控脚本
├── check_metrics.sh        # Metrics趋势分析
├── download_logs.sh        # 日志下载工具
├── create_experiment.sh    # 创建新实验（支持模块化配置）
├── run_training.sh         # 单次训练启动（支持模块化配置）
├── quick_experiment.sh     # 快速启动实验（模块化配置）
├── list_configs.sh         # 列出可用配置模板
├── quick_check.sh          # 快速状态检查
├── optimization_tracker.sh # 优化追踪
└── fix_json_serialize.sh   # JSON修复工具
```

## 🧩 模块化配置系统

新的模块化配置系统将训练配置分解为三个独立模块，便于灵活组合和快速实验：

- **Loss 模块**: 损失函数配置 (L1, EAGLE3D, Charbonnier, L1+SSIM, 多尺度)
- **Augmentation 模块**: 数据增强配置 (无增强, 翻转, 噪声, 组合)
- **Training 模块**: 训练参数配置 (优化器, 学习率, EMA, Warmup)

### 快速使用模块化配置

```bash
# 1. 查看可用配置
./list_configs.sh

# 2. 快速启动实验 (格式: <exp_name> <loss> <augment> <training> [epochs])
./quick_experiment.sh exp01 l1 flip adamw_ema 50
./quick_experiment.sh exp02 eagle3d combined advanced 100

# 3. 使用模块配置创建实验
./create_experiment.sh \
    --loss eagle3d \
    --augment flip \
    --training adamw_ema \
    --name "my_experiment"

# 4. 使用配置文件运行训练
./run_training.sh --config ../config/experiments/exp01.json 50
./run_training.sh --loss eagle3d --training advanced 100

# 5. 对比实验配置差异
./compare_experiments.sh --compare-modules --exp1 exp01 --exp2 exp02
```

详细文档：[MODULAR_SYSTEM_GUIDE.md](../../MODULAR_SYSTEM_GUIDE.md)

## 🚀 快速开始

### 1. 配置SSH连接

确保SSH免密登录已配置：
```bash
# 复制公钥到AutoDL服务器
ssh-copy-id -p <port> root@<host>

# 测试连接
./collect_results.sh --test
```

### 2. 检查训练状态

```bash
# 单次检查
./check_training.sh

# 持续监控模式（每60秒刷新）
./check_training.sh --watch
./check_training.sh --watch --interval 30
```

### 3. 收集训练结果

```bash
./collect_results.sh
```

结果将保存到 `logs/run_YYYYMMDD_HHMMSS/` 目录。

### 4. 启动监控守护进程

```bash
# 启动（默认8小时）
./monitor_daemon.sh start

# 指定持续时间
./monitor_daemon.sh start 12

# 查看状态
./monitor_daemon.sh status

# 查看日志
./monitor_daemon.sh log
./monitor_daemon.sh log 50

# 停止监控
./monitor_daemon.sh stop

# 测试告警
./monitor_daemon.sh alert-test
```

### 5. 运行自主训练循环

```bash
# 后台运行
./autonomous_loop.sh

# 查看日志
tail -f logs/autonomous/autonomous_*.log

# 停止
kill <PID>
```

### 6. 对比实验结果

```bash
# 快速概览
./compare_experiments.sh

# 生成完整报告
./compare_experiments.sh --report
./compare_experiments.sh --report --output my_report.md
```

## 📊 主要功能

### 改进的SSH连接稳定性

- **连接重试机制**：自动重试失败的SSH连接（最多5次，指数退避）
- **错误分类处理**：区分连接拒绝、超时、权限拒绝等不同错误
- **连接健康检查**：提供 `ssh_test` 和 `ssh_check_connection` 函数

### 完善的结果收集

- **自动日志收集**：自动下载所有训练日志文件
- **Metrics解析**：从CSV和日志中提取PSNR、SSIM、Loss指标
- **可视化图表**：生成ASCII趋势图和HTML报告
- **JSON报告**：结构化数据便于后续分析

### 增强的训练监控

- **实时监控**：每3分钟检查一次训练状态
- **异常检测**：
  - Loss爆炸检测（阈值：1000）
  - NaN/Inf检测
  - GPU温度监控（>85°C告警）
  - GPU内存监控（>95%告警）
  - 训练卡住检测（10分钟无变化）
- **自动告警**：去重机制避免重复告警
- **状态持久化**：保存监控状态便于恢复

### 改进的实验对比

- **自动数据收集**：从EXPERIMENTS.md和本地日志收集数据
- **性能排名**：Top 3最佳实验
- **趋势分析**：平均PSNR、性能差距计算
- **可视化对比**：ASCII柱状图

## 🔧 配置文件

编辑 `config.sh` 配置连接信息：

```bash
# AutoDL SSH 连接信息
export AUTODL_HOST="connect.westd.seetacloud.com"
export AUTODL_PORT="23086"
export AUTODL_USER="root"
export AUTODL_PASS="your_password"

# 路径配置
export AUTODL_REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"
export DATASET_KEY="dataset01_xuanwu"
export NET_IDX="xuanwu_50epoch"
```

## 📝 环境变量

可以通过环境变量覆盖配置：

```bash
# 修改服务器地址
AUTODL_HOST=new.host.com AUTODL_PORT=12345 ./check_training.sh

# 修改训练参数
EPOCHS=50 NET_IDX=my_experiment ./autonomous_loop.sh
```

## 🐛 故障排查

### SSH连接失败

```bash
# 测试连接
./collect_results.sh --test

# 检查SSH密钥
ssh -p <port> root@<host> echo "OK"

# 使用密码认证（如果密钥失败）
# 确保安装了expect: brew install expect
```

### 训练进程检测失败

```bash
# 手动检查
ssh -p <port> root@<host> "ps aux | grep trainxuanwu"

# 检查日志路径是否正确
grep REPO_PATH config.sh
```

### 监控告警不工作

```bash
# 测试告警系统
./monitor_daemon.sh alert-test

# 检查日志
cat logs/monitor/monitor_$(date +%Y%m%d).log
```

## 📈 日志文件

- **监控日志**: `logs/monitor/monitor_YYYYMMDD.log`
- **自主训练日志**: `logs/autonomous/autonomous_YYYYMMDD_HHMMSS.log`
- **运行结果**: `logs/run_YYYYMMDD_HHMMSS/`

## 🔒 安全提示

- `config.sh` 已加入 `.gitignore`，不会提交到GitHub
- 建议使用SSH密钥认证，避免密码存储
- 不要在日志中记录敏感信息

## 🛠️ 开发计划

- [ ] 添加Slack/钉钉通知支持
- [ ] Web监控界面
- [ ] 自动超参数调优建议
- [ ] 训练结果数据库

## 📞 支持

如有问题，请检查：
1. SSH连接是否正常
2. 远程路径配置是否正确
3. 训练脚本名称是否匹配
