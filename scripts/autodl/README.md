# AutoDL MLOps 配置指南

## 环境变量设置

在使用 `local_controller.py` 之前，需要配置以下环境变量：

```bash
# AutoDL API Token (从控制台获取)
export AUTODL_TOKEN="your_api_token_here"

# AutoDL 实例 ID
export AUTODL_INSTANCE_ID="your_instance_id"

# SSH 连接信息
export AUTODL_SSH_HOST="region.autodl.com"
export AUTODL_SSH_PORT="22"
export AUTODL_SSH_USER="root"
export AUTODL_SSH_PASS="your_instance_password"
```

或者创建配置文件 `~/.autodl_config.json`：

```json
{
  "api_token": "your_api_token_here",
  "instance_id": "your_instance_id",
  "ssh_host": "region.autodl.com",
  "ssh_port": 22,
  "ssh_username": "root",
  "ssh_password": "your_instance_password"
}
```

## 获取 AutoDL API Token

1. 登录 AutoDL 控制台: https://www.autodl.com/console
2. 点击右上角头像 → 个人信息
3. 找到 "API 密钥" 部分，复制 Token

## 使用示例

### 1. 手动执行一次训练

```bash
cd /path/to/rplhr-ct-training

# 使用 1:4 比例配置训练
python scripts/autodl/local_controller.py config/ratio4.txt

# 或使用默认 1:5 配置
python scripts/autodl/local_controller.py config/default.txt
```

### 2. 定时自动训练

```bash
# 编辑 crontab
crontab -e

# 每 6 小时检查并训练一次
0 */6 * * * cd /path/to/rplhr-ct-training && python scripts/autodl/local_controller.py config/ratio4.txt >> autodl_cron.log 2>&1

# 或每天凌晨 2 点训练
0 2 * * * cd /path/to/rplhr-ct-training && python scripts/autodl/local_controller.py config/ratio4.txt
```

### 3. 查看结果

```bash
# 查看最新报告
cat autodl_reports/report_*.md | tail -100

# 查看历史指标
cat autodl_reports/metrics_history.json | jq '.best'

# 查看所有运行记录
ls -la autodl_reports/
```

## 文件结构

```
rplhr-ct-training/
├── scripts/autodl/
│   ├── autodl_client.py      # AutoDL API 客户端
│   ├── local_controller.py   # 本地控制器 (在此运行)
│   └── README.md             # 本文件
├── autodl_train.py           # 训练流水线 (AutoDL 上运行)
├── config/
│   ├── default.txt           # 1:5 比例配置
│   └── ratio4.txt            # 1:4 比例配置
├── archives/                 # 最佳模型存档
│   └── model_best_20260331_143022/
│       ├── model.pth
│       ├── metrics.json
│       └── config.txt
├── reports/                  # 训练报告 (AutoDL 上生成)
│   └── latest_report.md
└── autodl_reports/           # 下载到本地的报告
    ├── report_20260331_143022.md
    └── metrics_history.json
```

## 依赖安装

```bash
pip install paramiko requests
```

## 故障排查

### SSH 连接失败
- 确认实例已开机
- 确认 SSH 密码正确
- 等待 30-60 秒后重试 (SSH 服务启动需要时间)

### API 调用失败
- 确认 API Token 正确且未过期
- 确认实例 ID 正确

### 训练失败
- 检查 AutoDL 上的日志: `ssh root@xxx.autodl.com`
- 手动执行训练排查: `cd /root/rplhr-ct-training && python autodl_train.py`
