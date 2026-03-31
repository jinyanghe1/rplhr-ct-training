# AutoDL MLOps 自动化方案 - RPLHR-CT Training

## 项目背景

- **Repo**: https://github.com/jinyanghe1/rplhr-ct-training
- **本地路径**: `/root/.openclaw/workspace/rplhr-ct-training/`
- **任务**: CT 体积超分辨率 (厚层→薄层)
- **新需求**: 支持 1:4 比例（原有是 1:5）

## AutoDL 数据集布局

```
/autodl-tmp/
├── dataset_hrct_full/          # 完整 HRCT 训练集 (~30GB)
├── dataset_xuanwu_ct/          # 宣武医院颅部 CT 薄厚扫数据集
└── dataset_hrct_tiny/          # HRCT 子集 (小训练集)
```

## 核心修改: 1:4 比例适配

### 现有问题
- 原有模型: `ratio = 5` (1:5 比例)
- 训练数据: 大多是 1:4 比例
- 之前方案: 插值模拟 1:4 → 1:5，**效果很差**

### 解决方案
**保持 Backbone 完全一致，仅修改 ratio 参数**

| 配置 | ratio | 适用场景 |
|------|-------|----------|
| default.txt | 5 | 1:5 比例数据 |
| **ratio4.txt** | **4** | **1:4 比例数据 (新增)** |

### 关键代码修改点

**1. 新增配置文件** (已完成)
```bash
config/ratio4.txt  # ratio = 4，其他参数完全一致
```

**2. 模型支持动态 ratio** (model_TransSR.py)
```python
# 原有代码已支持通过 opt.ratio 动态计算
self.out_z = (opt.c_z - 1) * opt.ratio + 1  # 自动适应 ratio=4 或 5
```

**3. 启动命令对比**
```bash
# 1:5 比例 (原有)
python train.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN

# 1:4 比例 (新增)
python train.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN --config ../config/ratio4.txt
```

## AutoDL MLOps 工作流设计

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        本地开发环境                               │
│  ┌──────────────┐    git push    ┌──────────────────────────┐  │
│  │ 修改代码      │ ─────────────→ │  GitHub (rplhr-ct-       │  │
│  │ - 新增 ratio4 │                │       training)          │  │
│  │ - 调参优化    │                └────────────┬─────────────┘  │
│  └──────────────┘                             │                │
│         ↑                                     │                │
│         │         ┌───────────────────────────┘                │
│         │         │                                            │
│         │    Webhook / 定时触发                                 │
│         │         │                                            │
│         │         ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              本地控制器 (local_controller.py)              │ │
│  │  - 接收 GitHub push 事件 or 定时轮询                        │ │
│  │  - 调用 AutoDL API 开机                                     │ │
│  │  - SSH 连接执行训练                                         │ │
│  │  - 下载结果 & 关机                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ SSH
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AutoDL 服务器 (GPU)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 1. git pull  │→ │ 2. 训练模型   │→ │ 3. 指标评估 & 存档    │  │
│  │    最新代码   │  │    TVSRN     │  │    (SSIM/MSE/PSNR)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                             │                   │
│                              优于历史最佳? ──┼──→ 存档模型+报告  │
│                                             │                   │
│                              否 ────────────┼──→ 仅记录日志     │
│                                             ▼                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. 关机 (节省费用)                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 代码实现

### 1. AutoDL API 客户端

```python
# scripts/autodl/autodl_client.py
import requests
import time

class AutoDLClient:
    def __init__(self, api_token):
        self.token = api_token
        self.base_url = "https://www.autodl.com/api/v1"
    
    def get_instance(self, instance_id):
        """获取实例状态"""
        resp = requests.get(
            f"{self.base_url}/instances/{instance_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return resp.json()
    
    def power_on(self, instance_id):
        """开机"""
        return requests.post(
            f"{self.base_url}/instances/{instance_id}/power_on",
            headers={"Authorization": f"Bearer {self.token}"}
        )
    
    def power_off(self, instance_id):
        """关机"""
        return requests.post(
            f"{self.base_url}/instances/{instance_id}/power_off",
            headers={"Authorization": f"Bearer {self.token}"}
        )
    
    def wait_for_running(self, instance_id, timeout=300):
        """等待实例启动完成"""
        start = time.time()
        while time.time() - start < timeout:
            info = self.get_instance(instance_id)
            if info.get('status') == 'running':
                return True
            time.sleep(10)
        raise TimeoutError("实例启动超时")
```

### 2. 训练流水线 (AutoDL 端)

```python
# autodl_train.py - 放在 repo 根目录，AutoDL 上执行
import os
import json
import subprocess
import torch
import numpy as np
from datetime import datetime

def git_sync():
    """同步最新代码"""
    subprocess.run(["git", "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    commit = subprocess.getoutput("git rev-parse --short HEAD")
    print(f"✅ 代码已同步: {commit}")
    return commit

def run_training(config_file="config/default.txt"):
    """执行训练"""
    cmd = [
        "python", "code/train.py", "train",
        "--path_key", "SRM",
        "--gpu_idx", "0",
        "--net_idx", "TVSRN",
        "--config", config_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def evaluate_metrics(checkpoint_path, test_data_path):
    """评估指标: SSIM, MSE, PSNR"""
    # 加载模型并测试
    # 这里调用 code/val.py 或 code/test.py
    cmd = [
        "python", "code/val.py", "val",
        "--path_key", "SRM",
        "--gpu_idx", "0",
        "--net_idx", "TVSRN"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 解析输出获取指标 (需要根据实际输出格式调整)
    metrics = {
        "psnr": parse_psnr(result.stdout),
        "ssim": parse_ssim(result.stdout),
        "mse": parse_mse(result.stdout),
        "timestamp": datetime.now().isoformat()
    }
    return metrics

def check_and_archive(metrics, config_name):
    """检查是否历史最佳，存档模型"""
    history_file = "metrics_history.json"
    
    # 加载历史
    if os.path.exists(history_file):
        history = json.load(open(history_file))
    else:
        history = {"best": {}, "runs": []}
    
    # 判断改进 (PSNR 越高越好)
    best_psnr = history["best"].get("psnr", 0)
    improved = metrics["psnr"] > best_psnr
    
    if improved:
        # 存档模型
        archive_name = f"model_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(f"archives/{archive_name}", exist_ok=True)
        
        # 复制模型权重
        subprocess.run([
            "cp", "model/TVSRN/best_model.pth",
            f"archives/{archive_name}/model.pth"
        ])
        
        # 保存配置和指标
        json.dump(metrics, open(f"archives/{archive_name}/metrics.json", "w"))
        json.dump({"config": config_name}, open(f"archives/{archive_name}/config.json", "w"))
        
        # 更新历史最佳
        history["best"] = metrics
        print(f"🏆 新历史最佳! 已存档: archives/{archive_name}")
    
    # 记录本次运行
    history["runs"].append(metrics)
    json.dump(history, open(history_file, "w"))
    
    return improved

def generate_report(metrics, improved, config_name):
    """生成训练报告"""
    report = f"""
# RPLHR-CT 训练报告

**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**配置**: {config_name}

## 指标
- PSNR: {metrics['psnr']:.2f} dB
- SSIM: {metrics['ssim']:.4f}
- MSE: {metrics['mse']:.6f}

## 历史最佳对比
- 当前最佳 PSNR: {metrics['psnr']:.2f}

## 是否改进
{'✅ 是 - 已存档最佳模型' if improved else '❌ 否'}

## Git Commit
{metrics.get('git_commit', 'N/A')}
"""
    report_path = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    return report_path

if __name__ == "__main__":
    # 根据数据集选择配置
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/default.txt"
    
    commit = git_sync()
    run_training(config)
    metrics = evaluate_metrics("model/TVSRN/best_model.pth", "data/test")
    metrics["git_commit"] = commit
    improved = check_and_archive(metrics, config)
    report = generate_report(metrics, improved, config)
    print(f"📄 报告: {report}")
```

### 3. 本地控制器

```python
# scripts/autodl/local_controller.py
import paramiko
import time
from autodl_client import AutoDLClient

class RPLHRController:
    def __init__(self, api_token, instance_id, ssh_config):
        self.autodl = AutoDLClient(api_token)
        self.instance_id = instance_id
        self.ssh_config = ssh_config
        self.ssh = None
    
    def start_and_connect(self):
        """开机并建立 SSH 连接"""
        # 开机
        self.autodl.power_on(self.instance_id)
        self.autodl.wait_for_running(self.instance_id)
        
        # SSH 连接
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 等待 SSH 服务启动
        time.sleep(30)
        
        self.ssh.connect(
            hostname=self.ssh_config["host"],
            port=self.ssh_config["port"],
            username=self.ssh_config["username"],
            password=self.ssh_config["password"]
        )
        print("✅ SSH 连接成功")
    
    def run_experiment(self, config_file="config/ratio4.txt"):
        """运行实验"""
        cmd = f"cd /root/rplhr-ct-training && python autodl_train.py {config_file}"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        
        # 实时输出
        for line in stdout:
            print(line, end="")
        
        error = stderr.read().decode()
        if error:
            print(f"❌ 错误: {error}")
            return None
        
        return stdout.read().decode()
    
    def download_results(self):
        """下载结果到本地"""
        sftp = self.ssh.open_sftp()
        
        # 下载报告
        remote_reports = "/root/rplhr-ct-training/reports/"
        local_reports = "/root/.openclaw/workspace/rplhr-ct-training/autodl_reports/"
        
        try:
            sftp.get(
                remote_reports + "latest_report.md",
                local_reports + f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"
            )
        except:
            print("⚠️ 报告下载失败")
        
        sftp.close()
    
    def shutdown(self):
        """关机"""
        if self.ssh:
            self.ssh.close()
        self.autodl.power_off(self.instance_id)
        print("✅ 服务器已关机")
    
    def run_pipeline(self, config_file="config/ratio4.txt"):
        """完整流程"""
        try:
            self.start_and_connect()
            self.run_experiment(config_file)
            self.download_results()
        finally:
            self.shutdown()

# 使用示例
if __name__ == "__main__":
    controller = RPLHRController(
        api_token="your_autodl_token",
        instance_id="your_instance_id",
        ssh_config={
            "host": "region.autodl.com",
            "port": 22,
            "username": "root",
            "password": "your_password"
        }
    )
    
    # 运行 1:4 比例实验
    controller.run_pipeline("config/ratio4.txt")
```

## 使用流程

### 1. 本地开发 & Push

```bash
cd /root/.openclaw/workspace/rplhr-ct-training/

# 修改代码 (例如: 优化 TVSRN 架构)
vim code/net/model_TransSR.py

# 提交并推送
git add .
git commit -m "优化 1:4 比例 Decoder 注意力机制"
git push origin main
```

### 2. 触发 AutoDL 训练

**方式 A: 手动触发**
```bash
python scripts/autodl/local_controller.py
```

**方式 B: GitHub Webhook (自动)**
```bash
# 本地启动 webhook 接收服务
python scripts/autodl/webhook_server.py
```

**方式 C: 定时任务**
```bash
# crontab -e
# 每 6 小时检查一次
0 */6 * * * cd /root/.openclaw/workspace/rplhr-ct-training && python scripts/autodl/local_controller.py
```

### 3. 查看结果

```bash
# 训练报告
cat autodl_reports/report_*.md

# 历史指标
cat metrics_history.json | jq '.best'

# 最佳模型存档
ls archives/
```

## 实验计划: 1:4 比例优化

### 第一步: Baseline (ratio=4)
```bash
# 使用原有架构，仅修改 ratio=4
python train.py train --config ../config/ratio4.txt
```

### 第二步: 对比实验

| 实验 | 修改点 | 预期效果 |
|------|--------|----------|
| ratio4_baseline | ratio=4，其他不变 | Baseline |
| ratio4_decoder | 调整 Decoder 层数 | 提升细节 |
| ratio4_pos_enc | 优化位置编码 | 改善 Z 轴连续性 |
| ratio4_dual | 双尺度训练 | 增强泛化 |

### 第三步: 指标追踪

```python
# 自动记录到 metrics_history.json
{
  "best": {
    "psnr": 42.35,
    "ssim": 0.9823,
    "mse": 0.00042,
    "config": "ratio4_decoder",
    "commit": "a1b2c3d"
  },
  "runs": [...]
}
```

## 成本估算

- AutoDL RTX 3090: ~¥1.8/小时
- 单次训练 (2小时): ¥3.6
- 每天 3 次实验: ¥10.8
- **自动关机节省**: 训练时间外零费用

## 下一步行动

1. **新增 ratio4 配置** ✅ (已完成 - config/ratio4.txt)
2. **编写 AutoDL API 客户端** 🔄 (下一步)
3. **编写训练流水线脚本** 🔄 (下一步)
4. **编写本地控制器** 🔄 (下一步)
5. **提交代码到 GitHub** 🔄 (准备中)
6. **测试完整流程**

---

## 当前状态

### ✅ 已完成
- [x] 分析现有代码结构，定位 ratio 参数
- [x] 创建 1:4 比例配置文件 (config/ratio4.txt)
- [x] 设计完整 MLOps 架构方案
- [x] 编写方案文档 (AUTODL_MLOPS_PLAN.md)

### 🔄 待完成
- [ ] 编写 scripts/autodl/autodl_client.py
- [ ] 编写 autodl_train.py (训练流水线)
- [ ] 编写 scripts/autodl/local_controller.py
- [ ] 提交所有代码到 GitHub
- [ ] 配置 AutoDL API 密钥
- [ ] 测试端到端流程

---

## 立即执行

需要我现在开始编写 AutoDL API 客户端和训练流水线代码吗？完成后立即提交到 GitHub。
