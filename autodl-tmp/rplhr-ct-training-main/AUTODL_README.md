# AutoDL 云训练方案

本方案实现：**本地修改代码 → GitHub 同步 → AutoDL 训练**

## 📁 文件说明

| 文件 | 用途 | 运行位置 |
|-----|------|---------|
| `prepare_data_local.sh` | 打包本地数据 | 本地 Mac |
| `autodl_setup.sh` | AutoDL 环境初始化 | AutoDL |
| `download_data.sh` | 从阿里云盘下载数据 | AutoDL |
| `start_training.sh` | 启动训练 | AutoDL |
| `quick_start.sh` | 一键启动 | AutoDL |

---

## 🚀 快速开始

### 第一步：准备 GitHub 仓库

```bash
# 在本地项目目录
cd /Users/hejinyang/毕业设计_0306/RPLHR-CT-main

# 初始化 Git
git init

# 添加远程仓库（替换为你的地址）
git remote add origin https://github.com/你的用户名/rplhr-ct-training.git

# 提交代码
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 第二步：打包并上传数据

```bash
# 本地打包数据
bash prepare_data_local.sh ../data ./packaged_data

# 在阿里云盘创建 RPLHR-CT-Dataset 文件夹
# 上传 packaged_data 中的 .tar.gz 文件到该文件夹
```

### 第三步：AutoDL 初始化

```bash
# 1. 创建 AutoDL 实例（PyTorch 2.0 + CUDA 11.8）
# 2. 通过 JupyterLab 或 SSH 连接

# 3. 上传本项目的脚本文件（autodl_setup.sh, download_data.sh, start_training.sh）

# 4. 执行初始化
bash autodl_setup.sh

# 5. 登录阿里云盘
aliyunpan login

# 6. 下载数据
bash download_data.sh
```

### 第四步：启动训练

```bash
# 启动训练（自动拉取最新代码）
bash start_training.sh

# 查看日志
tail -f /root/autodl-tmp/RPLHR-CT/train.log

# 查看 GPU
watch -n 1 nvidia-smi
```

---

## 🔄 日常开发流程

### 修改代码并训练

```bash
# ===== 本地 =====
# 修改代码...
vim code/train.py

# 推送更新
git add .
git commit -m "优化学习率"
git push origin main

# ===== AutoDL =====
# 重新连接到实例
ssh root@实例地址 -p 端口号

# 拉取最新代码并训练
cd /root/autodl-tmp/RPLHR-CT/code/code
git pull origin main
nohup python train.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN > train.log 2>&1 &
```

---

## ⚙️ 配置修改

### 修改 GitHub 仓库地址

编辑以下文件，替换为你的 GitHub 仓库地址：
- `autodl_setup.sh` (第 28 行)
- `start_training.sh` (第 20 行)

### 修改阿里云盘路径

编辑 `download_data.sh`，修改为你的阿里云盘文件夹路径：
```bash
FILES=(
    "你的文件夹路径/train_5mm.tar.gz"
    ...
)
```

---

## 💡 提示

1. **代码更新**: 修改代码后推送到 GitHub，AutoDL 会自动拉取最新版本
2. **数据备份**: 定期将模型结果上传到阿里云盘
3. **费用控制**: 不训练时关闭 AutoDL 实例
4. **日志查看**: 使用 `tail -f train.log` 实时查看训练进度

---

## 📚 详细文档

查看 `AUTODL_TRAINING_GUIDE.md` 获取完整的 Step-by-Step 指南。
