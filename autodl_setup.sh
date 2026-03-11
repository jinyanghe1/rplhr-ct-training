#!/bin/bash
# AutoDL 实例初始化脚本
# 在实例启动时自动执行

set -e

echo "========================================"
echo "AutoDL 训练环境初始化"
echo "========================================"

# 设置工作目录
WORK_DIR="/root/autodl-tmp/RPLHR-CT"
DATA_DIR="$WORK_DIR/data"
CODE_DIR="$WORK_DIR/code"
mkdir -p "$WORK_DIR"

echo "[1/5] 安装 aliyunpan 工具..."
if ! command -v aliyunpan &> /dev/null; then
    curl -fsSL http://file.tickstep.com/apt/pgp | gpg --dearmor | \
        tee /etc/apt/trusted.gpg.d/tickstep-packages-archive-keyring.gpg > /dev/null
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/tickstep-packages-archive-keyring.gpg arch=amd64,arm64] http://file.tickstep.com/apt aliyunpan main" | \
        tee /etc/apt/sources.list.d/tickstep-aliyunpan.list > /dev/null
    apt-get update && apt-get install -y aliyunpan
    echo "✓ aliyunpan 安装完成"
else
    echo "✓ aliyunpan 已存在"
fi

echo "[2/5] 从 GitHub 拉取最新代码..."
if [ -d "$CODE_DIR/.git" ]; then
    cd "$CODE_DIR"
    git pull origin main
else
    rm -rf "$CODE_DIR"
    # 注意：请替换为你的 GitHub 仓库地址
    echo "请修改脚本中的 GitHub 仓库地址！"
    # git clone https://github.com/你的用户名/rplhr-ct-training.git "$CODE_DIR"
fi
echo "✓ 代码拉取完成"

echo "[3/5] 安装 Python 依赖..."
if [ -d "$CODE_DIR/code" ]; then
    cd "$CODE_DIR/code"
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -q SimpleITK numpy tqdm opencv-python fire timm einops
    echo "✓ 依赖安装完成"
fi

echo "[4/5] 修改配置文件..."
if [ -d "$CODE_DIR/code/config" ]; then
    cat > "$CODE_DIR/code/config/SRM_dict.json" << 'EOFCFG'
{"path_img": "/root/autodl-tmp/RPLHR-CT/data/"}
EOFCFG
    echo "✓ 配置修改完成"
fi

echo "[5/5] 设置环境变量..."
echo "export PYTHONPATH=$CODE_DIR/code:\$PYTHONPATH" >> ~/.bashrc
echo "export DATA_DIR=$DATA_DIR" >> ~/.bashrc
echo "export CODE_DIR=$CODE_DIR" >> ~/.bashrc

echo "========================================"
echo "初始化完成！"
echo "工作目录: $WORK_DIR"
echo "数据目录: $DATA_DIR"
echo "代码目录: $CODE_DIR"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 登录阿里云盘: aliyunpan login"
echo "  2. 下载数据: bash download_data.sh"
echo "  3. 开始训练: bash start_training.sh"
