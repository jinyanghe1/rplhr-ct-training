#!/bin/bash
# 启动训练

set -e

WORK_DIR="/root/autodl-tmp/RPLHR-CT"
CODE_DIR="$WORK_DIR/code/code"
DATA_DIR="$WORK_DIR/data"

echo "========================================"
echo "启动训练"
echo "========================================"

# 检查数据
if [ ! -d "$DATA_DIR/train/5mm" ] || [ ! -d "$DATA_DIR/train/1mm" ]; then
    echo "错误: 训练数据不存在，请先运行 download_data.sh"
    exit 1
fi

# 检查代码
if [ ! -d "$CODE_DIR" ]; then
    echo "错误: 代码目录不存在"
    exit 1
fi

cd "$CODE_DIR"

# 更新代码（可选）
echo "[1/2] 拉取最新代码..."
git pull origin main
echo "✓ 代码更新完成"

# 创建模型保存目录
mkdir -p "$WORK_DIR/model"
mkdir -p "$WORK_DIR/val_output"

# 启动训练
echo ""
echo "[2/2] 开始训练..."
echo ""

# 使用 nohup 后台运行
nohup python train.py train \
    --path_key SRM \
    --gpu_idx 0 \
    --net_idx TVSRN \
    --epoch 2000 \
    --lr 0.0003 \
    --train_bs 1 \
    --val_bs 1 \
    > "$WORK_DIR/train.log" 2>&1 &

PID=$!
echo "训练进程已启动，PID: $PID"
echo ""
echo "查看日志: tail -f $WORK_DIR/train.log"
echo "查看GPU: watch -n 1 nvidia-smi"
echo ""
echo "========================================"
