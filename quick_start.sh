#!/bin/bash
# AutoDL 一键启动脚本
# 包含：环境初始化、数据下载、开始训练

set -e

echo "========================================"
echo "RPLHR-CT AutoDL 一键启动"
echo "========================================"

WORK_DIR="/root/autodl-tmp/RPLHR-CT"

# 第一步：环境初始化
if [ -f "autodl_setup.sh" ]; then
    bash autodl_setup.sh
else
    echo "错误: 未找到 autodl_setup.sh"
    exit 1
fi

# 第二步：检查是否需要下载数据
if [ ! -d "$WORK_DIR/data/train/5mm" ] || [ ! -d "$WORK_DIR/data/train/1mm" ]; then
    echo ""
    echo "需要下载训练数据..."
    
    # 检查是否已登录
    if ! aliyunpan whoami &> /dev/null; then
        echo ""
        echo "========================================"
        echo "请先登录阿里云盘"
        echo "========================================"
        aliyunpan login
    fi
    
    bash download_data.sh
fi

# 第三步：启动训练
bash start_training.sh

echo ""
echo "========================================"
echo "一键启动完成！"
echo "========================================"
