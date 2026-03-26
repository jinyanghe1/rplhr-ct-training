#!/bin/bash
# 宣武数据集训练启动脚本（后台运行）
# Xuanwu Dataset Training Start Script (Background)

LOG_FILE="/root/autodl-tmp/rplhr-ct-training-main/train_xuanwu_50epoch_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "🚀 启动宣武数据集训练 (后台模式)"
echo "📊 Dataset: XUANWU (宣武数据集)"
echo "📅 Date: $(date)"
echo "📁 Log: $LOG_FILE"
echo "========================================"

cd /root/autodl-tmp/rplhr-ct-training-main/code

# 创建标记文件
mkdir -p ../train_log/dataset01_xuanwu/xuanwu_50epoch
echo "XUANWU DATASET - 50 EPOCHS" > ../train_log/dataset01_xuanwu/xuanwu_50epoch/DATASET_INFO.txt
echo "Start: $(date)" >> ../train_log/dataset01_xuanwu/xuanwu_50epoch/DATASET_INFO.txt
echo "Features: Conservative Augmentation" >> ../train_log/dataset01_xuanwu/xuanwu_50epoch/DATASET_INFO.txt

# 后台启动训练
nohup python trainxuanwu.py train \
    --net_idx="xuanwu_50epoch" \
    --path_key="dataset01_xuanwu" \
    --epoch=50 \
    --use_augmentation=True \
    --aug_prob=0.5 \
    --clip_ct=True \
    --min_hu=-1024 \
    --max_hu=3071 \
    --normalize_ct=False \
    --num_workers=2 \
    --test_num_workers=2 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo $PID > /tmp/train_xuanwu.pid

echo ""
echo "监控命令:"
echo "  查看日志: tail -f $LOG_FILE"
echo "  查看进程: ps aux | grep trainxuanwu"
echo "  停止训练: kill $PID"
echo ""
echo "训练将在后台运行，预计需要数小时完成50个epoch"
