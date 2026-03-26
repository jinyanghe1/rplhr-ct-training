#!/bin/bash
# 宣武数据集训练 - 修复 #1 验证脚本 (10 Epoch)
# 修复内容:
#   1. 启用数据归一化 (normalize_ct=True)
#   2. 使用仅几何增强 (GEOMETRY_ONLY_AUG)

echo "========================================"
echo "🚀 启动宣武数据集训练 - 修复 #1 验证"
echo "📊 Dataset: XUANWU (宣武数据集)"
echo "🔧 Fix: 归一化 + 仅几何增强"
echo "📅 Date: $(date)"
echo "========================================"

cd /root/autodl-tmp/rplhr-ct-training-main/code

# 创建输出目录
mkdir -p ../train_log/dataset01_xuanwu/xuanwu_fix1_10epoch
echo "XUANWU DATASET - FIX #1 VALIDATION (10 Epoch)" > ../train_log/dataset01_xuanwu/xuanwu_fix1_10epoch/DATASET_INFO.txt
echo "Start: $(date)" >> ../train_log/dataset01_xuanwu/xuanwu_fix1_10epoch/DATASET_INFO.txt
echo "Features: normalize_ct=True + GEOMETRY_ONLY_AUG" >> ../train_log/dataset01_xuanwu/xuanwu_fix1_10epoch/DATASET_INFO.txt

# 启动训练
LOG_FILE="/root/autodl-tmp/rplhr-ct-training-main/train_xuanwu_fix1_10epoch_$(date +%Y%m%d_%H%M%S).log"

python trainxuanwu.py train \
    --net_idx="xuanwu_fix1_10epoch" \
    --path_key="dataset01_xuanwu" \
    --epoch=10 \
    --use_augmentation=True \
    --aug_prob=0.5 \
    --normalize_ct=True \
    --window_center=40 \
    --window_width=400 \
    --clip_ct=False \
    --num_workers=2 \
    --test_num_workers=2 \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "✅ 10 Epoch 验证完成"
echo "📁 日志: $LOG_FILE"
echo "========================================"
