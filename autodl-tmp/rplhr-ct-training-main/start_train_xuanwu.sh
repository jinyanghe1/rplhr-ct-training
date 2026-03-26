#!/bin/bash
# 宣武数据集训练启动脚本
# Xuanwu Dataset Training Start Script
# 
# 配置说明:
# - 使用保守数据增强方案
# - 训练50个epoch
# - 输出到指定目录并标明宣武数据集

echo "========================================"
echo "🚀 启动宣武数据集训练"
echo "📊 Dataset: XUANWU (宣武数据集)"
echo "📅 Date: $(date)"
echo "========================================"

# 进入代码目录
cd /root/autodl-tmp/rplhr-ct-training-main/code

# 创建输出目录
mkdir -p ../train_log/xuanwu_50epoch
touch ../train_log/xuanwu_50epoch/DATASET_XUANWU

# 启动训练
# 使用 fire 库的正确参数格式
python trainxuanwu.py train \
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
    --test_num_workers=2

echo "========================================"
echo "✅ 训练完成"
echo "📁 输出目录:"
echo "   - 模型: ../model/dataset01_xuanwu/xuanwu_50epoch/"
echo "   - 日志: ../train_log/dataset01_xuanwu/xuanwu_50epoch/"
echo "   - 验证输出: ../val_output/"
echo "   - 可视化: ../val_viz/"
echo "========================================"
