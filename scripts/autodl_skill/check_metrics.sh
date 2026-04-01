#!/bin/bash
#===============================================================================
# Metrics CSV检查与趋势分析
# 用法: ./check_metrics.sh [--plot]
#===============================================================================

HOST="connect.westd.seetacloud.com"
PORT="23086"
USER="root"
CHECKPOINT_PATH="/root/autodl-tmp/rplhr-ct-training-main/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv"

echo "=============================================="
echo "  Metrics 趋势分析"
echo "=============================================="
echo ""

# 获取metrics
METRICS=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p $PORT $USER@$HOST "cat $CHECKPOINT_PATH 2>/dev/null")

if [ -z "$METRICS" ]; then
    echo "[ERROR] 无法获取metrics.csv"
    exit 1
fi

# 显示表头
echo "完整Metrics数据:"
echo "$METRICS" | column -t -s','
echo ""

# 统计信息
echo "统计摘要:"
echo "------------------------------"

# 获取所有PSNR值
PSNR_VALUES=$(echo "$METRICS" | tail -n +2 | cut -d',' -f4 | grep -v '^$')
SSIM_VALUES=$(echo "$METRICS" | tail -n +2 | cut -d',' -f5 | grep -v '^$')

if [ -n "$PSNR_VALUES" ]; then
    BEST_PSNR=$(echo "$PSNR_VALUES" | sort -n | tail -1)
    WORST_PSNR=$(echo "$PSNR_VALUES" | sort -n | head -1)
    AVG_PSNR=$(echo "$PSNR_VALUES" | awk '{sum+=$1; count++} END {printf "%.2f", sum/count}')
    
    echo "PSNR (dB):"
    echo "  最佳: $BEST_PSNR"
    echo "  最差: $WORST_PSNR"
    echo "  平均: $AVG_PSNR"
fi

if [ -n "$SSIM_VALUES" ]; then
    BEST_SSIM=$(echo "$SSIM_VALUES" | sort -n | tail -1)
    echo ""
    echo "SSIM:"
    echo "  最佳: $BEST_SSIM"
fi

echo "------------------------------"
echo ""

# 趋势分析
if [ -n "$PSNR_VALUES" ]; then
    PSNR_COUNT=$(echo "$PSNR_VALUES" | wc -l)
    if [ $PSNR_COUNT -ge 2 ]; then
        FIRST_PSNR=$(echo "$PSNR_VALUES" | head -1)
        LAST_PSNR=$(echo "$PSNR_VALUES" | tail -1)
        TREND=$(echo "$LAST_PSNR - $FIRST_PSNR" | bc)
        
        echo "趋势分析:"
        if (( $(echo "$TREND > 0" | bc -l) )); then
            echo "  ✅ PSNR上升: +$TREND dB (从 $FIRST_PSNR 到 $LAST_PSNR)"
        else
            echo "  ⚠️ PSNR下降: $TREND dB (从 $FIRST_PSNR 到 $LAST_PSNR)"
        fi
    fi
fi

echo ""
echo "=============================================="
