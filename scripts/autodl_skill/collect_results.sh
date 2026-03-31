#!/bin/bash
#===============================================================================
# 收集 AutoDL 训练结果到本地
#===============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ssh_cmd() {
    expect << EOF
        set timeout 60
        log_user 0
        spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $AUTODL_PORT $AUTODL_USER@$AUTODL_HOST
        expect "password:"
        send "$AUTODL_PASS\r"
        expect "$ "
        send "$1\r"
        expect "$ "
        send "exit\r"
        expect eof
EOF
}

collect() {
    local run_dir="$LOCAL_LOG_DIR/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"

    echo -e "${BLUE}[INFO]${NC} 下载训练日志..."
    ssh_cmd "cat $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null" > "$run_dir/training.log"

    echo -e "${BLUE}[INFO]${NC} 提取关键指标..."
    local psnr=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
    local ssim=$(grep -oP 'SSIM[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
    local mse=$(grep -oP 'MSE[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")

    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"psnr\": \"$psnr\","
        echo "  \"ssim\": \"$ssim\","
        echo "  \"mse\": \"$mse\""
        echo "}"
    } > "$run_dir/metrics.json"

    {
        echo "# 训练心得 - $(date)"
        echo ""
        echo "## 指标"
        echo "- PSNR: $psnr"
        echo "- SSIM: $ssim"
        echo "- MSE: $mse"
        echo ""
        echo "## 日志摘要"
        tail -100 "$run_dir/training.log"
    } > "$run_dir/insights.md"

    echo -e "${GREEN}[SUCCESS]${NC} 结果已保存: $run_dir/"
    cat "$run_dir/metrics.json"
}

collect
