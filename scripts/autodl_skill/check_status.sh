#!/bin/bash
#===============================================================================
# 检查 AutoDL 训练状态
#===============================================================================

set -e

RED='\033[0;31m'
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

echo ""
echo "=============================================="
echo "  AutoDL 训练状态检查"
echo "=============================================="
echo ""

# GPU 状态
echo -e "${BLUE}[GPU]${NC}"
ssh_cmd "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
echo ""

# 训练进程
echo -e "${BLUE}[Training Processes]${NC}"
ssh_cmd "ps aux | grep trainxuanwu | grep -v grep || echo 'No training process running'"
echo ""

# 最新日志
echo -e "${BLUE}[Latest Log]${NC}"
LOG_FILE=$(ssh_cmd "ls -t $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null | head -1" | tail -1)
if [[ -n "$LOG_FILE" && "$LOG_FILE" != *"No such file"* ]]; then
    echo "Log: $LOG_FILE"
    ssh_cmd "tail -20 '$LOG_FILE'"
else
    echo "No log file found"
fi
echo ""

# 最新指标
echo -e "${BLUE}[Latest Metrics]${NC}"
ssh_cmd "grep -E 'PSNR|SSIM|MSE|L1' $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null | tail -10 || echo 'No metrics found'"
echo ""

echo "=============================================="
