#!/bin/bash
#===============================================================================
# 下载 AutoDL 训练日志到本地
#===============================================================================

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
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

echo -e "${BLUE}[INFO]${NC} 下载所有训练日志..."

LOCAL_LOG_DIR="$SCRIPT_DIR/logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOCAL_LOG_DIR"

# 下载最新的日志文件
LOG_COUNT=0
for i in {1..5}; do
    LOG_FILE=$(ssh_cmd "ls -t $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null | head -$i | tail -1" | tail -1)
    if [[ -n "$LOG_FILE" && "$LOG_FILE" != *"No such"* ]]; then
        FILENAME=$(basename "$LOG_FILE")
        ssh_cmd "cat '$LOG_FILE'" > "$LOCAL_LOG_DIR/$FILENAME"
        echo -e "${GREEN}[+]${NC} Downloaded: $FILENAME"
        LOG_COUNT=$((LOG_COUNT + 1))
    fi
done

# 下载模型文件 (如果存在)
echo ""
echo -e "${BLUE}[INFO]${NC} 检查模型文件..."
ssh_cmd "ls -la $AUTODL_REPO_PATH/model/ 2>/dev/null || echo 'No model directory'"

echo ""
echo -e "${GREEN}[SUCCESS]${NC} 下载完成: $LOG_COUNT 个日志文件"
echo "保存位置: $LOCAL_LOG_DIR/"
ls -la "$LOCAL_LOG_DIR/"
