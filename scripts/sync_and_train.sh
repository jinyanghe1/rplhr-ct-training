#!/bin/bash
#===============================================================================
# AutoDL 同步与训练脚本
# 用法: ./scripts/sync_and_train.sh [epochs]
#===============================================================================

set -e

# 配置
REMOTE_HOST="connect.westd.seetacloud.com"
REMOTE_PORT="23086"
REMOTE_USER="root"
REMOTE_PASS="Z9wdTD/ZA6fZ"
REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"
EPOCHS="${1:-50}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#===============================================================================
# SSH 执行命令
#===============================================================================
run_ssh() {
    local cmd="$1"
    expect << EOF
        set timeout 120
        spawn ssh -o StrictHostKeyChecking=no -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST
        expect "password:"
        send "$REMOTE_PASS\r"
        expect "# "
        send "$cmd\r"
        expect "# "
        send "exit\r"
        expect eof
EOF
}

#===============================================================================
# 主流程
#===============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  AutoDL 同步与训练"
    echo "=============================================="
    log_info "Git Commit: $(git rev-parse --short HEAD)"
    log_info "Epochs: $EPOCHS"
    echo ""

    # Step 1: 检查 GPU
    log_info "检查 GPU 状态..."
    run_ssh "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

    # Step 2: Git 同步
    log_info "同步代码 (git pull)..."
    run_ssh "source /etc/network_turbo && cd /root && git pull origin main"

    # Step 3: 确认训练脚本存在
    log_info "确认训练脚本..."
    run_ssh "ls -la $REPO_PATH/code/trainxuanwu.py"

    # Step 4: 启动训练
    log_info "启动训练 ($EPOCHS epochs, 宣武数据集 1:4)..."
    run_ssh "cd $REPO_PATH/code && \
        LOG_FILE=\"../train_autodl_\$(date +%Y%m%d_%H%M%S).log\" && \
        echo \"Log: \$LOG_FILE\" && \
        nohup python trainxuanwu.py train \
            --net_idx=xuanwu_50epoch \
            --path_key=dataset01_xuanwu \
            --epoch=$EPOCHS \
            --use_augmentation=True \
            --aug_prob=0.5 \
            --clip_ct=True \
            --min_hu=-1024 \
            --max_hu=3071 \
            --normalize_ct=False \
            --num_workers=4 \
            --test_num_workers=2 \
            > \"\$LOG_FILE\" 2>&1 & \
        echo \"Started: \$(date)\" && \
        echo \"PID: \$!\""

    # Step 5: 等待启动
    sleep 5

    # Step 6: 确认进程运行
    log_info "确认训练进程..."
    run_ssh "ps aux | grep trainxuanwu | grep -v grep | head -2"

    # Step 7: 查看日志
    log_info "初始日志 (最后 20 行)..."
    run_ssh "ls -t $REPO_PATH/train_autodl_*.log 2>/dev/null | head -1 | xargs tail -20 2>/dev/null || echo '日志文件尚未创建'"

    echo ""
    log_success "训练已在后台启动!"
    echo ""
    echo "=============================================="
    echo "  后续操作"
    echo "=============================================="
    echo "  监控日志: ssh $REMOTE_USER@$REMOTE_HOST -p $REMOTE_PORT"
    echo "  tail -f $REPO_PATH/train_autodl_*.log"
    echo "=============================================="
}

main "$@"
