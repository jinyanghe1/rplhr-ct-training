#!/bin/bash
#===============================================================================
# 训练状态检查脚本
# 用法: ./check_training.sh
#===============================================================================

# SSH配置
HOST="connect.westd.seetacloud.com"
PORT="23086"
USER="root"
REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# SSH执行
ssh_run() {
    ssh -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -p $PORT $USER@$HOST "$1" 2>/dev/null
}

echo "=============================================="
echo "  训练状态检查"
echo "=============================================="
echo ""

# 1. 检查训练进程
log_info "1. 检查训练进程..."
PID=$(ssh_run "ps aux | grep '[t]rainxuanwu' | awk '{print \$2}' | head -1")
if [ -n "$PID" ]; then
    log_info "   训练进程运行中 (PID: $PID)"
    
    # 获取GPU使用情况
    GPU_INFO=$(ssh_run "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep $PID" || echo "N/A")
    if [ "$GPU_INFO" != "N/A" ]; then
        log_info "   GPU使用: $GPU_INFO"
    fi
else
    log_warn "   训练进程未运行"
fi
echo ""

# 2. 检查当前Epoch
log_info "2. 检查当前训练进度..."
CURRENT_EPOCH=$(ssh_run "grep -E 'Epoch [0-9]+' $REPO_PATH/train_daemon.log 2>/dev/null | tail -1 | grep -oE 'Epoch [0-9]+' | awk '{print \$2}'")
if [ -n "$CURRENT_EPOCH" ]; then
    log_info "   当前Epoch: $CURRENT_EPOCH"
else
    log_warn "   无法获取当前Epoch"
fi
echo ""

# 3. 获取最新Metrics
log_info "3. 获取最新指标..."
METRICS=$(ssh_run "cat $REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv 2>/dev/null | tail -1")
if [ -n "$METRICS" ]; then
    EPOCH=$(echo $METRICS | cut -d',' -f1)
    PSNR=$(echo $METRICS | cut -d',' -f4)
    SSIM=$(echo $METRICS | cut -d',' -f5)
    log_info "   Epoch: $EPOCH"
    log_info "   PSNR: ${PSNR:-N/A} dB"
    log_info "   SSIM: ${SSIM:-N/A}"
else
    log_warn "   无法获取metrics"
fi
echo ""

# 4. 检查Git分支和状态
log_info "4. 检查Git状态..."
BRANCH=$(ssh_run "cd $REPO_PATH && git branch --show-current")
COMMIT=$(ssh_run "cd $REPO_PATH && git rev-parse --short HEAD")
log_info "   分支: $BRANCH"
log_info "   Commit: $COMMIT"

# 检查是否有未提交修改
CHANGES=$(ssh_run "cd $REPO_PATH && git status --porcelain | wc -l")
if [ "$CHANGES" -gt 0 ]; then
    log_warn "   有 $CHANGES 个未提交修改"
    ssh_run "cd $REPO_PATH && git status --short" | while read line; do
        echo "      $line"
    done
else
    log_info "   工作区干净"
fi
echo ""

# 5. 检查使用的Loss
log_info "5. 检查Loss配置..."
LOSS_TYPE=$(ssh_run "grep 'train_criterion = ' $REPO_PATH/code/trainxuanwu.py | tail -1")
if echo "$LOSS_TYPE" | grep -q "EAGLELoss"; then
    log_info "   Loss: EAGLELoss ✅"
elif echo "$LOSS_TYPE" | grep -q "L1Loss"; then
    log_info "   Loss: L1Loss"
else
    log_warn "   Loss: 未知 ($LOSS_TYPE)"
fi
echo ""

# 6. 检查最新日志
log_info "6. 最新训练日志 (最后3行)..."
echo "----------------------------------------------"
ssh_run "tail -3 $REPO_PATH/train_daemon.log 2>/dev/null" | while read line; do
    echo "   $line"
done
echo "----------------------------------------------"
echo ""

echo "=============================================="
echo "  检查完成"
echo "=============================================="
