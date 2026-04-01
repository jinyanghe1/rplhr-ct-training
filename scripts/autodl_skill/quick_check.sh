#!/bin/bash
#===============================================================================
# 快速检查训练状态 - 单次检查
#===============================================================================

HOST="connect.westd.seetacloud.com"
PORT="23086"
USER="root"
REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_highlight() { echo -e "${BLUE}$1${NC}"; }

ssh_run() {
    ssh -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -p $PORT $USER@$HOST "$1" 2>/dev/null
}

echo ""
log_highlight "╔════════════════════════════════════════╗"
log_highlight "║       训练状态快速检查                 ║"
log_highlight "╚════════════════════════════════════════╝"
echo ""

# 1. 检查进程
PID=$(ssh_run "ps aux | grep '[t]rainxuanwu' | awk '{print \$2}' | head -1")
if [ -n "$PID" ]; then
    log_info "训练进程运行中 (PID: $PID)"
    
    # 运行时间
    ELAPSED=$(ssh_run "ps -p $PID -o etime= 2>/dev/null | tr -d ' '" || echo "N/A")
    echo "  运行时间: $ELAPSED"
    
    # GPU使用
    GPU_MEM=$(ssh_run "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep $PID | cut -d',' -f2 | tr -d ' '" || echo "")
    if [ -n "$GPU_MEM" ]; then
        echo "  GPU显存: ${GPU_MEM}MB"
    fi
else
    log_error "训练进程未运行"
fi
echo ""

# 2. 当前Epoch
CURRENT_EPOCH=$(ssh_run "grep -E 'Epoch [0-9]+' $REPO_PATH/train_daemon_new.log 2>/dev/null | tail -1 | grep -oE 'Epoch [0-9]+' | awk '{print \$2}'")
if [ -n "$CURRENT_EPOCH" ]; then
    log_info "当前训练Epoch: $CURRENT_EPOCH"
else
    log_warn "无法获取当前Epoch"
fi
echo ""

# 3. 最新Metrics
METRICS=$(ssh_run "tail -1 $REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv 2>/dev/null")
if [ -n "$METRICS" ] && [[ "$METRICS" != *"epoch"* ]]; then
    EPOCH=$(echo "$METRICS" | cut -d',' -f1)
    LOSS=$(echo "$METRICS" | cut -d',' -f3)
    PSNR=$(echo "$METRICS" | cut -d',' -f4)
    SSIM=$(echo "$METRICS" | cut -d',' -f5)
    
    log_highlight "📊 最新验证Metrics (Epoch $EPOCH)"
    [ -n "$LOSS" ] && echo "  Train Loss: $LOSS"
    [ -n "$PSNR" ] && echo "  Val PSNR:   ${PSNR}dB"
    [ -n "$SSIM" ] && echo "  Val SSIM:   $SSIM"
else
    log_warn "无法获取Metrics"
fi
echo ""

# 4. 最近日志
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_highlight "📝 最近3条日志:"
ssh_run "tail -3 $REPO_PATH/train_daemon_new.log 2>/dev/null" | while read line; do
    echo "  $line"
done
echo ""

# 5. 趋势
log_highlight "📈 近期PSNR趋势:"
TREND=$(ssh_run "grep -E '^[0-9]+,' $REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv 2>/dev/null | grep -v ',,,' | tail -5")
echo "$TREND" | while IFS=',' read -r e lr loss psnr ssim rest; do
    if [ -n "$psnr" ] && [ "$psnr" != "" ]; then
        printf "  Epoch %3s: %.2f dB\n" "$e" "$psnr"
    fi
done

echo ""
log_highlight "═══════════════════════════════════════"
echo "$(date '+%Y-%m-%d %H:%M:%S') 检查完成"
echo ""
