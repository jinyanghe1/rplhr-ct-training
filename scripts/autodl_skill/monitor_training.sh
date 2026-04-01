#!/bin/bash
#===============================================================================
# 训练监控脚本 - 每3分钟检查一次训练状态
# 用法: ./monitor_training.sh [持续时间(小时)]
# 示例: ./monitor_training.sh 2  (监控2小时)
#===============================================================================

# SSH配置
HOST="connect.westd.seetacloud.com"
PORT="23086"
USER="root"
REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"

# 监控配置
CHECK_INTERVAL=180  # 3分钟 = 180秒
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs/monitor"
mkdir -p "$LOG_DIR"

# 持续时间（默认8小时）
DURATION_HOURS="${1:-8}"
DURATION_SECONDS=$((DURATION_HOURS * 3600))

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_highlight() { echo -e "${BLUE}$1${NC}"; }

# SSH执行命令
ssh_run() {
    ssh -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -p $PORT $USER@$HOST "$1" 2>/dev/null
}

# 获取当前时间
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# 获取训练状态
get_training_status() {
    local log_file="$LOG_DIR/monitor_$(date +%Y%m%d).log"
    local timestamp=$(get_timestamp)
    
    echo "" >> "$log_file"
    echo "========================================" >> "$log_file"
    echo "[$timestamp] 训练状态检查" >> "$log_file"
    echo "========================================" >> "$log_file"
    
    # 1. 检查训练进程
    local pid=$(ssh_run "ps aux | grep '[t]rainxuanwu' | awk '{print \$2}' | head -1")
    if [ -n "$pid" ]; then
        echo "✅ 训练进程运行中 (PID: $pid)" >> "$log_file"
        
        # 获取运行时间
        local elapsed=$(ssh_run "ps -p $pid -o etime= 2>/dev/null | tr -d ' '" || echo "N/A")
        echo "   运行时间: $elapsed" >> "$log_file"
        
        # 获取GPU使用
        local gpu_mem=$(ssh_run "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep $pid | cut -d',' -f2 | tr -d ' '" || echo "N/A")
        if [ "$gpu_mem" != "N/A" ] && [ -n "$gpu_mem" ]; then
            echo "   GPU显存: ${gpu_mem}MB" >> "$log_file"
        fi
    else
        echo "❌ 训练进程未运行" >> "$log_file"
    fi
    
    # 2. 获取当前Epoch
    local current_epoch=$(ssh_run "grep -E 'Epoch [0-9]+' $REPO_PATH/train_daemon_new.log 2>/dev/null | tail -1 | grep -oE 'Epoch [0-9]+' | awk '{print \$2}'")
    if [ -n "$current_epoch" ]; then
        echo "📊 当前Epoch: $current_epoch" >> "$log_file"
    fi
    
    # 3. 获取最新metrics
    local metrics=$(ssh_run "tail -1 $REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv 2>/dev/null")
    if [ -n "$metrics" ] && [[ "$metrics" != *"epoch"* ]]; then
        local epoch=$(echo "$metrics" | cut -d',' -f1)
        local loss=$(echo "$metrics" | cut -d',' -f3)
        local psnr=$(echo "$metrics" | cut -d',' -f4)
        local ssim=$(echo "$metrics" | cut -d',' -f5)
        
        echo "📈 最新Metrics (Epoch $epoch):" >> "$log_file"
        [ -n "$loss" ] && echo "   Loss: $loss" >> "$log_file"
        [ -n "$psnr" ] && echo "   PSNR: ${psnr}dB" >> "$log_file"
        [ -n "$ssim" ] && echo "   SSIM: $ssim" >> "$log_file"
    fi
    
    # 4. 获取最后几行日志
    echo "📝 最近日志:" >> "$log_file"
    ssh_run "tail -3 $REPO_PATH/train_daemon_new.log 2>/dev/null" >> "$log_file"
    
    echo "[$timestamp] 检查完成" >> "$log_file"
    echo "$log_file"
}

# 打印头部信息
print_header() {
    clear
    log_highlight "========================================"
    log_highlight "  训练状态监控"
    log_highlight "========================================"
    echo ""
    log_info "监控间隔: 3分钟"
    log_info "持续时间: $DURATION_HOURS 小时"
    log_info "日志目录: $LOG_DIR"
    echo ""
    log_highlight "========================================"
    echo ""
}

# 主监控循环
main() {
    print_header
    
    local start_time=$(date +%s)
    local end_time=$((start_time + DURATION_SECONDS))
    local check_count=0
    
    log_info "监控开始，按 Ctrl+C 停止"
    echo ""
    
    while [ $(date +%s) -lt $end_time ]; do
        check_count=$((check_count + 1))
        local current_time=$(get_timestamp)
        local elapsed=$(( ($(date +%s) - start_time) / 60 ))
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[$current_time] 第 $check_count 次检查 (已运行 ${elapsed}分钟)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 获取状态并显示
        local log_file=$(get_training_status)
        cat "$log_file" | tail -20
        
        # 计算下次检查时间
        local next_check=$(date -d "+3 minutes" '+%H:%M:%S' 2>/dev/null || echo "3分钟后")
        echo ""
        echo "⏰ 下次检查: $next_check"
        echo ""
        
        # 等待3分钟
        sleep $CHECK_INTERVAL
    done
    
    echo ""
    log_info "监控结束，共检查 $check_count 次"
    log_info "完整日志: $LOG_DIR/monitor_$(date +%Y%m%d).log"
}

# 捕获Ctrl+C
trap 'echo ""; log_info "监控已停止"; exit 0' INT

main "$@"
