#!/bin/bash
#===============================================================================
# 训练状态检查脚本 - 增强版 v2.0
# 改进: 详细监控指标、异常检测、趋势分析
# 用法: ./check_training.sh [--watch] [--interval N]
#===============================================================================

set -o pipefail

# 加载工具库
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh" 2>/dev/null || true
source "$SCRIPT_DIR/lib_ssh.sh"
source "$SCRIPT_DIR/lib_monitor.sh"

# 配置
HOST="${AUTODL_HOST:-connect.westd.seetacloud.com}"
PORT="${AUTODL_PORT:-23086}"
USER="${AUTODL_USER:-root}"
REPO_PATH="${AUTODL_REPO_PATH:-/root/autodl-tmp/rplhr-ct-training-main}"

# 导出SSH配置
export SSH_HOST="$HOST"
export SSH_PORT="$PORT"
export SSH_USER="$USER"

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_section() { echo -e "${CYAN}$1${NC}"; }

#===============================================================================
# 检查训练进程详情
#===============================================================================

check_process_details() {
    local pid
    pid=$(get_training_pid)
    
    if [[ -z "$pid" ]]; then
        log_error "训练进程未运行"
        return 1
    fi
    
    log_info "训练进程运行中 (PID: $pid)"
    
    # 获取进程详情
    local process_info
    process_info=$(ssh_run_with_retry "ps -p $pid -o pid,ppid,cmd,etime,%cpu,%mem 2>/dev/null" 10 1)
    
    if [[ -n "$process_info" ]]; then
        echo ""
        echo "进程详情:"
        echo "$process_info" | sed 's/^/  /'
    fi
    
    # 获取GPU使用情况
    echo ""
    local gpu_info
    gpu_info=$(get_gpu_usage "$pid")
    
    if [[ -n "$gpu_info" && "$gpu_info" != "N/A" ]]; then
        log_info "GPU使用情况:"
        echo "$gpu_info" | sed 's/^/  /'
    else
        log_warn "无法获取GPU使用情况"
    fi
    
    return 0
}

#===============================================================================
# 检查训练进度
#===============================================================================

check_progress() {
    echo ""
    log_section "📊 训练进度"
    echo "─────────────────────────────────────────────"
    
    # 1. 从日志获取当前epoch
    local current_epoch
    current_epoch=$(ssh_run_with_retry "grep -E 'Epoch [0-9]+' $REPO_PATH/train_autodl_*.log 2>/dev/null | tail -1 | grep -oE 'Epoch [0-9]+' | awk '{print \$2}'" 10 1)
    
    if [[ -n "$current_epoch" ]]; then
        log_info "当前Epoch: $current_epoch"
    else
        log_warn "无法获取当前Epoch"
    fi
    
    # 2. 从metrics.csv获取详细信息
    local metrics_path="$REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv"
    local metrics
    metrics=$(ssh_run_with_retry "cat $metrics_path 2>/dev/null" 10 1)
    
    if [[ -n "$metrics" ]]; then
        local total_epochs
        total_epochs=$(echo "$metrics" | wc -l)
        total_epochs=$((total_epochs - 1))  # 去掉表头
        
        echo ""
        echo "已训练: $total_epochs epochs"
        
        # 显示最近5个epoch的指标
        echo ""
        echo "最近训练记录:"
        echo "  Epoch  |  Loss   |  PSNR   |  SSIM  "
        echo "  ───────┼─────────┼─────────┼────────"
        
        echo "$metrics" | tail -6 | while IFS=',' read -r epoch loss psnr ssim rest; do
            [[ "$epoch" == "epoch" ]] && continue
            [[ -z "$epoch" ]] && continue
            printf "  %6s | %7s | %7s | %6s\n" "$epoch" "${loss:-N/A}" "${psnr:-N/A}" "${ssim:-N/A}"
        done
        
        # 统计信息
        echo ""
        echo "统计摘要:"
        
        local psnr_values best_psnr worst_psnr avg_psnr
        psnr_values=$(echo "$metrics" | tail -n +2 | cut -d',' -f4 | grep -v '^$')
        
        if [[ -n "$psnr_values" ]]; then
            best_psnr=$(echo "$psnr_values" | sort -n | tail -1)
            worst_psnr=$(echo "$psnr_values" | sort -n | head -1)
            avg_psnr=$(echo "$psnr_values" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count}')
            
            echo "  PSNR 最佳: ${best_psnr} dB"
            echo "  PSNR 最差: ${worst_psnr} dB"
            echo "  PSNR 平均: ${avg_psnr} dB"
        fi
        
        # 趋势分析
        if [[ $(echo "$psnr_values" | wc -l) -ge 2 ]]; then
            local first_psnr last_psnr trend
            first_psnr=$(echo "$psnr_values" | head -1)
            last_psnr=$(echo "$psnr_values" | tail -1)
            trend=$(echo "scale=2; $last_psnr - $first_psnr" | bc 2>/dev/null || echo "0")
            
            echo ""
            if (( $(echo "$trend > 0" | bc -l 2>/dev/null || echo "0") )); then
                log_info "趋势: PSNR ↑ +$trend dB"
            elif (( $(echo "$trend < 0" | bc -l 2>/dev/null || echo "0") )); then
                log_warn "趋势: PSNR ↓ $trend dB"
            else
                echo "趋势: PSNR → 持平"
            fi
        fi
    else
        log_warn "无法获取metrics数据"
    fi
}

#===============================================================================
# 检查系统资源
#===============================================================================

check_system_resources() {
    echo ""
    log_section "💻 系统资源"
    echo "─────────────────────────────────────────────"
    
    # 1. GPU状态
    local gpu_status
    gpu_status=$(ssh_run_with_retry "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null" 10 1)
    
    if [[ -n "$gpu_status" ]]; then
        echo "GPU状态:"
        echo "  索引 | 名称 | 温度 | 利用率 | 显存 | 功耗"
        echo "$gpu_status" | while IFS=',' read -r idx name temp util mem_used mem_total power; do
            printf "  %4s | %s | %s | %s | %s/%s | %s\n" \
                "$(echo "$idx" | tr -d ' ')" \
                "$(echo "$name" | cut -c1-15)" \
                "$(echo "$temp" | tr -d ' ')°C" \
                "$(echo "$util" | tr -d ' ')" \
                "$(echo "$mem_used" | tr -d ' ')" \
                "$(echo "$mem_total" | tr -d ' ')" \
                "$(echo "$power" | tr -d ' ')"
        done
    else
        log_warn "无法获取GPU状态"
    fi
    
    # 2. 磁盘空间
    echo ""
    local disk_usage
    disk_usage=$(ssh_run_with_retry "df -h /root | tail -1" 10 1)
    
    if [[ -n "$disk_usage" ]]; then
        local usage_pct
        usage_pct=$(echo "$disk_usage" | awk '{print $5}' | tr -d '%')
        
        echo "磁盘使用:"
        if [[ "$usage_pct" -gt 90 ]]; then
            log_warn "  使用率: ${usage_pct}% (空间不足!)"
        else
            log_info "  使用率: ${usage_pct}%"
        fi
        echo "$disk_usage" | awk '{printf "  总计: %s, 已用: %s, 可用: %s\n", $2, $3, $4}' | sed 's/^/  /'
    fi
}

#===============================================================================
# 检查训练异常
#===============================================================================

check_anomalies() {
    echo ""
    log_section "🔍 异常检测"
    echo "─────────────────────────────────────────────"
    
    local has_anomaly=false
    
    # 1. 检查最新日志
    local latest_log
    latest_log=$(ssh_run_with_retry "ls -t $REPO_PATH/train_autodl_*.log 2>/dev/null | head -1" 10 1)
    
    if [[ -n "$latest_log" ]]; then
        # 下载最后100行进行检查
        local log_tail
        log_tail=$(ssh_run_with_retry "tail -100 '$latest_log'" 10 1)
        
        # 检查NaN/Inf
        if echo "$log_tail" | grep -qiE "nan|inf|NaN"; then
            log_error "检测到 NaN/Inf!"
            echo "$log_tail" | grep -iE "nan|inf|NaN" | tail -3 | sed 's/^/  /'
            has_anomaly=true
        fi
        
        # 检查Error
        local errors
        errors=$(echo "$log_tail" | grep -iE "error|exception|traceback" | head -5)
        if [[ -n "$errors" ]]; then
            log_warn "检测到错误或异常:"
            echo "$errors" | sed 's/^/  /'
            has_anomaly=true
        fi
        
        # 检查Loss爆炸
        local last_loss
        last_loss=$(echo "$log_tail" | grep -iE "loss" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
        if [[ -n "$last_loss" ]]; then
            if (( $(echo "$last_loss > 1000" | bc -l 2>/dev/null || echo "0") )); then
                log_warn "Loss 可能过高: $last_loss"
                has_anomaly=true
            fi
        fi
    fi
    
    # 2. 检查GPU异常
    local gpu_alerts
    gpu_alerts=$(check_gpu_anomaly)
    if [[ -n "$gpu_alerts" ]]; then
        log_warn "GPU相关警告:"
        echo "$gpu_alerts" | sed 's/^/  /'
        has_anomaly=true
    fi
    
    if [[ "$has_anomaly" == "false" ]]; then
        log_info "未检测到明显异常"
    fi
}

#===============================================================================
# 显示最新日志
#===============================================================================

show_recent_logs() {
    echo ""
    log_section "📝 最新日志 (最近5行)"
    echo "─────────────────────────────────────────────"
    
    local latest_log
    latest_log=$(ssh_run_with_retry "ls -t $REPO_PATH/train_autodl_*.log 2>/dev/null | head -1" 10 1)
    
    if [[ -n "$latest_log" ]]; then
        local logs
        logs=$(ssh_run_with_retry "tail -5 '$latest_log'" 10 1)
        if [[ -n "$logs" ]]; then
            echo "$logs" | sed 's/^/  /'
        else
            echo "  (无日志)"
        fi
    else
        echo "  (未找到日志文件)"
    fi
}

#===============================================================================
# 主检查流程
#===============================================================================

run_check() {
    clear 2>/dev/null || true
    
    echo ""
    echo "╔════════════════════════════════════════════════════╗"
    echo "║         RPLHR-CT 训练状态检查 (增强版)              ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo ""
    echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "服务器: $HOST:$PORT"
    echo ""
    
    # 1. 检查进程
    log_section "🔄 训练进程"
    echo "─────────────────────────────────────────────"
    check_process_details
    
    # 2. 检查进度
    check_progress
    
    # 3. 检查系统资源
    check_system_resources
    
    # 4. 检查异常
    check_anomalies
    
    # 5. 显示日志
    show_recent_logs
    
    echo ""
    echo "═══════════════════════════════════════════════════════"
}

#===============================================================================
# 监控模式
#===============================================================================

watch_mode() {
    local interval="${1:-60}"
    
    echo "进入监控模式，每 ${interval} 秒刷新一次"
    echo "按 Ctrl+C 退出"
    echo ""
    
    trap 'echo ""; echo "已退出监控模式"; exit 0' INT
    
    while true; do
        run_check
        echo ""
        echo "下次刷新: $(date -d "+${interval} seconds" '+%H:%M:%S' 2>/dev/null || echo "${interval}秒后")"
        sleep "$interval"
    done
}

#===============================================================================
# 入口
#===============================================================================

# 解析参数
WATCH_MODE=false
INTERVAL=60

while [[ $# -gt 0 ]]; do
    case $1 in
        --watch|-w)
            WATCH_MODE=true
            shift
            ;;
        --interval|-i)
            INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -w, --watch          监控模式 (定时刷新)"
            echo "  -i, --interval N     设置刷新间隔(秒)，默认60"
            echo "  -h, --help           显示帮助"
            echo ""
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 测试SSH连接
if ! ssh_test >/dev/null 2>&1; then
    log_error "SSH连接失败，请检查配置"
    exit 1
fi

# 执行
if [[ "$WATCH_MODE" == "true" ]]; then
    watch_mode "$INTERVAL"
else
    run_check
fi
