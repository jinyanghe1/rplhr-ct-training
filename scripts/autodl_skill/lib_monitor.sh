#!/bin/bash
#===============================================================================
# 训练监控工具库 - 提供训练状态监控和异常检测功能
# 用法: source $0
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_ssh.sh" 2>/dev/null || true

# 监控配置
MONITOR_LOG_DIR="${SCRIPT_DIR}/logs/monitor"
mkdir -p "$MONITOR_LOG_DIR"

# 异常检测阈值
LOSS_EXPLOSION_THRESHOLD=1000.0    # Loss爆炸阈值
LOSS_NAN_THRESHOLD="nan\|inf\|NaN" # NaN/Inf检测
PSNR_DROP_THRESHOLD=5.0            # PSNR下降阈值(dB)
SSIM_DROP_THRESHOLD=0.1            # SSIM下降阈值
GPU_MEM_THRESHOLD=95               # GPU内存使用阈值(%)

# 告警计数器（用于避免重复告警）
declare -A ALERT_COUNT
declare -A LAST_ALERT_TIME

#===============================================================================
# 日志函数
#===============================================================================
monitor_log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$MONITOR_LOG_DIR/monitor_$(date +%Y%m%d).log"
    
    echo "[$timestamp] [$level] $message" | tee -a "$log_file"
}

#===============================================================================
# 获取训练指标
#===============================================================================
get_training_metrics() {
    local repo_path="${1:-/root/autodl-tmp/rplhr-ct-training-main}"
    local checkpoint_path="$repo_path/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv"
    
    local metrics
    metrics=$(ssh_run_with_retry "cat $checkpoint_path 2>/dev/null" 5 2)
    
    if [[ -n "$metrics" ]]; then
        # 返回最后5行用于趋势分析
        echo "$metrics" | tail -5
    fi
}

#===============================================================================
# 从日志提取指标
#===============================================================================
extract_metrics_from_log() {
    local log_content="$1"
    local metric_type="$2"  # psnr, ssim, loss
    
    case "$metric_type" in
        psnr)
            echo "$log_content" | grep -iE "PSNR|psnr" | tail -5
            ;;
        ssim)
            echo "$log_content" | grep -iE "SSIM|ssim" | tail -5
            ;;
        loss)
            echo "$log_content" | grep -iE "Loss|loss|LOSS" | tail -10
            ;;
        epoch)
            echo "$log_content" | grep -iE "Epoch [0-9]+" | tail -3
            ;;
    esac
}

#===============================================================================
# 检查Loss异常（爆炸或NaN）
#===============================================================================
check_loss_anomaly() {
    local log_file="$1"
    local alerts=""
    
    if [[ ! -f "$log_file" ]]; then
        return 0
    fi
    
    # 检查NaN/Inf
    if grep -qiE "$LOSS_NAN_THRESHOLD" "$log_file"; then
        local nan_line
        nan_line=$(grep -niE "$LOSS_NAN_THRESHOLD" "$log_file" | tail -1)
        alerts="${alerts}[CRITICAL] 检测到NaN/Inf: $nan_line\n"
    fi
    
    # 检查Loss爆炸
    local last_loss
    last_loss=$(grep -iE "Loss" "$log_file" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
    if [[ -n "$last_loss" ]]; then
        if (( $(echo "$last_loss > $LOSS_EXPLOSION_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
            alerts="${alerts}[WARNING] Loss爆炸: $last_loss > $LOSS_EXPLOSION_THRESHOLD\n"
        fi
    fi
    
    echo -e "$alerts"
}

#===============================================================================
# 检查性能退化
#===============================================================================
check_performance_degradation() {
    local metrics_csv="$1"
    local alerts=""
    
    if [[ ! -f "$metrics_csv" ]] || [[ $(wc -l < "$metrics_csv") -lt 3 ]]; then
        return 0
    fi
    
    # 提取PSNR值
    local psnr_values
    psnr_values=$(tail -n +2 "$metrics_csv" | cut -d',' -f4 | grep -v '^$')
    
    if [[ $(echo "$psnr_values" | wc -l) -ge 2 ]]; then
        local first_psnr
        local last_psnr
        first_psnr=$(echo "$psnr_values" | head -1)
        last_psnr=$(echo "$psnr_values" | tail -1)
        
        if [[ -n "$first_psnr" && -n "$last_psnr" ]]; then
            local drop
            drop=$(echo "$first_psnr - $last_psnr" | bc -l 2>/dev/null || echo "0")
            if (( $(echo "$drop > $PSNR_DROP_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
                alerts="${alerts}[WARNING] PSNR显著下降: ${drop}dB\n"
            fi
        fi
    fi
    
    echo -e "$alerts"
}

#===============================================================================
# 检查GPU异常
#===============================================================================
check_gpu_anomaly() {
    local alerts=""
    
    local gpu_info
    gpu_info=$(ssh_run_with_retry "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null" 5 1)
    
    if [[ -z "$gpu_info" ]]; then
        alerts="[WARNING] 无法获取GPU信息\n"
    else
        # 解析GPU信息
        local temp util mem_used mem_total
        temp=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
        util=$(echo "$gpu_info" | cut -d',' -f2 | tr -d ' ')
        mem_used=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
        mem_total=$(echo "$gpu_info" | cut -d',' -f4 | tr -d ' ')
        
        # 检查温度
        if [[ -n "$temp" && "$temp" -gt 85 ]]; then
            alerts="${alerts}[WARNING] GPU温度过高: ${temp}°C\n"
        fi
        
        # 检查内存使用
        if [[ -n "$mem_used" && -n "$mem_total" && "$mem_total" -gt 0 ]]; then
            local mem_pct
            mem_pct=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")
            if [[ "$mem_pct" -gt $GPU_MEM_THRESHOLD ]]; then
                alerts="${alerts}[WARNING] GPU内存使用过高: ${mem_pct}%\n"
            fi
        fi
        
        # 检查利用率（如果为0可能是卡住）
        if [[ -n "$util" && "$util" -eq 0 ]]; then
            # 只有在训练运行时才警告
            if is_training_running; then
                alerts="${alerts}[NOTICE] GPU利用率为0，可能训练卡住\n"
            fi
        fi
    fi
    
    echo -e "$alerts"
}

#===============================================================================
# 发送告警（去重）
#===============================================================================
send_alert() {
    local alert_type="$1"
    local message="$2"
    local current_time
    current_time=$(date +%s)
    local alert_key="${alert_type}_$(echo "$message" | md5sum | cut -d' ' -f1)"
    
    # 检查是否重复告警（5分钟内不重复）
    if [[ -n "${LAST_ALERT_TIME[$alert_key]}" ]]; then
        local last_time="${LAST_ALERT_TIME[$alert_key]}"
        if [[ $((current_time - last_time)) -lt 300 ]]; then
            return 0
        fi
    fi
    
    # 更新告警时间和计数
    LAST_ALERT_TIME[$alert_key]=$current_time
    ALERT_COUNT[$alert_key]=$((${ALERT_COUNT[$alert_key]:-0} + 1))
    
    # 记录告警
    monitor_log "ALERT" "[$alert_type] $message"
    
    # 输出到stderr以便捕获
    echo -e "\033[0;31m[ALERT][$alert_type]\033[0m $message" >&2
    
    # 可以在这里添加其他通知方式（邮件、Slack等）
}

#===============================================================================
# 获取训练进度摘要
#===============================================================================
get_training_summary() {
    local repo_path="${1:-/root/autodl-tmp/rplhr-ct-training-main}"
    local summary=""
    
    # 检查进程
    local pid
    pid=$(get_training_pid)
    if [[ -n "$pid" ]]; then
        summary="进程: 运行中 (PID: $pid)\n"
        local uptime
        uptime=$(get_training_uptime "$pid")
        summary="${summary}运行时间: $uptime\n"
    else
        summary="进程: 未运行\n"
    fi
    
    # 获取最新指标
    local metrics
    metrics=$(get_training_metrics "$repo_path")
    if [[ -n "$metrics" ]]; then
        local latest
        latest=$(echo "$metrics" | tail -1)
        local epoch psnr ssim loss
        epoch=$(echo "$latest" | cut -d',' -f1)
        loss=$(echo "$latest" | cut -d',' -f3)
        psnr=$(echo "$latest" | cut -d',' -f4)
        ssim=$(echo "$latest" | cut -d',' -f5)
        
        summary="${summary}Epoch: $epoch | Loss: $loss | PSNR: $psnr | SSIM: $ssim\n"
    fi
    
    echo -e "$summary"
}

#===============================================================================
# 实时监控循环
#===============================================================================
monitor_loop() {
    local duration_minutes="${1:-60}"
    local interval_sec="${2:-180}"
    local repo_path="${3:-/root/autodl-tmp/rplhr-ct-training-main}"
    
    local start_time
    start_time=$(date +%s)
    local end_time=$((start_time + duration_minutes * 60))
    local check_count=0
    
    monitor_log "INFO" "监控启动: 持续${duration_minutes}分钟, 间隔${interval_sec}秒"
    
    while [[ $(date +%s) -lt $end_time ]]; do
        check_count=$((check_count + 1))
        local current_time
        current_time=$(date '+%H:%M:%S')
        
        monitor_log "INFO" "--- 检查 #$check_count [$current_time] ---"
        
        # 获取本地日志路径
        local local_log
        local_log=$(ls -t "$MONITOR_LOG_DIR"/*.log 2>/dev/null | head -1)
        
        # 1. 检查训练状态
        if ! is_training_running; then
            send_alert "TRAINING_STOPPED" "训练进程未运行"
        else
            monitor_log "INFO" "训练进程正常运行"
        fi
        
        # 2. 检查Loss异常
        if [[ -n "$local_log" ]]; then
            local loss_alerts
            loss_alerts=$(check_loss_anomaly "$local_log")
            if [[ -n "$loss_alerts" ]]; then
                send_alert "LOSS_ANOMALY" "$loss_alerts"
            fi
        fi
        
        # 3. 检查GPU异常
        local gpu_alerts
        gpu_alerts=$(check_gpu_anomaly)
        if [[ -n "$gpu_alerts" ]]; then
            send_alert "GPU_ANOMALY" "$gpu_alerts"
        fi
        
        # 4. 输出摘要
        local summary
        summary=$(get_training_summary "$repo_path")
        monitor_log "INFO" "摘要: $summary"
        
        # 等待下一次检查
        sleep $interval_sec
    done
    
    monitor_log "INFO" "监控结束: 共检查$check_count次"
}

#===============================================================================
# 生成监控报告
#===============================================================================
generate_monitor_report() {
    local output_file="${1:-$MONITOR_LOG_DIR/report_$(date +%Y%m%d_%H%M%S).md}"
    
    {
        echo "# 训练监控报告"
        echo ""
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "## 训练状态摘要"
        echo ""
        echo "\`\`\`"
        get_training_summary
        echo "\`\`\`"
        echo ""
        echo "## 告警记录"
        echo ""
        grep "ALERT" "$MONITOR_LOG_DIR/monitor_$(date +%Y%m%d).log" 2>/dev/null | tail -20 | sed 's/^/- /'
        echo ""
        echo "## 完整日志"
        echo ""
        echo "查看: \`tail -100 $MONITOR_LOG_DIR/monitor_$(date +%Y%m%d).log\`"
    } > "$output_file"
    
    echo "$output_file"
}
