#!/bin/bash
#===============================================================================
# 监控守护进程 - 增强版 v2.0
# 改进: 自动告警、状态持久化、优雅停止
# 用法: ./monitor_daemon.sh [start|stop|status|log|alert-test]
#===============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/config.sh" 2>/dev/null || true
source "$SCRIPT_DIR/lib_ssh.sh"
source "$SCRIPT_DIR/lib_monitor.sh"

PID_FILE="/tmp/rplhr_monitor.pid"
STATE_FILE="/tmp/rplhr_monitor.state"
LOG_DIR="$SCRIPT_DIR/logs/monitor"
mkdir -p "$LOG_DIR"

#===============================================================================
# 状态管理
#===============================================================================

save_state() {
    local key="$1"
    local value="$2"
    echo "$key=$value" >> "$STATE_FILE"
}

load_state() {
    local key="$1"
    grep "^$key=" "$STATE_FILE" 2>/dev/null | tail -1 | cut -d'=' -f2
}

clear_state() {
    rm -f "$STATE_FILE"
}

#===============================================================================
# 告警测试
#===============================================================================

alert_test() {
    echo "测试告警系统..."
    
    # 模拟各种告警
    send_alert "TEST" "这是测试告警消息"
    send_alert "TRAINING_STOPPED" "训练进程停止测试"
    send_alert "LOSS_ANOMALY" "Loss异常测试"
    
    echo "测试完成，检查日志: $LOG_DIR/monitor_$(date +%Y%m%d).log"
}

#===============================================================================
# 守护进程主循环
#===============================================================================

daemon_loop() {
    local duration_hours="${1:-8}"
    local check_interval="${2:-180}"
    
    local start_time
    start_time=$(date +%s)
    local end_time=$((start_time + duration_hours * 3600))
    local check_count=0
    
    monitor_log "INFO" "监控守护进程启动 PID: $$"
    monitor_log "INFO" "持续 ${duration_hours}小时 间隔 ${check_interval}秒"
    
    # 保存PID
    echo $$ > "$PID_FILE"
    save_state "start_time" "$start_time"
    save_state "duration_hours" "$duration_hours"
    
    # 主循环
    while [[ -f "$PID_FILE" ]] && [[ $(date +%s) -lt $end_time ]]; do
        check_count=$((check_count + 1))
        local current_time
        current_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        monitor_log "INFO" "=== 检查 #$check_count [$current_time] ==="
        
        # 1. 检查SSH连接
        if ! ssh_check_connection; then
            monitor_log "WARN" "SSH连接失败，稍后重试"
            sleep 30
            continue
        fi
        
        # 2. 检查训练进程
        if ! is_training_running; then
            send_alert "TRAINING_STOPPED" "训练进程未运行"
            save_state "last_status" "stopped"
        else
            local pid
            pid=$(get_training_pid)
            monitor_log "INFO" "训练进程正常 (PID: $pid)"
            save_state "last_status" "running"
            save_state "last_pid" "$pid"
            
            # 3. 获取训练摘要
            local summary
            summary=$(get_training_summary)
            monitor_log "INFO" "训练摘要: $summary"
            
            # 4. 检查GPU异常
            local gpu_alerts
            gpu_alerts=$(check_gpu_anomaly)
            if [[ -n "$gpu_alerts" ]]; then
                send_alert "GPU_ANOMALY" "$gpu_alerts"
            fi
        fi
        
        # 5. 保存检查次数
        save_state "check_count" "$check_count"
        save_state "last_check" "$(date +%s)"
        
        # 等待下次检查
        sleep "$check_interval"
    done
    
    monitor_log "INFO" "监控守护进程结束 (总检查: $check_count 次)"
    
    # 清理
    rm -f "$PID_FILE"
}

#===============================================================================
# 启动守护进程
#===============================================================================

cmd_start() {
    local duration="${1:-8}"
    
    # 检查是否已在运行
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "监控守护进程已在运行 (PID: $old_pid)"
            echo "使用 '$0 status' 查看状态"
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi
    
    echo "启动监控守护进程..."
    echo "持续时间: ${duration}小时"
    echo ""
    
    # 在后台启动
    nohup "$0" daemon "$duration" > /dev/null 2>&1 &
    
    # 等待PID文件创建
    local wait_count=0
    while [[ ! -f "$PID_FILE" ]] && [[ $wait_count -lt 10 ]]; do
        sleep 0.5
        wait_count=$((wait_count + 1))
    done
    
    if [[ -f "$PID_FILE" ]]; then
        echo "✓ 监控已启动 (PID: $(cat $PID_FILE))"
        echo "日志: $LOG_DIR/monitor_$(date +%Y%m%d).log"
        echo "查看日志: tail -f $LOG_DIR/monitor_$(date +%Y%m%d).log"
    else
        echo "✗ 启动失败"
        return 1
    fi
}

#===============================================================================
# 停止守护进程
#===============================================================================

cmd_stop() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "监控守护进程未运行"
        return 0
    fi
    
    local pid
    pid=$(cat "$PID_FILE")
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "停止监控守护进程 (PID: $pid)..."
        kill "$pid"
        
        # 等待进程结束
        local wait_count=0
        while kill -0 "$pid" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
            sleep 0.5
            wait_count=$((wait_count + 1))
        done
        
        rm -f "$PID_FILE"
        echo "✓ 监控已停止"
    else
        echo "监控进程已不存在"
        rm -f "$PID_FILE"
    fi
}

#===============================================================================
# 查看状态
#===============================================================================

cmd_status() {
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "           RPLHR-CT 监控守护进程状态"
    echo "═══════════════════════════════════════════════════"
    echo ""
    
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "状态: \033[0;32m运行中\033[0m"
            echo "PID: $pid"
            
            # 显示运行时间
            local start_time elapsed
            start_time=$(load_state "start_time")
            if [[ -n "$start_time" ]]; then
                elapsed=$(($(date +%s) - start_time))
                printf "运行时间: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
            fi
            
            # 显示检查次数
            local check_count
            check_count=$(load_state "check_count")
            echo "检查次数: ${check_count:-0}"
            
            # 显示最后状态
            local last_status
            last_status=$(load_state "last_status")
            echo "最后状态: ${last_status:-unknown}"
            
        else
            echo -e "状态: \033[0;31m已停止\033[0m (PID文件残留)"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "状态: \033[0;33m未运行\033[0m"
    fi
    
    echo ""
    echo "日志位置: $LOG_DIR/"
    echo "状态文件: $STATE_FILE"
    echo ""
}

#===============================================================================
# 查看日志
#===============================================================================

cmd_log() {
    local lines="${1:-30}"
    local log_file
    log_file="$LOG_DIR/monitor_$(date +%Y%m%d).log"
    
    if [[ -f "$log_file" ]]; then
        echo "═══════════════════════════════════════════════════"
        echo "           最新 $lines 行日志"
        echo "═══════════════════════════════════════════════════"
        echo ""
        tail -"$lines" "$log_file"
    else
        echo "暂无日志文件"
    fi
}

#===============================================================================
# 入口
#===============================================================================

# 导出SSH配置
export SSH_HOST="${AUTODL_HOST:-connect.westd.seetacloud.com}"
export SSH_PORT="${AUTODL_PORT:-23086}"
export SSH_USER="${AUTODL_USER:-root}"

case "${1:-status}" in
    start)
        cmd_start "${2:-8}"
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    log)
        cmd_log "${2:-30}"
        ;;
    daemon)
        # 内部使用，直接启动守护进程循环
        daemon_loop "${2:-8}"
        ;;
    alert-test)
        alert_test
        ;;
    *)
        echo "用法: $0 {start|stop|status|log|alert-test}"
        echo ""
        echo "命令:"
        echo "  start [小时]    启动监控守护进程 (默认8小时)"
        echo "  stop           停止监控守护进程"
        echo "  status         查看监控状态"
        echo "  log [行数]     查看最新日志 (默认30行)"
        echo "  alert-test     测试告警系统"
        echo ""
        exit 1
        ;;
esac
