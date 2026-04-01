#!/bin/bash
#===============================================================================
# 监控守护进程 - 后台持续监控训练状态
# 用法: ./monitor_daemon.sh [start|stop|status]
#===============================================================================

PID_FILE="/tmp/rplhr_monitor.pid"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-start}" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "监控已在运行 (PID: $(cat $PID_FILE))"
            exit 0
        fi
        
        echo "启动监控守护进程..."
        nohup "$SCRIPT_DIR/monitor_training.sh" 8 > /dev/null 2>&1 &
        echo $! > "$PID_FILE"
        echo "监控已启动 (PID: $!)"
        echo "日志: $SCRIPT_DIR/logs/monitor/"
        ;;
    
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                rm -f "$PID_FILE"
                echo "监控已停止"
            else
                echo "监控未运行"
                rm -f "$PID_FILE"
            fi
        else
            echo "监控未运行"
        fi
        ;;
    
    status)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "监控运行中 (PID: $(cat $PID_FILE))"
            echo "最新日志:"
            tail -5 "$SCRIPT_DIR/logs/monitor/monitor_$(date +%Y%m%d).log" 2>/dev/null || echo "暂无日志"
        else
            echo "监控未运行"
        fi
        ;;
    
    log)
        LOG_FILE="$SCRIPT_DIR/logs/monitor/monitor_$(date +%Y%m%d).log"
        if [ -f "$LOG_FILE" ]; then
            tail -30 "$LOG_FILE"
        else
            echo "暂无日志文件"
        fi
        ;;
    
    *)
        echo "用法: $0 {start|stop|status|log}"
        exit 1
        ;;
esac
