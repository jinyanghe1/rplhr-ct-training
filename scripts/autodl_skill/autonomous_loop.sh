#!/bin/bash
#===============================================================================
# AutoDL 自主训练循环 - 增强版 v2.0
# 改进: SSH连接稳定性、异常检测、详细监控
#===============================================================================

set -o pipefail

# 加载工具库
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib_ssh.sh"
source "$SCRIPT_DIR/lib_monitor.sh"

# 核心配置
HOST="${AUTODL_HOST:-connect.westd.seetacloud.com}"
PORT="${AUTODL_PORT:-23086}"
USER="${AUTODL_USER:-root}"
REPO_PATH="${AUTODL_REPO_PATH:-/root/autodl-tmp/rplhr-ct-training-main}"
CODE_PATH="$REPO_PATH/code"
DATASET_KEY="${DATASET_KEY:-dataset01_xuanwu}"
NET_IDX="${NET_IDX:-xuanwu_50epoch}"
EPOCHS=${EPOCHS:-30}
WAIT_CYCLE=${WAIT_CYCLE:-1800}
MAX_LOOPS=${MAX_LOOPS:-200}
SSH_TIMEOUT=15

# 导出SSH配置
export SSH_HOST="$HOST"
export SSH_PORT="$PORT"
export SSH_USER="$USER"

# 目录设置
LOG_DIR="$SCRIPT_DIR/logs/autonomous"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/autonomous_${TIMESTAMP}.log"

#===============================================================================
# 日志函数
#===============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$MAIN_LOG"
    monitor_log "INFO" "$1"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1"
    echo -e "\033[0;31m$msg\033[0m" | tee -a "$MAIN_LOG"
    monitor_log "ERROR" "$1"
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1"
    echo -e "\033[1;33m$msg\033[0m" | tee -a "$MAIN_LOG"
    monitor_log "WARN" "$1"
}

log_success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1"
    echo -e "\033[0;32m$msg\033[0m" | tee -a "$MAIN_LOG"
}

#===============================================================================
# SSH执行 - 使用库函数
#===============================================================================

ssh_exec() {
    local cmd="$1"
    local max_retries="${2:-3}"
    
    log "[SSH] 执行: ${cmd:0:80}..."
    ssh_run_with_retry "$cmd" 60 "$max_retries"
}

#===============================================================================
# 检查训练进程
#===============================================================================

check_training_status() {
    local status="stopped"
    local pid=""
    local uptime="N/A"
    local gpu_mem="N/A"
    
    pid=$(get_training_pid)
    if [[ -n "$pid" ]]; then
        status="running"
        uptime=$(get_training_uptime "$pid")
        gpu_mem=$(get_gpu_usage "$pid" | cut -d',' -f2 | tr -d ' ')
    fi
    
    echo "status:$status|pid:$pid|uptime:$uptime|gpu_mem:$gpu_mem"
}

#===============================================================================
# 获取最新日志文件路径
#===============================================================================

get_latest_log() {
    local log_pattern="$REPO_PATH/train_autodl_*.log"
    local latest_log
    
    latest_log=$(ssh_exec "ls -t $log_pattern 2>/dev/null | head -1" 2)
    
    if [[ -n "$latest_log" && "$latest_log" != *"No such"* ]]; then
        echo "$latest_log"
    else
        # 尝试默认路径
        echo "$REPO_PATH/train_autodl.log"
    fi
}

#===============================================================================
# 收集训练结果
#===============================================================================

collect_results() {
    local run_dir="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"
    
    log "开始收集训练结果到: $run_dir"
    
    # 1. 获取日志路径
    local log_path
    log_path=$(get_latest_log)
    log "目标日志: $log_path"
    
    # 2. 下载训练日志
    if [[ -n "$log_path" ]]; then
        if ssh_exec "test -f $log_path" 1 &>/dev/null; then
            log "下载训练日志..."
            ssh_exec "cat '$log_path'" 2 > "$run_dir/training.log" 2>/dev/null || {
                log_warn "下载日志失败，尝试scp方式..."
            }
        fi
    fi
    
    # 3. 下载metrics.csv
    local metrics_path="$REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4/metrics.csv"
    log "下载metrics..."
    ssh_exec "cat $metrics_path 2>/dev/null" 2 > "$run_dir/metrics.csv" 2>/dev/null || {
        log_warn "metrics.csv下载失败"
    }
    
    # 4. 提取关键指标
    local best_psnr="N/A" best_ssim="N/A" best_epoch="N/A"
    local final_psnr="N/A" final_ssim="N/A" final_loss="N/A"
    
    if [[ -f "$run_dir/metrics.csv" && -s "$run_dir/metrics.csv" ]]; then
        # 提取最佳指标
        best_psnr=$(tail -n +2 "$run_dir/metrics.csv" | cut -d',' -f4 | sort -n | tail -1)
        best_epoch=$(grep "$best_psnr" "$run_dir/metrics.csv" | head -1 | cut -d',' -f1)
        best_ssim=$(grep "$best_psnr" "$run_dir/metrics.csv" | head -1 | cut -d',' -f5)
        
        # 提取最终指标
        final_psnr=$(tail -1 "$run_dir/metrics.csv" | cut -d',' -f4)
        final_ssim=$(tail -1 "$run_dir/metrics.csv" | cut -d',' -f5)
        final_loss=$(tail -1 "$run_dir/metrics.csv" | cut -d',' -f3)
    fi
    
    # 从日志提取额外信息
    local total_epochs="N/A" training_time="N/A"
    if [[ -f "$run_dir/training.log" ]]; then
        total_epochs=$(grep -cE "Epoch [0-9]+" "$run_dir/training.log" 2>/dev/null || echo "N/A")
    fi
    
    # 5. 生成JSON报告
    cat > "$run_dir/metrics.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "epochs": "$EPOCHS",
    "total_epochs_trained": "$total_epochs",
    "best": {
        "epoch": "$best_epoch",
        "psnr": "$best_psnr",
        "ssim": "$best_ssim"
    },
    "final": {
        "psnr": "$final_psnr",
        "ssim": "$final_ssim",
        "loss": "$final_loss"
    }
}
EOF
    
    # 6. 生成Markdown报告
    cat > "$run_dir/report.md" << EOF
# 训练报告 - $(date '+%Y-%m-%d %H:%M:%S')

## 配置信息
- 目标Epochs: $EPOCHS
- 实际训练Epochs: $total_epochs
- 网络: $NET_IDX
- 数据集: $DATASET_KEY

## 最佳指标
| 指标 | 值 | Epoch |
|------|------|-------|
| PSNR | $best_psnr dB | $best_epoch |
| SSIM | $best_ssim | $best_epoch |

## 最终指标
| 指标 | 值 |
|------|------|
| PSNR | $final_psnr dB |
| SSIM | $final_ssim |
| Loss | $final_loss |

## 日志摘要
\`\`\`
$(tail -30 "$run_dir/training.log" 2>/dev/null || echo "无日志")
\`\`\`

## 文件位置
- 训练日志: $run_dir/training.log
- Metrics CSV: $run_dir/metrics.csv
- JSON报告: $run_dir/metrics.json
EOF
    
    log_success "结果收集完成"
    log "Best PSNR: $best_psnr dB (Epoch $best_epoch)"
    log "Final PSNR: $final_psnr dB"
    
    # 返回best_psnr供后续使用
    echo "$best_psnr"
}

#===============================================================================
# 启动训练
#===============================================================================

start_training() {
    log "启动新训练 (${EPOCHS} epochs)..."
    
    # 先清理可能存在的旧进程
    log "清理可能的僵尸进程..."
    ssh_exec "pkill -f 'trainxuanwu' 2>/dev/null || true" 1 &>/dev/null
    sleep 2
    
    # 构建训练命令
    local log_file="$REPO_PATH/train_autodl_$(date +%Y%m%d_%H%M%S).log"
    local TRAIN_CMD="cd $CODE_PATH && source /etc/network_turbo && nohup /root/miniconda3/bin/python trainxuanwu.py train --net_idx=$NET_IDX --path_key=$DATASET_KEY --epoch=$EPOCHS --use_augmentation=True --aug_prob=0.5 --clip_ct=True --min_hu=-1024 --max_hu=3071 --normalize_ct=True --num_workers=4 --test_num_workers=2 > $log_file 2>&1 & echo \"PID:\$!\""
    
    log "训练日志: $log_file"
    
    # 执行训练命令
    local output
    output=$(ssh_exec "$TRAIN_CMD" 3)
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "训练启动失败"
        return 1
    fi
    
    # 等待训练启动
    log "等待训练进程启动..."
    local wait_count=0
    local max_wait=10
    
    while [[ $wait_count -lt $max_wait ]]; do
        sleep 3
        if is_training_running; then
            local pid
            pid=$(get_training_pid)
            log_success "训练已启动! (PID: $pid)"
            return 0
        fi
        wait_count=$((wait_count + 1))
        log "等待中... ($wait_count/$max_wait)"
    done
    
    log_warn "训练进程未检测到，可能启动较慢"
    return 1
}

#===============================================================================
# 等待训练完成 - 带异常检测
#===============================================================================

wait_training() {
    log "等待训练完成..."
    
    local elapsed=0
    local check_interval=60
    local max_wait=18000  # 5小时
    local last_epoch=0
    local stuck_count=0
    
    while true; do
        # 检查进程状态
        if ! is_training_running; then
            log "训练进程已结束"
            return 0
        fi
        
        # 获取当前epoch
        local current_epoch
        current_epoch=$(ssh_exec "grep -E 'Epoch [0-9]+' $REPO_PATH/train_autodl_*.log 2>/dev/null | tail -1 | grep -oE 'Epoch [0-9]+' | awk '{print \$2}'" 1)
        
        if [[ -n "$current_epoch" && "$current_epoch" =~ ^[0-9]+$ ]]; then
            if [[ "$current_epoch" == "$last_epoch" ]]; then
                stuck_count=$((stuck_count + 1))
                if [[ $stuck_count -ge 10 ]]; then
                    log_warn "训练可能卡住: Epoch $current_epoch 已持续 $((stuck_count * check_interval / 60)) 分钟未变化"
                    send_alert "TRAINING_STUCK" "Epoch $current_epoch 卡住超过10分钟"
                fi
            else
                stuck_count=0
                last_epoch=$current_epoch
            fi
            
            log "[${elapsed}s] Epoch $current_epoch/${EPOCHS}"
        else
            log "[${elapsed}s] 训练进行中..."
        fi
        
        # 检查异常
        local gpu_alerts
        gpu_alerts=$(check_gpu_anomaly)
        if [[ -n "$gpu_alerts" ]]; then
            log_warn "GPU异常: $gpu_alerts"
        fi
        
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        
        if [[ $elapsed -ge $max_wait ]]; then
            log_warn "训练等待超时(${max_wait}s)，继续收集结果"
            return 1
        fi
    done
}

#===============================================================================
# 主循环
#===============================================================================

main() {
    log "=========================================="
    log "AutoDL 自主训练循环启动 (增强版 v2.0)"
    log "SSH 主机: $HOST:$PORT"
    log "每次循环: ~${WAIT_CYCLE}s"
    log "最多循环: ${MAX_LOOPS}次"
    log "=========================================="
    
    # 测试SSH连接
    if ! ssh_test; then
        log_error "SSH连接测试失败，退出"
        exit 1
    fi
    
    local loop=1
    local consecutive_failures=0
    local max_failures=3
    
    while [[ $loop -le $MAX_LOOPS ]]; do
        log ""
        log "=== 循环 #$loop/${MAX_LOOPS} ==="
        
        # 检测训练状态
        local status_info
        status_info=$(check_training_status)
        log "当前状态: $status_info"
        
        if is_training_running; then
            log "检测到训练进行中，等待完成..."
            if wait_training; then
                collect_results
                consecutive_failures=0
            else
                log_warn "训练等待异常"
                consecutive_failures=$((consecutive_failures + 1))
            fi
        else
            log "无训练运行，启动新训练..."
            if start_training; then
                if wait_training; then
                    collect_results
                    consecutive_failures=0
                else
                    consecutive_failures=$((consecutive_failures + 1))
                fi
            else
                log_error "训练启动失败"
                consecutive_failures=$((consecutive_failures + 1))
            fi
        fi
        
        # 检查连续失败次数
        if [[ $consecutive_failures -ge $max_failures ]]; then
            log_error "连续失败 $consecutive_failures 次，停止循环"
            send_alert "LOOP_STOPPED" "连续失败 $consecutive_failures 次，已停止"
            break
        fi
        
        log "=== 循环 #$loop 完成，等待 ${WAIT_CYCLE}s 后继续... ==="
        sleep $WAIT_CYCLE
        
        loop=$((loop + 1))
    done
    
    log "达到最大循环次数 ${MAX_LOOPS}，训练循环结束"
    
    # 生成最终报告
    local report_file
    report_file=$(generate_monitor_report "$LOG_DIR/final_report_$(date +%Y%m%d_%H%M%S).md")
    log "最终报告: $report_file"
}

#===============================================================================
# 启动
#===============================================================================

# 确保stdin不是终端
if [[ -t 0 ]]; then
    exec </dev/null
fi

# 后台运行
main &
PID=$!
echo "后台进程 PID: $PID"
echo "日志: $MAIN_LOG"
echo "停止命令: kill $PID"
log "主进程 PID: $PID"
