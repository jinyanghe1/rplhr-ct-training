#!/bin/bash
#===============================================================================
# AutoDL 训练守护进程 - 保证长时间训练不中断
# 使用 screen 保持训练在后台运行
#===============================================================================

# 核心配置
HOST="connect.westd.seetacloud.com"
PORT="23086"
USER="root"
REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"
CODE_PATH="$REPO_PATH/code"
DATASET_KEY="dataset01_xuanwu"
NET_IDX="xuanwu_ratio4"
CONFIG_PATH="../config/xuanwu_ratio4.txt"
EPOCHS=100
SCREEN_NAME="rplhr_training"
TRAIN_LOG="$REPO_PATH/train_daemon.log"

# 训练参数
USE_AUG="--use_augmentation=True --aug_prob=0.5"
NORM="--normalize_ct=True --clip_ct=True --min_hu=-1024 --max_hu=3071"
TTA="--use_tta=True"
WORKERS="--num_workers=4 --test_num_workers=2"

# 目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/daemon_${TIMESTAMP}.log"

#===============================================================================
# 日志函数
#===============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# SSH 执行
ssh_run() {
    local cmd="$1"
    ssh -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -o ConnectTimeout=15 \
        -p $PORT $USER@$HOST "$cmd" 2>&1
}

#===============================================================================
# 检查训练是否在运行
#===============================================================================
is_training_running() {
    local count=$(ssh_run "screen -ls | grep -c '$SCREEN_NAME'" 2>/dev/null || echo "0")
    [[ "$count" != "0" ]]
}

#===============================================================================
# 获取最新 PSNR
#===============================================================================
get_latest_psnr() {
    ssh_run "grep 'psnr_val' $REPO_PATH/checkpoints/dataset01_xuanwu/$NET_IDX/metrics.csv 2>/dev/null | tail -1 | awk -F',' '{print \$4}'"
}

#===============================================================================
# 启动训练
#===============================================================================
start_training() {
    log "启动训练 (epoch=$EPOCHS, screen=$SCREEN_NAME)..."

    local train_cmd="cd $CODE_PATH && source /etc/network_turbo && source /root/miniconda3/bin/activate base && "
    train_cmd+="python trainxuanwu.py train "
    train_cmd+="--net_idx=$NET_IDX "
    train_cmd+="--path_key=$DATASET_KEY "
    train_cmd+="--config=$CONFIG_PATH "
    train_cmd+="--epoch=$EPOCHS "
    train_cmd+="$USE_AUG $NORM $TTA $WORKERS "
    train_cmd+="> $TRAIN_LOG 2>&1"

    # 使用 screen 启动持久训练
    ssh_run "screen -dmS $SCREEN_NAME bash -c '$train_cmd; exec bash'"

    sleep 5

    if is_training_running; then
        log "训练已在 screen 中启动: $SCREEN_NAME"
        return 0
    else
        log "[错误] 训练启动失败"
        return 1
    fi
}

#===============================================================================
# 停止训练
#===============================================================================
stop_training() {
    log "停止训练..."
    ssh_run "screen -S $SCREEN_NAME -X quit 2>/dev/null"
}

#===============================================================================
# 查看训练输出
#===============================================================================
view_output() {
    ssh_run "tail -30 $TRAIN_LOG"
}

#===============================================================================
# 主循环
#===============================================================================
main() {
    log "=========================================="
    log "AutoDL 训练守护进程启动"
    log "Screen: $SCREEN_NAME"
    log "Log: $MAIN_LOG"
    log "=========================================="

    local check_interval=60
    local max_epochs_without_improvement=20
    local last_best_psnr=0
    local epochs_without_improvement=0

    while true; do
        if ! is_training_running; then
            log "检测到训练未运行，启动新训练..."
            start_training
        else
            log "[检测] 训练运行中..."

            # 获取当前最佳 PSNR
            local current_psnr=$(get_latest_psnr)
            if [[ -n "$current_psnr" && "$current_psnr" != "" ]]; then
                log "[PSNR] 当前: $current_psnr dB (最佳: $last_best_psnr)"

                # 检查是否有改善
                local psnr_diff=$(echo "$current_psnr - $last_best_psnr" | bc 2>/dev/null || echo "0.001")
                if (( $(echo "$psnr_diff > 0.01" | bc -l 2>/dev/null) )); then
                    last_best_psnr=$current_psnr
                    epochs_without_improvement=0
                    log "[改善] PSNR 提升到 $current_psnr dB"
                else
                    ((epochs_without_improvement++))
                    if [[ $epochs_without_improvement -ge $max_epochs_without_improvement ]]; then
                        log "[收敛] PSNR 连续 $max_epochs_without_improvement 个 epoch 无改善，训练完成"
                        break
                    fi
                fi
            fi
        fi

        # 每分钟检查一次
        sleep $check_interval
    done

    log "守护进程结束"
    log "最终 PSNR: $last_best_psnr dB"
}

#===============================================================================
# 命令处理
#===============================================================================
case "${1:-start}" in
    start)
        main
        ;;
    stop)
        stop_training
        ;;
    status)
        if is_training_running; then
            echo "训练运行中"
            view_output
        else
            echo "训练未运行"
        fi
        ;;
    view)
        view_output
        ;;
    *)
        echo "用法: $0 {start|stop|status|view}"
        ;;
esac
