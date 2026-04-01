#!/bin/bash
#===============================================================================
# AutoDL 自主训练循环 - 完全自主版本 (永不阻塞)
#===============================================================================

# 核心配置
HOST="connect.westd.seetacloud.com"
PORT="23086"
USER="root"
REPO_PATH="/root/autodl-tmp/rplhr-ct-training-main"
CODE_PATH="$REPO_PATH/code"
DATASET_KEY="dataset01_xuanwu"
NET_IDX="xuanwu_50epoch"
EPOCHS=30
WAIT_CYCLE=1800
MAX_LOOPS=200                    # 一晚上约够用
SSH_TIMEOUT=15                   # SSH超时秒数

# 目录设置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/autonomous_${TIMESTAMP}.log"

#===============================================================================
# 核心函数 - 全部永不阻塞
#===============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# SSH执行 - 永不阻塞 (使用SSH内置超时)
ssh_run() {
    local cmd="$1"
    log "[SSH] 执行: ${cmd:0:60}..."
    ssh -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -o ServerAliveInterval=5 \
        -o ServerAliveCountMax=3 \
        -p $PORT $USER@$HOST "$cmd" 2>&1 || {
        echo "[SSH_ERROR] SSH命令执行失败"
        return 1
    }
}

# 带重试的SSH执行
ssh_run_retry() {
    local cmd="$1"
    local max_retries=3
    local delay=5

    for i in $(seq 1 $max_retries); do
        if ssh_run "$cmd"; then
            return 0
        fi
        log "[重试 $i/$max_retries] SSH命令执行失败，${delay}s后重试..."
        sleep $delay
    done
    log "[失败] SSH命令多次重试后仍失败"
    return 1
}

# 检查训练进程是否在运行 - 永不阻塞 (macOS兼容)
is_training_running() {
    local output
    # 使用 SSH 内置超时替代 timeout 命令 (macOS 无 timeout)
    output=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 \
        -p $PORT $USER@$HOST "ps aux | grep -c '[t]rainxuanwu'" 2>/dev/null || echo "0")
    local count=$(echo "$output" | grep -E '^[0-9]+$' | tail -1 || echo "0")
    [[ "$count" != "0" && -n "$count" ]]
}

# 获取最新日志文件
get_latest_log() {
    ssh_run "ls -t $REPO_PATH/train_autodl_*.log 2>/dev/null | head -1" 2>/dev/null || echo ""
}

# 收集结果 - 永不阻塞
collect_results() {
    local run_dir="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"

    local log_path=$(get_latest_log)
    log "收集结果: $log_path"

    # 下载日志 - 失败也继续
    if [[ -n "$log_path" && "$log_path" != *[Nn]o\ such* ]]; then
        ssh_run "cat '$log_path'" > "$run_dir/training.log" 2>/dev/null || true
    fi

    # 提取指标 (使用 awk 代替 grep -P)
    local psnr="N/A" ssim="N/A" mse="N/A" best="N/A"
    if [[ -f "$run_dir/training.log" && -s "$run_dir/training.log" ]]; then
        psnr=$(grep "Val PSNR" "$run_dir/training.log" 2>/dev/null | awk '{print $NF}' | tail -1 || echo "N/A")
        ssim=$(grep "Val SSIM" "$run_dir/training.log" 2>/dev/null | awk '{print $NF}' | tail -1 || echo "N/A")
        mse=$(grep "Val MSE" "$run_dir/training.log" 2>/dev/null | awk '{print $NF}' | tail -1 || echo "N/A")
        best=$(grep "Best PSNR" "$run_dir/training.log" 2>/dev/null | awk '{print $NF}' | tail -1 || echo "N/A")
    fi

    cat << EOF > "$run_dir/metrics.json"
{
    "timestamp": "$(date -Iseconds)",
    "epochs": $EPOCHS,
    "best_psnr": "$best",
    "val_psnr": "$psnr",
    "val_ssim": "$ssim",
    "val_mse": "$mse"
}
EOF

    cat << EOF > "$run_dir/insights.md"
# 训练报告 - $(date)

## 指标
- Best PSNR: $best
- Val PSNR: $psnr
- Val SSIM: $ssim
- Val MSE: $mse

## 日志摘要
$(tail -50 "$run_dir/training.log" 2>/dev/null || echo "无日志")
EOF

    log "结果已保存: $run_dir"
    log "Best PSNR: $best, Val PSNR: $psnr, SSIM: $ssim"

    echo "$psnr"
}

# 启动训练 - 直接执行方式 (避免 heredoc 问题)
start_training() {
    log "启动新训练 ($EPOCHS epochs)..."

    # 构建训练命令 - 直接在远程执行
    local TRAIN_CMD="cd $CODE_PATH && source /etc/network_turbo && nohup /root/miniconda3/bin/python trainxuanwu.py train --net_idx=$NET_IDX --path_key=$DATASET_KEY --epoch=$EPOCHS --use_augmentation=True --aug_prob=0.5 --clip_ct=True --min_hu=-1024 --max_hu=3071 --normalize_ct=True --num_workers=4 --test_num_workers=2 > $REPO_PATH/train_autodl.log 2>&1 &"

    # 直接执行训练命令
    log "执行: $TRAIN_CMD"
    ssh_run_retry "$TRAIN_CMD" || {
        log "[警告] SSH执行失败，尝试直接执行..."
        ssh_run "$TRAIN_CMD" || true
    }

    sleep 8

    if is_training_running; then
        log "训练已启动!"
    else
        log "[警告] 训练进程未检测到，等待后重试..."
        sleep 10
        is_training_running && log "训练已启动!" || log "[注意] 可能训练仍在启动中，继续监控"
    fi
}

# 等待训练完成 - 永不阻塞
wait_training() {
    log "等待训练完成..."

    local elapsed=0
    local check_interval=60
    local max_wait=14400  # 4小时

    while true; do
        if ! is_training_running; then
            log "训练已完成"
            return 0
        fi

        # 获取最新日志进度
        local log_file=$(get_latest_log)
        if [[ -n "$log_file" && "$log_file" != *[Nn]o\ such* ]]; then
            local last_line
            last_line=$(ssh_run "tail -1 '$log_file' 2>/dev/null | head -c 150" 2>/dev/null || echo "")
            log "[${elapsed}s] 训练进行中... ${last_line:0:100}"
        else
            log "[${elapsed}s] 训练进行中... (日志文件未找到)"
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))

        if [[ $elapsed -ge $max_wait ]]; then
            log "[超时] 等待时间超过${max_wait}s，停止等待"
            return 1
        fi
    done
}

#===============================================================================
# 主循环 - 永不阻塞
#===============================================================================

main() {
    log "=========================================="
    log "AutoDL 自主训练循环启动 (永不阻塞版)"
    log "SSH 免密码: 已配置"
    log "每次循环: ~${WAIT_CYCLE}s"
    log "最多循环: ${MAX_LOOPS}次"
    log "=========================================="

    local loop=1

    while [[ $loop -le $MAX_LOOPS ]]; do
        log ""
        log "=== 循环 #$loop/${MAX_LOOPS} ==="

        # 检测训练状态 - 失败也继续
        if is_training_running; then
            log "检测到训练进行中，等待完成..."
            wait_training
            collect_results
        else
            log "无训练运行，启动新训练..."
            start_training
            wait_training
            collect_results
        fi

        log "=== 循环 #$loop 完成，等待 ${WAIT_CYCLE}s 后继续... ==="
        sleep $WAIT_CYCLE

        loop=$((loop + 1))
    done

    log "达到最大循环次数 ${MAX_LOOPS}，训练循环结束"
}

#===============================================================================
# 启动 (后台运行，输出PID)
#===============================================================================

# 确保 stdin 不是终端（防止任何阻塞提示）
if [[ -t 0 ]]; then
    exec </dev/null
fi

main &
PID=$!
echo "后台进程 PID: $PID"
echo "日志: $MAIN_LOG"
echo "停止命令: kill $PID"
log "主进程 PID: $PID"