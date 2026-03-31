#!/bin/bash
#===============================================================================
# AutoDL 训练启动脚本
# 用法: ./run_training.sh [epochs] [--collect-only]
#
# 流程:
# 1. 检查 GPU 状态
# 2. Git 同步最新代码
# 3. 启动训练 (后台运行)
# 4. 等待训练完成 (指数退让)
# 5. 收集结果到本地 logs/
#===============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# 加载配置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# 解析参数
EPOCHS="${1:-10}"
COLLECT_ONLY=false
if [[ "$2" == "--collect-only" ]]; then
    COLLECT_ONLY=true
fi

# 创建本地运行目录
mkdir -p "$LOCAL_LOG_DIR/run_$(date +%Y%m%d_%H%M%S)"

#===============================================================================
# SSH 执行命令
#===============================================================================
ssh_cmd() {
    local cmd="$1"
    expect << EOF
        set timeout 120
        log_user 0
        spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $AUTODL_PORT $AUTODL_USER@$AUTODL_HOST
        expect "password:"
        send "$AUTODL_PASS\r"
        expect "$ "
        send "$cmd\r"
        expect "$ "
        send "exit\r"
        expect eof
EOF
}

#===============================================================================
# 检查 GPU 状态
#===============================================================================
check_gpu() {
    log_step "检查 GPU 状态"
    ssh_cmd "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"
}

#===============================================================================
# Git 同步代码
#===============================================================================
git_sync() {
    log_step "同步代码 (git pull)"
    ssh_cmd "source /etc/network_turbo && cd $AUTODL_REPO_PATH && git fetch origin $GIT_BRANCH && git pull origin $GIT_BRANCH"
}

#===============================================================================
# 检查是否有正在运行的训练
#===============================================================================
check_running() {
    ssh_cmd "ps aux | grep trainxuanwu | grep -v grep | head -1"
}

#===============================================================================
# 启动训练
#===============================================================================
start_training() {
    log_step "启动训练 ($EPOCHS epochs)"
    ssh_cmd "cd $AUTODL_CODE_PATH && \
        LOG_FILE='../train_autodl_\$(date +%Y%m%d_%H%M%S).log' && \
        echo \"Log: \$LOG_FILE\" && \
        nohup python $TRAIN_SCRIPT train \
            --net_idx=$NET_IDX \
            --path_key=$DATASET_KEY \
            --epoch=$EPOCHS \
            --use_augmentation=True \
            --aug_prob=0.5 \
            --clip_ct=True \
            --min_hu=-1024 \
            --max_hu=3071 \
            --normalize_ct=True \
            --num_workers=4 \
            --test_num_workers=2 \
            > \"\$LOG_FILE\" 2>&1 & \
        sleep 2 && \
        ps aux | grep trainxuanwu | grep -v grep | head -1"
}

#===============================================================================
# 等待训练完成 (指数退让)
#===============================================================================
wait_for_completion() {
    log_step "等待训练完成 (最多 60 分钟)"

    local max_wait=3600  # 60 分钟
    local interval=30     # 初始间隔 30 秒
    local max_interval=300 # 最大间隔 5 分钟
    local elapsed=0
    local log_file=""

    # 等待训练进程启动
    log_info "等待训练进程启动..."
    while true; do
        local result=$(ssh_cmd "ps aux | grep -c '[t]rainxuanwu' || echo 0")
        if [[ "$result" != "0" ]]; then
            log_success "训练进程已启动"
            break
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        if [[ $elapsed -ge 120 ]]; then
            log_warn "等待启动超时，继续监控..."
            break
        fi
    done

    # 获取日志文件名
    log_file=$(ssh_cmd "ls -t $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null | head -1")
    log_info "日志文件: $log_file"

    # 监控训练进度
    elapsed=0
    while true; do
        # 检查进程是否还在运行
        local running=$(ssh_cmd "ps aux | grep -c '[t]rainxuanwu' || echo 0")

        if [[ "$running" == "0" ]]; then
            log_success "训练已完成!"
            break
        fi

        # 获取当前进度
        local epoch_info=$(ssh_cmd "grep -E 'Epoch [0-9]+.*lr=' $log_file 2>/dev/null | tail -1 || echo 'Training...'")
        local time_info=$(ssh_cmd "tail -5 $log_file 2>/dev/null | grep -oE '[0-9]+it.*' | tail -1 || echo ''")

        echo -ne "${BLUE}[$(date +%H:%M:%S)]${NC} 已运行 ${elapsed}s | $epoch_info | $time_info    \r"

        sleep $interval
        elapsed=$((elapsed + interval))

        # 指数退让
        if [[ $interval -lt $max_interval ]]; then
            interval=$((interval * 2))
            [[ $interval -gt $max_interval ]] && interval=$max_interval
        fi

        # 超时退出
        if [[ $elapsed -ge $max_wait ]]; then
            log_warn "达到最大等待时间，停止监控"
            break
        fi
    done
    echo ""  # 换行
}

#===============================================================================
# 收集结果
#===============================================================================
collect_results() {
    log_step "收集训练结果"

    local run_dir="$LOCAL_LOG_DIR/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"

    # 下载最新日志
    log_info "下载训练日志..."
    ssh_cmd "cat $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null | tail -200" > "$run_dir/training.log"

    # 提取关键指标
    log_info "提取关键指标..."
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"epochs\": $EPOCHS,"

        # 提取最后的指标
        local psnr=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
        local ssim=$(grep -oP 'SSIM[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
        local mse=$(grep -oP 'MSE[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
        local loss=$(grep -oP 'L1[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")

        echo "  \"psnr\": \"$psnr\","
        echo "  \"ssim\": \"$ssim\","
        echo "  \"mse\": \"$mse\","
        echo "  \"loss\": \"$loss\","
        echo "  \"log_file\": \"$run_dir/training.log\""
        echo "}"
    } > "$run_dir/metrics.json"

    # 生成心得报告
    log_info "生成训练心得..."
    {
        echo "# 训练心得 - $(date)"
        echo ""
        echo "## 配置"
        echo "- Epochs: $EPOCHS"
        echo "- 数据集: 宣武数据集 (1:4)"
        echo "- 网络: $NET_IDX"
        echo ""
        echo "## 指标"
        cat "$run_dir/metrics.json"
        echo ""
        echo "## 日志摘要 (最后 50 行)"
        tail -50 "$run_dir/training.log"
    } > "$run_dir/insights.md"

    log_success "结果已保存到: $run_dir/"
    echo ""
    cat "$run_dir/metrics.json"
}

#===============================================================================
# 主流程
#===============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  AutoDL MLOps 训练闭环"
    echo "=============================================="
    log_info "Git Commit: $(git rev-parse --short HEAD)"
    log_info "Epochs: $EPOCHS"
    log_info "模式: $(if $COLLECT_ONLY; then echo '仅收集结果'; else echo '完整流程'; fi)"
    echo ""

    if $COLLECT_ONLY; then
        collect_results
        return
    fi

    # Step 1: 检查 GPU
    check_gpu
    echo ""

    # Step 2: Git 同步
    git_sync
    echo ""

    # Step 3: 检查是否已有训练在运行
    log_info "检查训练状态..."
    local running=$(ssh_cmd "ps aux | grep -c '[t]rainxuanwu' || echo 0")
    if [[ "$running" != "0" ]]; then
        log_warn "已有训练在运行，是否等待完成? (Ctrl+C 取消)"
        sleep 5
    fi
    echo ""

    # Step 4: 启动训练
    start_training
    echo ""

    # Step 5: 等待完成
    wait_for_completion
    echo ""

    # Step 6: 收集结果
    collect_results

    echo ""
    echo "=============================================="
    log_success "训练闭环完成!"
    echo "=============================================="
}

main "$@"
