#!/bin/bash
#===============================================================================
# AutoDL 训练启动脚本 (支持模块化配置)
# 
# 用法: ./run_training.sh [选项] [epochs] [--collect-only]
#
# 选项:
#   --config <file>          使用配置文件
#   --loss <module>          使用 Loss 模块配置
#   --augment <module>       使用 Augmentation 模块配置
#   --training <module>      使用 Training 模块配置
#   --exp <name>             实验名称
#   --collect-only           仅收集结果
#
# 示例:
#   ./run_training.sh 50
#   ./run_training.sh --loss eagle3d --augment flip --training adamw_ema 100
#   ./run_training.sh --config ../config/experiments/exp01.json 50
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

MODULES_DIR="${SCRIPT_DIR}/../../config/modules"
EXPERIMENTS_DIR="${SCRIPT_DIR}/../../config/experiments"

# 颜色
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

# 默认参数
EPOCHS="${DEFAULT_EPOCHS:-50}"
COLLECT_ONLY=false
USE_CONFIG=""
LOSS_MODULE=""
AUGMENT_MODULE=""
TRAINING_MODULE=""
EXP_NAME=""
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            USE_CONFIG="$2"
            shift 2
            ;;
        --loss)
            LOSS_MODULE="$2"
            shift 2
            ;;
        --augment)
            AUGMENT_MODULE="$2"
            shift 2
            ;;
        --training)
            TRAINING_MODULE="$2"
            shift 2
            ;;
        --exp)
            EXP_NAME="$2"
            shift 2
            ;;
        --collect-only)
            COLLECT_ONLY=true
            shift
            ;;
        --help|-h)
            echo "AutoDL 训练启动脚本"
            echo ""
            echo "用法: $0 [选项] [epochs] [--collect-only]"
            echo ""
            echo "选项:"
            echo "  --config <file>      使用实验配置文件"
            echo "  --loss <module>      Loss 模块 (如: l1, eagle3d, charbonnier)"
            echo "  --augment <module>   Augmentation 模块 (如: none, flip, combined)"
            echo "  --training <module>  Training 模块 (如: baseline, adamw_ema, advanced)"
            echo "  --exp <name>         实验名称"
            echo "  --collect-only       仅收集结果"
            echo "  -h, --help           显示帮助"
            echo ""
            echo "示例:"
            echo "  $0 50                                    # 基础用法"
            echo "  $0 --loss eagle3d --training advanced 100  # 模块化配置"
            echo "  $0 --config ../config/experiments/exp.json 50"
            echo ""
            echo "可用模块:"
            "$SCRIPT_DIR/list_configs.sh"
            exit 0
            ;;
        [0-9]*)
            EPOCHS="$1"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 生成训练参数
generate_train_params() {
    local params=""
    
    # 从配置文件加载
    if [[ -n "$USE_CONFIG" ]] && [[ -f "$USE_CONFIG" ]]; then
        log_info "使用配置文件: $USE_CONFIG"
        
        # 解析 JSON 配置
        if command -v python3 &> /dev/null; then
            LOSS_TYPE=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('loss',{}).get('type','L1'))")
            AUG_ENABLED=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('augmentation',{}).get('enabled',False))")
            OPTIM=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('training',{}).get('optimizer','AdamW'))")
            LR=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('training',{}).get('lr',0.0003))")
            WD=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('training',{}).get('wd',0.0001))")
            USE_EMA=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('training',{}).get('use_ema',False))")
            USE_WARMUP=$(python3 -c "import sys,json; print(json.load(open('$USE_CONFIG')).get('training',{}).get('use_warmup',False))")
            
            params="--loss_f=$LOSS_TYPE --optim=$OPTIM --lr=$LR --wd=$WD"
            params="$params --use_ema=$USE_EMA --use_warmup=$USE_WARMUP --use_augmentation=$AUG_ENABLED"
        fi
        
        echo "$params"
        return
    fi
    
    # 从模块加载
    if [[ -n "$LOSS_MODULE" ]]; then
        local loss_file="${MODULES_DIR}/loss/${LOSS_MODULE}.txt"
        if [[ -f "$loss_file" ]]; then
            LOSS_TYPE=$(grep "^type" "$loss_file" | cut -d'=' -f2 | tr -d ' "')
            params="$params --loss_f=$LOSS_TYPE"
            log_info "Loss: $LOSS_MODULE -> $LOSS_TYPE"
        fi
    fi
    
    if [[ -n "$AUGMENT_MODULE" ]]; then
        local aug_file="${MODULES_DIR}/augment/${AUGMENT_MODULE}.txt"
        if [[ -f "$aug_file" ]]; then
            AUG_ENABLED=$(grep "^enabled" "$aug_file" | cut -d'=' -f2 | tr -d ' ')
            params="$params --use_augmentation=$AUG_ENABLED"
            log_info "Augmentation: $AUGMENT_MODULE -> enabled=$AUG_ENABLED"
        fi
    fi
    
    if [[ -n "$TRAINING_MODULE" ]]; then
        local train_file="${MODULES_DIR}/training/${TRAINING_MODULE}.txt"
        if [[ -f "$train_file" ]]; then
            OPTIM=$(grep "^optimizer" "$train_file" | cut -d'=' -f2 | tr -d ' "')
            LR=$(grep "^lr" "$train_file" | cut -d'=' -f2 | tr -d ' ')
            WD=$(grep "^wd" "$train_file" | cut -d'=' -f2 | tr -d ' ')
            USE_EMA=$(grep "^use_ema" "$train_file" | cut -d'=' -f2 | tr -d ' ')
            USE_WARMUP=$(grep "^use_warmup" "$train_file" | cut -d'=' -f2 | tr -d ' ')
            
            params="$params --optim=$OPTIM --lr=$LR --wd=$WD"
            params="$params --use_ema=$USE_EMA --use_warmup=$USE_WARMUP"
            log_info "Training: $TRAINING_MODULE -> optim=$OPTIM, lr=$LR"
        fi
    fi
    
    echo "$params"
}

# 生成实验名称
generate_exp_name() {
    if [[ -n "$EXP_NAME" ]]; then
        echo "$EXP_NAME"
        return
    fi
    
    local name=""
    if [[ -n "$LOSS_MODULE" ]]; then
        name="${LOSS_MODULE}"
    fi
    if [[ -n "$AUGMENT_MODULE" ]]; then
        name="${name}_${AUGMENT_MODULE}"
    fi
    if [[ -n "$TRAINING_MODULE" ]]; then
        name="${name}_${TRAINING_MODULE}"
    fi
    if [[ -z "$name" ]]; then
        name="${NET_IDX}"
    fi
    
    echo "$name"
}

# SSH 执行命令
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

# 检查 GPU 状态
check_gpu() {
    log_step "检查 GPU 状态"
    ssh_cmd "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"
}

# Git 同步代码
git_sync() {
    log_step "同步代码 (git pull)"
    ssh_cmd "source /etc/network_turbo && cd $AUTODL_REPO_PATH && git fetch origin $GIT_BRANCH && git pull origin $GIT_BRANCH"
}

# 启动训练
start_training() {
    local train_params="$1"
    local exp_name="$2"
    
    log_step "启动训练 ($EPOCHS epochs)"
    
    # 构建完整命令
    local cmd="cd $AUTODL_CODE_PATH && \\
        LOG_FILE='../train_${exp_name}_\$(date +%Y%m%d_%H%M%S).log' && \\
        echo \"Log: \$LOG_FILE\" && \\
        nohup python $TRAIN_SCRIPT train \\"
    
    # 添加网络和数据集配置
    cmd="$cmd
            --net_idx=$exp_name \\"
    cmd="$cmd
            --path_key=$DATASET_KEY \\"
    cmd="$cmd
            --epoch=$EPOCHS \\"
    
    # 添加默认参数
    cmd="$cmd
            --clip_ct=True \\"
    cmd="$cmd
            --min_hu=-1024 \\"
    cmd="$cmd
            --max_hu=3071 \\"
    cmd="$cmd
            --normalize_ct=True \\"
    cmd="$cmd
            --num_workers=4 \\"
    cmd="$cmd
            --test_num_workers=2 \\"
    
    # 添加模块化配置参数
    if [[ -n "$train_params" ]]; then
        cmd="$cmd
            $train_params \\"
    fi
    
    # 添加额外参数
    if [[ -n "$EXTRA_ARGS" ]]; then
        cmd="$cmd
            $EXTRA_ARGS \\"
    fi
    
    # 输出重定向
    cmd="$cmd
            > \"\$LOG_FILE\" 2>&1 & \\"
    cmd="$cmd
        sleep 2 && \\"
    cmd="$cmd
        ps aux | grep trainxuanwu | grep -v grep | head -1"
    
    ssh_cmd "$cmd"
}

# 等待训练完成
wait_for_completion() {
    log_step "等待训练完成 (最多 60 分钟)"
    
    local max_wait=3600
    local interval=30
    local max_interval=300
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
    log_file=$(ssh_cmd "ls -t $AUTODL_REPO_PATH/train_*.log 2>/dev/null | head -1")
    log_info "日志文件: $log_file"
    
    # 监控训练进度
    elapsed=0
    while true; do
        local running=$(ssh_cmd "ps aux | grep -c '[t]rainxuanwu' || echo 0")
        
        if [[ "$running" == "0" ]]; then
            log_success "训练已完成!"
            break
        fi
        
        local epoch_info=$(ssh_cmd "grep -E 'Epoch [0-9]+.*lr=' $log_file 2>/dev/null | tail -1 || echo 'Training...'")
        local time_info=$(ssh_cmd "tail -5 $log_file 2>/dev/null | grep -oE '[0-9]+it.*' | tail -1 || echo '')"
        
        echo -ne "${BLUE}[$(date +%H:%M:%S)]${NC} 已运行 ${elapsed}s | $epoch_info | $time_info    \r"
        
        sleep $interval
        elapsed=$((elapsed + interval))
        
        if [[ $interval -lt $max_interval ]]; then
            interval=$((interval * 2))
            [[ $interval -gt $max_interval ]] && interval=$max_interval
        fi
        
        if [[ $elapsed -ge $max_wait ]]; then
            log_warn "达到最大等待时间，停止监控"
            break
        fi
    done
    echo ""
}

# 收集结果
collect_results() {
    log_step "收集训练结果"
    
    local exp_name="$1"
    local run_dir="$LOCAL_LOG_DIR/run_${exp_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"
    
    # 下载最新日志
    log_info "下载训练日志..."
    ssh_cmd "cat $AUTODL_REPO_PATH/train_*.log 2>/dev/null | tail -200" > "$run_dir/training.log"
    
    # 提取关键指标
    log_info "提取关键指标..."
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"exp_name\": \"$exp_name\","
        echo "  \"epochs\": $EPOCHS,"
        
        local psnr=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
        local ssim=$(grep -oP 'SSIM[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
        local loss=$(grep -oP 'train_loss[:\s]+\K[\d.]+' "$run_dir/training.log" | tail -1 || echo "N/A")
        
        echo "  \"psnr\": \"$psnr\","
        echo "  \"ssim\": \"$ssim\","
        echo "  \"loss\": \"$loss\","
        echo "  \"modules\": {"
        echo "    \"loss\": \"${LOSS_MODULE:-default}\","
        echo "    \"augmentation\": \"${AUGMENT_MODULE:-default}\","
        echo "    \"training\": \"${TRAINING_MODULE:-default}\""
        echo "  },"
        echo "  \"log_file\": \"$run_dir/training.log\""
        echo "}"
    } > "$run_dir/metrics.json"
    
    # 生成报告
    log_info "生成训练报告..."
    {
        echo "# 训练报告 - $exp_name"
        echo ""
        echo "## 配置"
        echo "- Epochs: $EPOCHS"
        echo "- Loss: ${LOSS_MODULE:-default}"
        echo "- Augmentation: ${AUGMENT_MODULE:-default}"
        echo "- Training: ${TRAINING_MODULE:-default}"
        echo ""
        echo "## 指标"
        cat "$run_dir/metrics.json"
        echo ""
        echo "## 日志摘要 (最后 50 行)"
        echo "\`\`\`"
        tail -50 "$run_dir/training.log"
        echo "\`\`\`"
    } > "$run_dir/report.md"
    
    log_success "结果已保存到: $run_dir/"
    echo ""
    cat "$run_dir/metrics.json"
}

# 主流程
main() {
    local exp_name=$(generate_exp_name)
    local train_params=$(generate_train_params)
    
    echo ""
    echo "=============================================="
    echo "  AutoDL MLOps 训练闭环"
    echo "=============================================="
    log_info "实验名称: $exp_name"
    log_info "训练轮数: $EPOCHS"
    log_info "模式: $(if $COLLECT_ONLY; then echo '仅收集结果'; else echo '完整流程'; fi)"
    if [[ -n "$train_params" ]]; then
        log_info "附加参数: $train_params"
    fi
    echo ""
    
    if $COLLECT_ONLY; then
        collect_results "$exp_name"
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
    start_training "$train_params" "$exp_name"
    echo ""
    
    # Step 5: 等待完成
    wait_for_completion
    echo ""
    
    # Step 6: 收集结果
    collect_results "$exp_name"
    
    echo ""
    echo "=============================================="
    log_success "训练闭环完成!"
    echo "=============================================="
}

main "$@"
