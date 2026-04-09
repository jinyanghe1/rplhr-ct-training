#!/bin/bash
#===============================================================================
# 快速启动实验脚本
# 用法: ./quick_experiment.sh <exp_name> <loss_cfg> <aug_cfg> <train_cfg> [epochs]
#
# 示例:
#   ./quick_experiment.sh exp01 l1 flip adamw_ema 50
#   ./quick_experiment.sh exp02 eagle3d combined advanced 100
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

# 检查参数
if [ $# -lt 4 ]; then
    echo "用法: $0 <exp_name> <loss_cfg> <aug_cfg> <train_cfg> [epochs]"
    echo ""
    echo "参数:"
    echo "  exp_name    实验名称 (如: exp01, eagle_test)"
    echo "  loss_cfg    Loss 配置 (如: l1, eagle3d, charbonnier)"
    echo "  aug_cfg     Augmentation 配置 (如: none, flip, combined)"
    echo "  train_cfg   Training 配置 (如: baseline, adamw_ema, advanced)"
    echo "  epochs      训练轮数 (默认: 50)"
    echo ""
    echo "示例:"
    echo "  $0 exp01 l1 flip adamw_ema"
    echo "  $0 exp02 eagle3d combined advanced 100"
    echo ""
    echo "可用配置列表:"
    "$SCRIPT_DIR/list_configs.sh" --json | python3 -m json.tool 2>/dev/null || "$SCRIPT_DIR/list_configs.sh"
    exit 1
fi

EXP_NAME="$1"
LOSS_CFG="$2"
AUG_CFG="$3"
TRAIN_CFG="$4"
EPOCHS="${5:-50}"

# 检查配置文件是否存在
check_config() {
    local cfg_type="$1"
    local cfg_name="$2"
    local cfg_file="${MODULES_DIR}/${cfg_type}/${cfg_name}.txt"
    
    if [[ ! -f "$cfg_file" ]]; then
        log_error "配置不存在: ${cfg_type}/${cfg_name}"
        log_info "可用配置:"
        ls -1 "${MODULES_DIR}/${cfg_type}"/*.txt 2>/dev/null | xargs -n1 basename | sed 's/\.txt$//' | sed 's/^/  - /'
        exit 1
    fi
}

log_step "检查配置..."
check_config "loss" "$LOSS_CFG"
check_config "augment" "$AUG_CFG"
check_config "training" "$TRAIN_CFG"
log_success "配置检查通过"

# 创建实验配置
log_step "创建实验配置..."
mkdir -p "$EXPERIMENTS_DIR"

EXP_CONFIG_FILE="${EXPERIMENTS_DIR}/${EXP_NAME}.json"

# 读取各模块配置
read_module_config() {
    local cfg_file="$1"
    local config="{}"
    
    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^# ]] && continue
        [[ ! "$line" =~ = ]] && continue
        
        key=$(echo "$line" | cut -d'=' -f1 | sed 's/[[:space:]]*$//')
        value=$(echo "$line" | cut -d'=' -f2- | sed 's/^[[:space:]]*//')
        
        # 尝试解析值类型
        if [[ "$value" =~ ^[0-9]+\. ]]; then
            # 浮点数
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']=float('$value'); print(json.dumps(d))")
        elif [[ "$value" =~ ^[0-9]+$ ]]; then
            # 整数
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']=int('$value'); print(json.dumps(d))")
        elif [[ "$value" == "True" ]] || [[ "$value" == "true" ]]; then
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']=True; print(json.dumps(d))")
        elif [[ "$value" == "False" ]] || [[ "$value" == "false" ]]; then
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']=False; print(json.dumps(d))")
        elif [[ "$value" =~ ^\[.*\]$ ]]; then
            # 列表
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']=$value; print(json.dumps(d))")
        elif [[ "$value" =~ ^\{.*\}$ ]]; then
            # 字典
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']=$value; print(json.dumps(d))")
        else
            # 字符串
            config=$(echo "$config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['$key']='$value'; print(json.dumps(d))")
        fi
    done < "$cfg_file"
    
    echo "$config"
}

LOSS_CONFIG=$(read_module_config "${MODULES_DIR}/loss/${LOSS_CFG}.txt")
AUGMENT_CONFIG=$(read_module_config "${MODULES_DIR}/augment/${AUG_CFG}.txt")
TRAINING_CONFIG=$(read_module_config "${MODULES_DIR}/training/${TRAIN_CFG}.txt")

# 创建完整实验配置
TIMESTAMP=$(date -Iseconds)
cat > "$EXP_CONFIG_FILE" << EOF
{
  "exp_name": "$EXP_NAME",
  "timestamp": "$TIMESTAMP",
  "epochs": $EPOCHS,
  "modules": {
    "loss": "$LOSS_CFG",
    "augmentation": "$AUG_CFG",
    "training": "$TRAIN_CFG"
  },
  "loss": $LOSS_CONFIG,
  "augmentation": $AUGMENT_CONFIG,
  "training": $TRAINING_CONFIG
}
EOF

log_success "实验配置已保存: $EXP_CONFIG_FILE"

# 生成训练命令参数
generate_train_args() {
    local args=""
    
    # Loss 参数
    local loss_type=$(echo "$LOSS_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('type','L1'))")
    args="$args --loss_type=$loss_type"
    
    # Augmentation 参数
    local aug_enabled=$(echo "$AUGMENT_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('enabled',False))")
    args="$args --use_augmentation=$aug_enabled"
    
    if [[ "$aug_enabled" == "True" ]]; then
        local flip_prob=$(echo "$AUGMENT_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('flip_prob',0.5))")
        args="$args --aug_prob=$flip_prob"
    fi
    
    # Training 参数
    local optim=$(echo "$TRAINING_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('optimizer','AdamW'))")
    local lr=$(echo "$TRAINING_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('lr',0.0003))")
    local wd=$(echo "$TRAINING_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('wd',0.0001))")
    local use_ema=$(echo "$TRAINING_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('use_ema',False))")
    local use_warmup=$(echo "$TRAINING_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('use_warmup',False))")
    local use_grad_clip=$(echo "$TRAINING_CONFIG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('use_grad_clip',False))")
    
    args="$args --optim=$optim --lr=$lr --wd=$wd"
    args="$args --use_ema=$use_ema --use_warmup=$use_warmup --use_grad_clip=$use_grad_clip"
    
    echo "$args"
}

TRAIN_ARGS=$(generate_train_args)

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              实验配置摘要                                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}实验名称:${NC}    $EXP_NAME"
echo -e "  ${BLUE}训练轮数:${NC}    $EPOCHS"
echo -e "  ${BLUE}Loss 配置:${NC}   $LOSS_CFG"
echo -e "  ${BLUE}增强配置:${NC}    $AUG_CFG"
echo -e "  ${BLUE}训练配置:${NC}    $TRAIN_CFG"
echo ""
echo -e "  ${BLUE}训练参数:${NC}"
echo "    $TRAIN_ARGS" | fold -s -w 60 | sed 's/^/    /'
echo ""

# 确认启动
if [[ -t 0 ]]; then
    read -p "是否确认启动实验? [Y/n] " confirm
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        log_info "实验已取消"
        exit 0
    fi
fi

# 同步配置到远程
log_step "同步配置到 AutoDL..."
scp -P "$AUTODL_PORT" -o StrictHostKeyChecking=no "$EXP_CONFIG_FILE" "${AUTODL_USER}@${AUTODL_HOST}:${AUTODL_REPO_PATH}/config/experiments/" 2>/dev/null || {
    log_warn "无法同步配置到远程，将在本地保存"
}

# 更新 EXPERIMENTS.md
log_step "更新实验记录..."
EXPERIMENTS_FILE="${SCRIPT_DIR}/EXPERIMENTS.md"
DATE=$(date '+%Y-%m-%d')

cat >> "$EXPERIMENTS_FILE" << EOF

---

## EXP_ID: ${EXP_NAME}

### 基础信息
- **实验名称**: ${EXP_NAME}
- **日期**: ${DATE}
- **配置**: Loss=${LOSS_CFG}, Augment=${AUG_CFG}, Training=${TRAIN_CFG}
- **状态**: ⏳ 运行中

### 配置详情
- **Loss**: ${LOSS_CFG}
  \`\`\`json
${LOSS_CONFIG}
  \`\`\`
- **Augmentation**: ${AUG_CFG}
  \`\`\`json
${AUGMENT_CONFIG}
  \`\`\`
- **Training**: ${TRAIN_CFG}
  \`\`\`json
${TRAINING_CONFIG}
  \`\`\`

### 实验结果
| Epoch | Train Loss | Val PSNR | Val SSIM | 备注 |
|-------|------------|----------|----------|------|

EOF

log_success "实验记录已更新"

# 启动训练
echo ""
log_step "启动训练..."
echo ""
echo "使用以下命令启动训练:"
echo ""
echo "  cd ${AUTODL_CODE_PATH} && \\"
echo "  nohup python ${TRAIN_SCRIPT} train \\"
echo "    --net_idx=${EXP_NAME} \\"
echo "    --path_key=${DATASET_KEY} \\"
echo "    --epoch=${EPOCHS} \\"
echo "    ${TRAIN_ARGS} \\"
echo "    > ../train_${EXP_NAME}.log 2>&1 &"
echo ""

# 询问是否立即启动
if [[ -t 0 ]]; then
    read -p "是否立即启动训练? [Y/n] " start_now
    if [[ ! "$start_now" =~ ^[Nn]$ ]]; then
        log_step "正在启动训练..."
        
        # 通过 SSH 启动
        ssh -p "$AUTODL_PORT" -o StrictHostKeyChecking=no "${AUTODL_USER}@${AUTODL_HOST}" << REMOTE_CMD
cd ${AUTODL_CODE_PATH} && \
nohup python ${TRAIN_SCRIPT} train \
  --net_idx=${EXP_NAME} \
  --path_key=${DATASET_KEY} \
  --epoch=${EPOCHS} \
  ${TRAIN_ARGS} \
  > ../train_${EXP_NAME}.log 2>&1 &
echo "训练已启动，PID: \$!"
REMOTE_CMD
        
        log_success "训练已启动!"
        log_info "查看日志: ./check_training.sh --tail"
        log_info "监控状态: ./monitor_training.sh"
    else
        log_info "训练命令已保存，可稍后手动启动"
    fi
else
    log_info "非交互模式，跳过自动启动"
fi

echo ""
log_success "实验 ${EXP_NAME} 设置完成!"
echo ""
