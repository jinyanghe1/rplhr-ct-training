#!/bin/bash
#===============================================================================
# 创建新实验记录 (支持模块化配置)
# 
# 用法: 
#   基本用法: ./create_experiment.sh <实验ID> <实验名称> [基线ID]
#   模块化配置: ./create_experiment.sh --loss <loss_cfg> --augment <aug_cfg> --training <train_cfg> --name <实验名称>
#
# 示例:
#   ./create_experiment.sh A1 "EAGLELoss3D" A0
#   ./create_experiment.sh --loss eagle3d --augment flip --training adamw_ema --name "exp_eagle_adamw"
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_FILE="$SCRIPT_DIR/EXPERIMENTS.md"
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

# 解析参数
USE_MODULES=false
LOSS_CFG=""
AUG_CFG=""
TRAIN_CFG=""
EXP_NAME=""
EXP_ID=""
BASELINE="A0"
EPOCHS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --loss)
            USE_MODULES=true
            LOSS_CFG="$2"
            shift 2
            ;;
        --augment)
            USE_MODULES=true
            AUG_CFG="$2"
            shift 2
            ;;
        --training)
            USE_MODULES=true
            TRAIN_CFG="$2"
            shift 2
            ;;
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --help|-h)
            echo "创建新实验记录"
            echo ""
            echo "用法:"
            echo "  基本用法: $0 <实验ID> <实验名称> [基线ID]"
            echo "  模块化配置: $0 --loss <cfg> --augment <cfg> --training <cfg> --name <名称>"
            echo ""
            echo "选项:"
            echo "  --loss <cfg>        Loss 配置 (如: l1, eagle3d, charbonnier)"
            echo "  --augment <cfg>     Augmentation 配置 (如: none, flip, combined)"
            echo "  --training <cfg>    Training 配置 (如: baseline, adamw_ema, advanced)"
            echo "  --name <名称>       实验名称"
            echo "  --epochs <N>        训练轮数"
            echo "  -h, --help          显示帮助"
            echo ""
            echo "示例:"
            echo "  $0 A1 \"EAGLELoss3D\" A0"
            echo "  $0 --loss eagle3d --augment flip --training adamw_ema --name exp_eagle"
            echo ""
            exit 0
            ;;
        *)
            # 位置参数
            if [[ -z "$EXP_ID" ]]; then
                EXP_ID="$1"
            elif [[ -z "$EXP_NAME" ]]; then
                EXP_NAME="$1"
            elif [[ -z "$BASELINE" ]]; then
                BASELINE="$1"
            fi
            shift
            ;;
    esac
done

# 如果没有提供基线，使用默认值
if [[ -z "$BASELINE" ]]; then
    BASELINE="A0"
fi

DATE=$(date '+%Y-%m-%d')

# 检查配置是否存在
check_module() {
    local type="$1"
    local name="$2"
    local cfg_file="${MODULES_DIR}/${type}/${name}.txt"
    
    if [[ ! -f "$cfg_file" ]]; then
        log_error "配置不存在: ${type}/${name}"
        log_info "可用配置:"
        "$SCRIPT_DIR/list_configs.sh" --${type}
        exit 1
    fi
}

# 读取模块配置内容
read_module_content() {
    local cfg_file="$1"
    grep -v "^#" "$cfg_file" | grep -v "^$" | head -20
}

# 创建模块化实验配置
create_modular_experiment() {
    log_step "创建模块化实验配置..."
    
    # 检查必需参数
    if [[ -z "$LOSS_CFG" ]] || [[ -z "$AUG_CFG" ]] || [[ -z "$TRAIN_CFG" ]]; then
        log_error "使用模块化配置时，--loss, --augment, --training 均为必需参数"
        exit 1
    fi
    
    # 检查配置
    check_module "loss" "$LOSS_CFG"
    check_module "augment" "$AUG_CFG"
    check_module "training" "$TRAIN_CFG"
    
    # 生成实验ID (如果没有提供)
    if [[ -z "$EXP_ID" ]]; then
        EXP_COUNT=$(grep -c "^## EXP_ID:" "$EXPERIMENTS_FILE" 2>/dev/null || echo "0")
        EXP_ID="M${EXP_COUNT}"
    fi
    
    # 生成实验名称 (如果没有提供)
    if [[ -z "$EXP_NAME" ]]; then
        EXP_NAME="${LOSS_CFG}_${AUG_CFG}_${TRAIN_CFG}"
    fi
    
    log_info "实验ID: $EXP_ID"
    log_info "实验名称: $EXP_NAME"
    log_info "Loss: $LOSS_CFG"
    log_info "Augment: $AUG_CFG"
    log_info "Training: $TRAIN_CFG"
    
    # 读取配置内容
    LOSS_CONTENT=$(read_module_content "${MODULES_DIR}/loss/${LOSS_CFG}.txt")
    AUG_CONTENT=$(read_module_content "${MODULES_DIR}/augment/${AUG_CFG}.txt")
    TRAIN_CONTENT=$(read_module_content "${MODULES_DIR}/training/${TRAIN_CFG}.txt")
    
    # 创建实验配置文件
    mkdir -p "$EXPERIMENTS_DIR"
    EXP_CONFIG_FILE="${EXPERIMENTS_DIR}/${EXP_ID}_${EXP_NAME}.json"
    
    cat > "$EXP_CONFIG_FILE" << EOF
{
  "exp_id": "${EXP_ID}",
  "exp_name": "${EXP_NAME}",
  "timestamp": "$(date -Iseconds)",
  "baseline": "${BASELINE}",
  "modules": {
    "loss": "${LOSS_CFG}",
    "augmentation": "${AUG_CFG}",
    "training": "${TRAIN_CFG}"
  }
}
EOF
    
    log_success "实验配置已保存: $EXP_CONFIG_FILE"
    
    # 创建实验记录
    cat >> "$EXPERIMENTS_FILE" << EOF

---

## EXP_ID: $EXP_ID

### 基础信息
- **实验名称**: $EXP_NAME
- **日期**: $DATE
- **基线**: $BASELINE
- **修改组件**: Loss=${LOSS_CFG}, Augment=${AUG_CFG}, Training=${TRAIN_CFG}
- **状态**: ⏳ 待执行

### 模块配置

#### Loss 配置 (${LOSS_CFG})
\`\`\`
${LOSS_CONTENT}
\`\`\`

#### Augmentation 配置 (${AUG_CFG})
\`\`\`
${AUG_CONTENT}
\`\`\`

#### Training 配置 (${TRAIN_CFG})
\`\`\`
${TRAIN_CONTENT}
\`\`\`

### 实验结果
| Epoch | Train Loss | Val PSNR | Val SSIM | Epoch Time | Val Time |
|-------|------------|----------|----------|------------|----------|

### 与基线对比
| 指标 | $BASELINE (基线) | 本实验 | 差异 | 显著性 |
|------|------------------|--------|------|--------|
| PSNR @ 50ep | | | | |
| SSIM @ 50ep | | | | |
| 收敛速度 | | | | |
| 训练时间 | | | | |

### 定性分析
- **收敛速度**: [待评估]
- **训练稳定性**: [待评估]
- **视觉质量**: [待评估]

### 决策
- [ ] **保留**: [理由]
- [ ] **回滚**: [理由]
- [ ] **再验证**: [理由]

### 备注
- 配置文件: \`config/experiments/${EXP_ID}_${EXP_NAME}.json\`
- 快速启动: \`./quick_experiment.sh ${EXP_ID} ${LOSS_CFG} ${AUG_CFG} ${TRAIN_CFG}\`

EOF
    
    log_success "实验记录已创建: EXP_ID=$EXP_ID"
}

# 创建传统实验记录
create_traditional_experiment() {
    if [ $# -lt 2 ]; then
        echo "用法: $0 <实验ID> <实验名称> [基线ID]"
        echo "或: $0 --loss <cfg> --augment <cfg> --training <cfg> --name <名称>"
        exit 1
    fi
    
    cat >> "$EXPERIMENTS_FILE" << EOF

---

## EXP_ID: $EXP_ID

### 基础信息
- **实验名称**: $EXP_NAME
- **日期**: $DATE
- **基线**: $BASELINE
- **修改组件**: [待填写]

### 配置详情
\`\`\`python
# 与基线不同的配置
net_idx = "${EXP_ID,,}_$(echo $EXP_NAME | tr '[:upper:]' '[:lower:]' | tr ' ' '_')"
# TODO: 填写其他关键配置
\`\`\`

### 实验结果
| Epoch | Train Loss | Val PSNR | Val SSIM | Epoch Time | Val Time |
|-------|------------|----------|----------|------------|----------|
| 10 | | | | | |
| 20 | | | | | |
| 50 | | | | | |

### 与基线对比
| 指标 | $BASELINE (基线) | 本实验 | 差异 | 显著性 |
|------|------------------|--------|------|--------|
| PSNR @ 50ep | | | | |
| SSIM @ 50ep | | | | |
| 收敛速度 | | | | |
| 训练时间 | | | | |

### 定性分析
- **收敛速度**: [待评估]
- **训练稳定性**: [待评估]
- **视觉质量**: [待评估]

### 决策
- [ ] **保留**: [理由]
- [ ] **回滚**: [理由]
- [ ] **再验证**: [理由]

### 备注
- [待补充]

EOF

    log_success "实验记录已创建: EXP_ID=$EXP_ID"
}

# 主逻辑
if $USE_MODULES; then
    create_modular_experiment
else
    if [[ -z "$EXP_ID" ]] || [[ -z "$EXP_NAME" ]]; then
        log_error "请提供实验ID和名称，或使用 --help 查看帮助"
        exit 1
    fi
    create_traditional_experiment
fi

echo ""
log_info "文件: $EXPERIMENTS_FILE"
echo ""
echo "下一步:"
echo "1. 编辑 EXPERIMENTS.md 填写配置详情 (如需要)"
if $USE_MODULES; then
    echo "2. 启动实验: ./quick_experiment.sh ${EXP_ID} ${LOSS_CFG} ${AUG_CFG} ${TRAIN_CFG}"
else
    echo "2. 执行实验"
fi
echo "3. 更新实验结果"
