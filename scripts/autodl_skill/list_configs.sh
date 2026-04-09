#!/bin/bash
#===============================================================================
# 列出所有可用配置模板
# 用法: ./list_configs.sh [选项]
# 选项:
#   --loss       只显示 Loss 配置
#   --augment    只显示 Augmentation 配置
#   --training   只显示 Training 配置
#   --json       以 JSON 格式输出
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULES_DIR="${SCRIPT_DIR}/../../config/modules"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 参数解析
SHOW_ALL=true
SHOW_LOSS=false
SHOW_AUGMENT=false
SHOW_TRAINING=false
OUTPUT_JSON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --loss)
            SHOW_ALL=false
            SHOW_LOSS=true
            shift
            ;;
        --augment)
            SHOW_ALL=false
            SHOW_AUGMENT=true
            shift
            ;;
        --training)
            SHOW_ALL=false
            SHOW_TRAINING=true
            shift
            ;;
        --json)
            OUTPUT_JSON=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --loss       只显示 Loss 配置"
            echo "  --augment    只显示 Augmentation 配置"
            echo "  --training   只显示 Training 配置"
            echo "  --json       以 JSON 格式输出"
            echo "  -h, --help   显示帮助"
            echo ""
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 获取模块列表
get_modules() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        ls -1 "$dir"/*.txt 2>/dev/null | xargs -n1 basename | sed 's/\.txt$//' | sort
    fi
}

# 获取模块详情
get_module_detail() {
    local file="$1"
    local desc=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^# ]] && [[ ! "$line" =~ ^### ]]; then
            desc=$(echo "$line" | sed 's/^# *//')
            if [[ -n "$desc" ]]; then
                echo "$desc"
                return
            fi
        fi
    done < "$file"
    echo "暂无描述"
}

# JSON 输出
if $OUTPUT_JSON; then
    echo "{"
    echo '  "modules": {'
    
    # Loss modules
    if $SHOW_ALL || $SHOW_LOSS; then
        echo '    "loss": ['
        first=true
        for f in "$MODULES_DIR"/loss/*.txt; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .txt)
            desc=$(get_module_detail "$f")
            if $first; then
                first=false
            else
                echo ","
            fi
            echo -n "      {\"name\": \"$name\", \"description\": \"$desc\"}"
        done
        echo ""
        echo '    ],'
    fi
    
    # Augment modules
    if $SHOW_ALL || $SHOW_AUGMENT; then
        echo '    "augment": ['
        first=true
        for f in "$MODULES_DIR"/augment/*.txt; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .txt)
            desc=$(get_module_detail "$f")
            if $first; then
                first=false
            else
                echo ","
            fi
            echo -n "      {\"name\": \"$name\", \"description\": \"$desc\"}"
        done
        echo ""
        echo '    ],'
    fi
    
    # Training modules
    if $SHOW_ALL || $SHOW_TRAINING; then
        echo '    "training": ['
        first=true
        for f in "$MODULES_DIR"/training/*.txt; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .txt)
            desc=$(get_module_detail "$f")
            if $first; then
                first=false
            else
                echo ","
            fi
            echo -n "      {\"name\": \"$name\", \"description\": \"$desc\"}"
        done
        echo ""
        echo '    ]'
    fi
    
    echo '  }'
    echo "}"
    exit 0
fi

# 标准输出
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         RPLHR-CT 模块化配置模板列表                      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Loss 配置
if $SHOW_ALL || $SHOW_LOSS; then
    echo -e "${BLUE}📊 Loss 配置:${NC}"
    for f in "$MODULES_DIR"/loss/*.txt; do
        [[ -f "$f" ]] || continue
        name=$(basename "$f" .txt)
        desc=$(get_module_detail "$f")
        printf "  ${GREEN}%-15s${NC} %s\n" "$name" "$desc"
    done
    echo ""
fi

# Augmentation 配置
if $SHOW_ALL || $SHOW_AUGMENT; then
    echo -e "${BLUE}🔄 Augmentation 配置:${NC}"
    for f in "$MODULES_DIR"/augment/*.txt; do
        [[ -f "$f" ]] || continue
        name=$(basename "$f" .txt)
        desc=$(get_module_detail "$f")
        printf "  ${GREEN}%-15s${NC} %s\n" "$name" "$desc"
    done
    echo ""
fi

# Training 配置
if $SHOW_ALL || $SHOW_TRAINING; then
    echo -e "${BLUE}⚙️  Training 配置:${NC}"
    for f in "$MODULES_DIR"/training/*.txt; do
        [[ -f "$f" ]] || continue
        name=$(basename "$f" .txt)
        desc=$(get_module_detail "$f")
        printf "  ${GREEN}%-15s${NC} %s\n" "$name" "$desc"
    done
    echo ""
fi

# 使用提示
echo -e "${YELLOW}💡 使用示例:${NC}"
echo ""
echo "  # 查看特定类型配置"
echo "  $0 --loss"
echo ""
echo "  # 快速启动实验"
echo "  ./quick_experiment.sh exp01 l1 flip adamw_ema"
echo ""
echo "  # 创建带配置的实验"
echo "  ./create_experiment.sh --loss eagle3d --augment combined --training advanced"
echo ""
