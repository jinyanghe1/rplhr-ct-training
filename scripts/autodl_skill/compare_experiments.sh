#!/bin/bash
#===============================================================================
# 实验对比报告生成 (支持模块化配置对比)
# 改进: 自动收集实验数据、性能对比图表、模块配置差异分析
# 
# 用法: ./compare_experiments.sh [选项]
#
# 选项:
#   --report, -r           生成完整Markdown报告
#   --output, -o FILE      指定输出文件
#   --compare-modules      对比模块配置
#   --exp1 <name>          实验1名称
#   --exp2 <name>          实验2名称
#   --help, -h             显示帮助
#===============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_FILE="$SCRIPT_DIR/EXPERIMENTS.md"
ROADMAP_FILE="$SCRIPT_DIR/ROADMAP.md"
RESULTS_DIR="$SCRIPT_DIR/logs"
EXPERIMENTS_DIR="${SCRIPT_DIR}/../../config/experiments"
MODULES_DIR="${SCRIPT_DIR}/../../config/modules"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_highlight() { echo -e "${CYAN}$1${NC}"; }
log_section() { echo -e "${MAGENTA}$1${NC}"; }

# 解析参数
GENERATE_REPORT=false
OUTPUT_FILE=""
COMPARE_MODULES=false
EXP1=""
EXP2=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --report|-r)
            GENERATE_REPORT=true
            shift
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --compare-modules)
            COMPARE_MODULES=true
            shift
            ;;
        --exp1)
            EXP1="$2"
            shift 2
            ;;
        --exp2)
            EXP2="$2"
            shift 2
            ;;
        --help|-h)
            echo "实验对比报告生成"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -r, --report           生成完整Markdown报告"
            echo "  -o, --output FILE      指定输出文件"
            echo "  --compare-modules      对比两个实验的模块配置"
            echo "  --exp1 <name>          实验1名称 (用于模块对比)"
            echo "  --exp2 <name>          实验2名称 (用于模块对比)"
            echo "  -h, --help             显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                     # 快速概览"
            echo "  $0 -r                  # 生成完整报告"
            echo "  $0 --compare-modules --exp1 exp1 --exp2 exp2"
            echo ""
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

#===============================================================================
# 解析实验数据
#===============================================================================

parse_experiments() {
    local file="$1"
    
    if [[ ! -f "$file" ]]; then
        return 1
    fi
    
    # 提取完成的实验
    grep -B5 -A10 "状态.*完成\|状态.*✅" "$file" 2>/dev/null | awk '
        /^## EXP_ID:/ { exp_id=$3 }
        /^- 架构:/ { arch=$3 }
        /^- Loss:/ { loss=$3 }
        /^- Epoch:/ { epoch=$3 }
        /^- 最佳 PSNR:/ { psnr=$4 }
        /^- 最佳 SSIM:/ { ssim=$4 }
        /---/ && exp_id { print exp_id "|" arch "|" loss "|" epoch "|" psnr "|" ssim; exp_id="" }
    '
}

#===============================================================================
# 从JSON文件收集实验数据
#===============================================================================

collect_from_json() {
    local data=""
    
    for json_file in "$RESULTS_DIR"/*/run_*/metrics.json; do
        [[ -f "$json_file" ]] || continue
        
        local dir_name
        dir_name=$(dirname "$json_file")
        
        local exp_id psnr ssim epoch
        exp_id=$(basename "$dir_name" | sed 's/run_//')
        psnr=$(grep -oE '"best_psnr": "[^"]+"' "$json_file" | cut -d'"' -f4)
        ssim=$(grep -oE '"best_ssim": "[^"]+"' "$json_file" | cut -d'"' -f4)
        epoch=$(grep -oE '"best_epoch": "[^"]+"' "$json_file" | cut -d'"' -f4)
        
        if [[ -n "$psnr" ]]; then
            data="${data}${exp_id}|unknown|unknown|${epoch}|${psnr}|${ssim}\n"
        fi
    done
    
    # 也检查实验配置目录
    for exp_config in "$EXPERIMENTS_DIR"/*.json; do
        [[ -f "$exp_config" ]] || continue
        
        local exp_name psnr ssim
        exp_name=$(basename "$exp_config" .json)
        
        # 尝试从结果目录找到对应数据
        for result_dir in "$RESULTS_DIR"/*/; do
            [[ -d "$result_dir" ]] || continue
            local metrics_file="${result_dir}metrics.json"
            if [[ -f "$metrics_file" ]]; then
                local exp_in_file
                exp_in_file=$(grep -oE '"exp_name": "[^"]+"' "$metrics_file" | cut -d'"' -f4)
                if [[ "$exp_in_file" == "$exp_name" ]]; then
                    psnr=$(grep -oE '"psnr": "[^"]+"' "$metrics_file" | cut -d'"' -f4)
                    ssim=$(grep -oE '"ssim": "[^"]+"' "$metrics_file" | cut -d'"' -f4)
                    data="${data}${exp_name}|modular|modular|-|${psnr}|${ssim}\n"
                    break
                fi
            fi
        done
    done
    
    echo -e "$data"
}

#===============================================================================
# 对比两个实验的模块配置
#===============================================================================

compare_module_configs() {
    if [[ -z "$EXP1" ]] || [[ -z "$EXP2" ]]; then
        log_error "使用 --compare-modules 时需要指定 --exp1 和 --exp2"
        exit 1
    fi
    
    local exp1_file="${EXPERIMENTS_DIR}/${EXP1}.json"
    local exp2_file="${EXPERIMENTS_DIR}/${EXP2}.json"
    
    if [[ ! -f "$exp1_file" ]]; then
        log_error "实验配置不存在: $EXP1"
        exit 1
    fi
    
    if [[ ! -f "$exp2_file" ]]; then
        log_error "实验配置不存在: $EXP2"
        exit 1
    fi
    
    echo ""
    log_highlight "╔══════════════════════════════════════════════════╗"
    log_highlight "║        模块配置差异对比                          ║"
    log_highlight "╚══════════════════════════════════════════════════╝"
    echo ""
    
    # 获取模块配置
    local exp1_modules exp2_modules
    exp1_modules=$(python3 -c "import sys,json; d=json.load(open('$exp1_file')); print(json.dumps(d.get('modules',{})))" 2>/dev/null || echo '{}')
    exp2_modules=$(python3 -c "import sys,json; d=json.load(open('$exp2_file')); print(json.dumps(d.get('modules',{})))" 2>/dev/null || echo '{}')
    
    echo -e "${BLUE}实验 1:${NC} $EXP1"
    echo "$exp1_modules" | python3 -m json.tool 2>/dev/null || echo "$exp1_modules"
    echo ""
    
    echo -e "${BLUE}实验 2:${NC} $EXP2"
    echo "$exp2_modules" | python3 -m json.tool 2>/dev/null || echo "$exp2_modules"
    echo ""
    
    # 详细差异
    log_section "详细差异:"
    echo ""
    
    for module_type in loss augmentation training; do
        local mod1 mod2
        mod1=$(echo "$exp1_modules" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$module_type','N/A'))")
        mod2=$(echo "$exp2_modules" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$module_type','N/A'))")
        
        if [[ "$mod1" != "$mod2" ]]; then
            echo -e "${YELLOW}${module_type}:${NC}"
            echo "  实验1: $mod1"
            echo "  实验2: $mod2"
            echo ""
            
            # 读取具体配置内容对比
            if [[ -f "${MODULES_DIR}/${module_type}/${mod1}.txt" ]] && [[ -f "${MODULES_DIR}/${module_type}/${mod2}.txt" ]]; then
                echo "  配置差异:"
                diff -u "${MODULES_DIR}/${module_type}/${mod1}.txt" "${MODULES_DIR}/${module_type}/${mod2}.txt" | grep -E '^[+-]' | grep -v '^[+-]{3}' | sed 's/^/    /'
                echo ""
            fi
        fi
    done
}

#===============================================================================
# 生成对比表格
#===============================================================================

generate_comparison_table() {
    local data="$1"
    
    echo "### 实验性能对比表"
    echo ""
    echo "| 实验ID | 架构 | Loss函数 | Epoch | PSNR (dB) | SSIM | 评级 |"
    echo "|--------|------|----------|-------|-----------|------|------|"
    
    # 排序并显示
    echo -e "$data" | sort -t'|' -k5 -rn | while IFS='|' read -r exp_id arch loss epoch psnr ssim; do
        [[ -z "$exp_id" ]] && continue
        
        # 评级
        local rating=""
        if [[ "$psnr" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            if (( $(echo "$psnr >= 40" | bc -l 2>/dev/null || echo "0") )); then
                rating="⭐⭐⭐"
            elif (( $(echo "$psnr >= 35" | bc -l 2>/dev/null || echo "0") )); then
                rating="⭐⭐"
            else
                rating="⭐"
            fi
        else
            rating="-"
        fi
        
        printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
            "$exp_id" "${arch:--}" "${loss:--}" "${epoch:--}" "${psnr:--}" "${ssim:--}" "$rating"
    done
    
    echo ""
}

#===============================================================================
# 生成性能排名
#===============================================================================

generate_ranking() {
    local data="$1"
    
    echo "### 🏆 性能排名"
    echo ""
    
    # 按PSNR排序
    local sorted_data
    sorted_data=$(echo -e "$data" | grep -v '^$' | sort -t'|' -k5 -rn)
    
    # 显示Top 3
    echo "**Top 3 (按 PSNR):**"
    echo ""
    
    local rank=1
    echo "$sorted_data" | while IFS='|' read -r exp_id arch loss epoch psnr ssim && [[ $rank -le 3 ]]; do
        [[ -z "$exp_id" ]] && continue
        
        local medal
        case $rank in
            1) medal="🥇" ;;
            2) medal="🥈" ;;
            3) medal="🥉" ;;
        esac
        
        echo "${medal} **#${rank}** ${exp_id}: PSNR=${psnr} dB, SSIM=${ssim}"
        rank=$((rank + 1))
    done
    
    echo ""
}

#===============================================================================
# 生成趋势分析
#===============================================================================

generate_trend_analysis() {
    local data="$1"
    
    echo "### 📈 趋势分析"
    echo ""
    
    # 提取所有PSNR值
    local psnr_values
    psnr_values=$(echo -e "$data" | grep -v '^$' | cut -d'|' -f5 | grep -E '^[0-9]+(\.[0-9]+)?$')
    
    if [[ $(echo "$psnr_values" | wc -l) -ge 2 ]]; then
        local best worst avg
        best=$(echo "$psnr_values" | sort -n | tail -1)
        worst=$(echo "$psnr_values" | sort -n | head -1)
        avg=$(echo "$psnr_values" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count}')
        
        echo "- **最佳 PSNR**: ${best} dB"
        echo "- **最差 PSNR**: ${worst} dB"
        echo "- **平均 PSNR**: ${avg} dB"
        echo "- **差距**: $(echo "scale=2; $best - $worst" | bc 2>/dev/null || echo "N/A") dB"
    else
        echo "数据不足，无法分析趋势"
    fi
    
    echo ""
}

#===============================================================================
# 生成可视化ASCII图表
#===============================================================================

generate_ascii_chart() {
    local data="$1"
    
    echo "### 📊 PSNR 可视化"
    echo ""
    echo "\`\`\`"
    
    # 获取PSNR范围和排序后的数据
    local psnr_values max_psnr min_psnr
    psnr_values=$(echo -e "$data" | grep -v '^$' | cut -d'|' -f5 | grep -E '^[0-9]+(\.[0-9]+)?$')
    max_psnr=$(echo "$psnr_values" | sort -n | tail -1)
    min_psnr=$(echo "$psnr_values" | sort -n | head -1)
    
    # 添加一些边距
    max_psnr=$(echo "scale=2; $max_psnr + 2" | bc 2>/dev/null || echo "$max_psnr")
    min_psnr=$(echo "scale=2; $min_psnr - 2" | bc 2>/dev/null || echo "$min_psnr")
    
    echo "$data" | grep -v '^$' | sort -t'|' -k5 -rn | while IFS='|' read -r exp_id arch loss epoch psnr ssim; do
        [[ -z "$exp_id" ]] && continue
        [[ ! "$psnr" =~ ^[0-9]+(\.[0-9]+)?$ ]] && continue
        
        # 计算条形长度
        local bar_len
        bar_len=$(echo "scale=0; ($psnr - $min_psnr) * 40 / ($max_psnr - $min_psnr)" | bc 2>/dev/null || echo "0")
        [[ "$bar_len" -lt 0 ]] && bar_len=0
        [[ "$bar_len" -gt 40 ]] && bar_len=40
        
        printf "%-12s " "$exp_id"
        for ((i=0; i<bar_len; i++)); do echo -n "█"; done
        echo " ${psnr} dB"
    done
    
    echo "\`\`\`"
    echo ""
}

#===============================================================================
# 统计实验状态
#===============================================================================

stat_experiments() {
    local file="$1"
    
    if [[ ! -f "$file" ]]; then
        echo "完成: 0 | 失败: 0 | 待执行: 0"
        return
    fi
    
    local completed failed pending
    completed=$(grep -c "状态.*完成\|状态.*✅" "$file" 2>/dev/null || echo "0")
    failed=$(grep -c "状态.*失败\|状态.*❌" "$file" 2>/dev/null || echo "0")
    pending=$(grep -c "状态.*待执行\|状态.*⏳" "$file" 2>/dev/null || echo "0")
    
    echo "完成: $completed | 失败: $failed | 待执行: $pending"
}

#===============================================================================
# 生成完整报告
#===============================================================================

generate_full_report() {
    local output_file="${1:-$SCRIPT_DIR/EXPERIMENT_COMPARISON_REPORT.md}"
    
    log_info "正在生成实验对比报告..."
    
    # 收集数据
    local exp_data
    exp_data=$(parse_experiments "$EXPERIMENTS_FILE")
    
    # 如果没有实验数据，尝试从JSON收集
    if [[ -z "$exp_data" ]]; then
        exp_data=$(collect_from_json)
    fi
    
    {
        echo "# RPLHR-CT 实验对比报告"
        echo ""
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        echo "## 📊 实验统计"
        echo ""
        echo "| 来源 | 状态 |"
        echo "|------|------|"
        echo "| EXPERIMENTS.md | $(stat_experiments "$EXPERIMENTS_FILE") |"
        echo "| 本地日志 | $(ls -d "$RESULTS_DIR"/*/run_* 2>/dev/null | wc -l) 个实验 |"
        echo "| 实验配置 | $(ls -1 "$EXPERIMENTS_DIR"/*.json 2>/dev/null | wc -l) 个配置 |"
        echo ""
        
        if [[ -n "$exp_data" ]]; then
            generate_ranking "$exp_data"
            generate_comparison_table "$exp_data"
            generate_trend_analysis "$exp_data"
            generate_ascii_chart "$exp_data"
        else
            echo "## 暂无实验数据"
            echo ""
            echo "请运行实验后再次生成报告。"
            echo ""
        fi
        
        # 添加模块配置说明
        echo "## 🔧 可用模块配置"
        echo ""
        echo "### Loss 模块"
        echo ""
        for f in "$MODULES_DIR"/loss/*.txt; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .txt)
            desc=$(grep "^#" "$f" | head -1 | sed 's/^# *//')
            echo "- **$name**: ${desc:-暂无描述}"
        done
        echo ""
        
        echo "### Augmentation 模块"
        echo ""
        for f in "$MODULES_DIR"/augment/*.txt; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .txt)
            desc=$(grep "^#" "$f" | head -1 | sed 's/^# *//')
            echo "- **$name**: ${desc:-暂无描述}"
        done
        echo ""
        
        echo "### Training 模块"
        echo ""
        for f in "$MODULES_DIR"/training/*.txt; do
            [[ -f "$f" ]] || continue
            name=$(basename "$f" .txt)
            desc=$(grep "^#" "$f" | head -1 | sed 's/^# *//')
            echo "- **$name**: ${desc:-暂无描述}"
        done
        echo ""
        
        echo "## 📋 待办事项"
        echo ""
        
        if [[ -f "$ROADMAP_FILE" ]]; then
            echo "### P0 - 立即执行"
            grep -A3 "P0 - 立即执行" "$ROADMAP_FILE" 2>/dev/null | grep "^- \[ \]" | head -3 | sed 's/^- \[ \]/- /'
            echo ""
            echo "### P1 - 本周完成"
            grep -A3 "P1 - 本周完成" "$ROADMAP_FILE" 2>/dev/null | grep "^- \[ \]" | head -3 | sed 's/^- \[ \]/- /'
        fi
        
        echo ""
        echo "---"
        echo ""
        echo "## 快捷命令"
        echo ""
        echo "\`\`\`bash"
        echo "# 创建新实验 (模块化)"
        echo "./create_experiment.sh --loss eagle3d --augment flip --training adamw_ema --name exp01"
        echo ""
        echo "# 快速启动实验"
        echo "./quick_experiment.sh exp01 l1 flip adamw_ema 50"
        echo ""
        echo "# 列出可用配置"
        echo "./list_configs.sh"
        echo ""
        echo "# 对比实验配置"
        echo "./compare_experiments.sh --compare-modules --exp1 exp1 --exp2 exp2"
        echo ""
        echo "# 检查训练状态"
        echo "./check_training.sh"
        echo ""
        echo "# 收集结果"
        echo "./collect_results.sh"
        echo "\`\`\`"
        
    } > "$output_file"
    
    log_info "报告已生成: $output_file"
}

#===============================================================================
# 快速概览模式
#===============================================================================

quick_overview() {
    echo ""
    log_highlight "╔══════════════════════════════════════════════════╗"
    log_highlight "║           RPLHR-CT 实验对比报告                  ║"
    log_highlight "╚══════════════════════════════════════════════════╝"
    echo ""
    
    # 统计
    echo -e "${BLUE}📊 实验统计:${NC}"
    echo "  EXPERIMENTS.md: $(stat_experiments "$EXPERIMENTS_FILE")"
    echo "  本地日志: $(ls -d "$RESULTS_DIR"/*/run_* 2>/dev/null | wc -l) 个实验"
    echo "  实验配置: $(ls -1 "$EXPERIMENTS_DIR"/*.json 2>/dev/null | wc -l) 个配置"
    echo ""
    
    # 显示模块配置概览
    echo -e "${BLUE}🔧 可用模块:${NC}"
    echo "  Loss: $(ls -1 "$MODULES_DIR"/loss/*.txt 2>/dev/null | wc -l) 个"
    echo "  Augmentation: $(ls -1 "$MODULES_DIR"/augment/*.txt 2>/dev/null | wc -l) 个"
    echo "  Training: $(ls -1 "$MODULES_DIR"/training/*.txt 2>/dev/null | wc -l) 个"
    echo ""
    
    # 显示完成的实验
    if [[ -f "$EXPERIMENTS_FILE" ]]; then
        local completed
        completed=$(grep -B3 "状态.*完成\|状态.*✅" "$EXPERIMENTS_FILE" 2>/dev/null | grep "EXP_ID" | sed 's/.*EXP_ID: //')
        
        if [[ -n "$completed" ]]; then
            echo -e "${GREEN}✅ 已完成实验:${NC}"
            echo "$completed" | head -5 | sed 's/^/  • /'
            echo ""
        fi
        
        # 显示待执行实验
        local pending
        pending=$(grep -B3 "状态.*待执行\|状态.*⏳" "$EXPERIMENTS_FILE" 2>/dev/null | grep "EXP_ID" | sed 's/.*EXP_ID: //')
        
        if [[ -n "$pending" ]]; then
            echo -e "${YELLOW}⏳ 待执行实验:${NC}"
            echo "$pending" | head -5 | sed 's/^/  • /'
            echo ""
        fi
    fi
    
    # 显示性能数据
    local exp_data
    exp_data=$(parse_experiments "$EXPERIMENTS_FILE")
    
    if [[ -z "$exp_data" ]]; then
        exp_data=$(collect_from_json)
    fi
    
    if [[ -n "$exp_data" ]]; then
        echo -e "${CYAN}🏆 最佳性能 (Top 3):${NC}"
        echo -e "$exp_data" | sort -t'|' -k5 -rn | head -3 | while IFS='|' read -r exp_id arch loss epoch psnr ssim; do
            printf "  %-10s PSNR: %6s dB  SSIM: %s\n" "$exp_id" "$psnr" "$ssim"
        done
        echo ""
    fi
    
    # 下一步行动
    echo -e "${MAGENTA}📝 下一步行动:${NC}"
    if [[ -f "$ROADMAP_FILE" ]]; then
        grep -A3 "P0 - 立即执行" "$ROADMAP_FILE" 2>/dev/null | grep "^- \[ \]" | head -2 | sed 's/^- \[ \]/  •/'
    fi
    echo ""
    
    # 快捷命令提示
    echo -e "${YELLOW}💡 快捷命令:${NC}"
    echo "  ./quick_experiment.sh exp01 l1 flip adamw_ema 50"
    echo "  ./list_configs.sh"
    echo ""
}

#===============================================================================
# 入口
#===============================================================================

if $COMPARE_MODULES; then
    compare_module_configs
elif $GENERATE_REPORT; then
    generate_full_report "${OUTPUT_FILE:-$SCRIPT_DIR/EXPERIMENT_COMPARISON_REPORT.md}"
else
    quick_overview
fi
