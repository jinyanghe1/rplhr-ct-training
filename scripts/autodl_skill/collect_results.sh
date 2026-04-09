#!/bin/bash
#===============================================================================
# 收集 AutoDL 训练结果到本地 - 增强版 v2.0
# 改进: 自动收集日志、提取关键指标、生成可视化
#===============================================================================

set -o pipefail

# 加载工具库
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/lib_ssh.sh"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#===============================================================================
# 提取指标 - 兼容不同格式
#===============================================================================

extract_metrics() {
    local log_file="$1"
    local metric_type="$2"
    
    case "$metric_type" in
        psnr)
            # 支持多种格式: "PSNR: 35.42", "Val PSNR: 35.42", "Best PSNR: 35.42"
            grep -iE "PSNR" "$log_file" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1
            ;;
        ssim)
            grep -iE "SSIM" "$log_file" 2>/dev/null | grep -oE "0\.[0-9]+" | tail -1
            ;;
        loss)
            grep -iE "Loss" "$log_file" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1
            ;;
        epoch)
            grep -iE "Epoch [0-9]+" "$log_file" 2>/dev/null | grep -oE "[0-9]+" | tail -1
            ;;
    esac
}

#===============================================================================
# 解析metrics.csv
#===============================================================================

parse_metrics_csv() {
    local csv_file="$1"
    
    if [[ ! -f "$csv_file" || ! -s "$csv_file" ]]; then
        return 1
    fi
    
    # 获取最佳指标
    local best_psnr best_epoch best_ssim
    best_psnr=$(tail -n +2 "$csv_file" | cut -d',' -f4 | grep -v '^$' | sort -n | tail -1)
    best_epoch=$(grep ",$best_psnr," "$csv_file" | head -1 | cut -d',' -f1)
    best_ssim=$(grep ",$best_psnr," "$csv_file" | head -1 | cut -d',' -f5)
    
    # 获取最新指标
    local latest_epoch latest_loss latest_psnr latest_ssim
    latest_epoch=$(tail -1 "$csv_file" | cut -d',' -f1)
    latest_loss=$(tail -1 "$csv_file" | cut -d',' -f3)
    latest_psnr=$(tail -1 "$csv_file" | cut -d',' -f4)
    latest_ssim=$(tail -1 "$csv_file" | cut -d',' -f5)
    
    echo "best_psnr:$best_psnr|best_epoch:$best_epoch|best_ssim:$best_ssim|latest_epoch:$latest_epoch|latest_loss:$latest_loss|latest_psnr:$latest_psnr|latest_ssim:$latest_ssim"
}

#===============================================================================
# 下载远程日志
#===============================================================================

download_logs() {
    local run_dir="$1"
    local download_count=0
    
    log_info "下载训练日志..."
    
    # 1. 下载最新的5个日志文件
    local log_files
    log_files=$(ssh_run_with_retry "ls -t $AUTODL_REPO_PATH/train_autodl_*.log 2>/dev/null | head -5" 5 2)
    
    if [[ -n "$log_files" && "$log_files" != *"No such"* ]]; then
        while IFS= read -r log_file; do
            [[ -z "$log_file" ]] && continue
            
            local filename
            filename=$(basename "$log_file")
            log_info "  下载: $filename"
            
            if ssh_run_with_retry "cat '$log_file'" 30 2 > "$run_dir/$filename" 2>/dev/null; then
                download_count=$((download_count + 1))
            else
                log_warn "  下载失败: $filename"
            fi
        done <<< "$log_files"
    fi
    
    # 2. 尝试下载train_autodl.log（默认日志）
    local default_log="$AUTODL_REPO_PATH/train_autodl.log"
    if ssh_run_with_retry "test -f $default_log" 1 &>/dev/null; then
        log_info "  下载: train_autodl.log"
        if ssh_run_with_retry "cat $default_log" 30 2 > "$run_dir/train_autodl.log" 2>/dev/null; then
            download_count=$((download_count + 1))
        fi
    fi
    
    # 3. 下载daemon日志
    local daemon_log="$AUTODL_REPO_PATH/train_daemon.log"
    if ssh_run_with_retry "test -f $daemon_log" 1 &>/dev/null; then
        log_info "  下载: train_daemon.log"
        ssh_run_with_retry "cat $daemon_log" 30 2 > "$run_dir/train_daemon.log" 2>/dev/null || true
    fi
    
    echo "$download_count"
}

#===============================================================================
# 下载模型和检查点
#===============================================================================

download_checkpoints() {
    local run_dir="$1"
    
    log_info "下载检查点..."
    
    local checkpoint_dir="$AUTODL_REPO_PATH/checkpoints/dataset01_xuanwu/xuanwu_ratio4"
    
    # 下载metrics.csv
    if ssh_run_with_retry "test -f $checkpoint_dir/metrics.csv" 1 &>/dev/null; then
        log_info "  下载: metrics.csv"
        ssh_run_with_retry "cat $checkpoint_dir/metrics.csv" 30 2 > "$run_dir/metrics.csv" 2>/dev/null || true
    fi
    
    # 列出可用的模型文件
    local model_files
    model_files=$(ssh_run_with_retry "ls -la $checkpoint_dir/*.pth 2>/dev/null || echo 'No models'" 10 1)
    
    if [[ -n "$model_files" && "$model_files" != *"No models"* ]]; then
        log_info "  可用模型:"
        echo "$model_files" | sed 's/^/    /'
    fi
}

#===============================================================================
# 生成可视化图表（ASCII格式）
#===============================================================================

generate_visualization() {
    local csv_file="$1"
    local output_file="$2"
    
    if [[ ! -f "$csv_file" || ! -s "$csv_file" ]]; then
        return 1
    fi
    
    {
        echo "## 训练曲线"
        echo ""
        
        # PSNR趋势
        echo "### PSNR 趋势"
        echo "\`\`\`"
        tail -n +2 "$csv_file" | cut -d',' -f1,4 | while IFS=',' read -r epoch psnr; do
            [[ -z "$psnr" ]] && continue
            local bar_len
            bar_len=$(echo "scale=0; $psnr * 2 / 1" | bc 2>/dev/null || echo "0")
            printf "Epoch %3s: " "$epoch"
            for ((i=0; i<bar_len && i<50; i++)); do echo -n "█"; done
            echo " $psnr"
        done
        echo "\`\`\`"
        echo ""
        
        # Loss趋势
        echo "### Loss 趋势"
        echo "\`\`\`"
        tail -n +2 "$csv_file" | cut -d',' -f1,3 | while IFS=',' read -r epoch loss; do
            [[ -z "$loss" || "$loss" == "loss" ]] && continue
            # 归一化loss用于显示 (假设loss在0-1之间)
            local bar_len
            bar_len=$(echo "scale=0; (1 - $loss) * 40 / 1" | bc 2>/dev/null || echo "0")
            [[ "$bar_len" -lt 0 ]] && bar_len=0
            [[ "$bar_len" -gt 40 ]] && bar_len=40
            printf "Epoch %3s: " "$epoch"
            for ((i=0; i<bar_len; i++)); do echo -n "█"; done
            echo " $loss"
        done
        echo "\`\`\`"
    } >> "$output_file"
}

#===============================================================================
# 生成Markdown报告
#===============================================================================

generate_report() {
    local run_dir="$1"
    local metrics_str="$2"
    
    log_info "生成报告..."
    
    # 解析指标
    local best_psnr best_epoch best_ssim latest_epoch latest_loss latest_psnr latest_ssim
    best_psnr=$(echo "$metrics_str" | grep -oE 'best_psnr:[^|]+' | cut -d':' -f2)
    best_epoch=$(echo "$metrics_str" | grep -oE 'best_epoch:[^|]+' | cut -d':' -f2)
    best_ssim=$(echo "$metrics_str" | grep -oE 'best_ssim:[^|]+' | cut -d':' -f2)
    latest_epoch=$(echo "$metrics_str" | grep -oE 'latest_epoch:[^|]+' | cut -d':' -f2)
    latest_loss=$(echo "$metrics_str" | grep -oE 'latest_loss:[^|]+' | cut -d':' -f2)
    latest_psnr=$(echo "$metrics_str" | grep -oE 'latest_psnr:[^|]+' | cut -d':' -f2)
    latest_ssim=$(echo "$metrics_str" | grep -oE 'latest_ssim:[^|]+' | cut -d':' -f2)
    
    local report_file="$run_dir/REPORT.md"
    
    {
        echo "# 训练结果报告"
        echo ""
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        echo "## 📊 性能指标"
        echo ""
        echo "### 最佳表现"
        echo "| 指标 | 值 | Epoch |"
        echo "|------|------|-------|"
        echo "| PSNR | ${best_psnr:-N/A} dB | ${best_epoch:-N/A} |"
        echo "| SSIM | ${best_ssim:-N/A} | ${best_epoch:-N/A} |"
        echo ""
        echo "### 最新结果 (Epoch ${latest_epoch:-N/A})"
        echo "| 指标 | 值 |"
        echo "|------|------|"
        echo "| PSNR | ${latest_psnr:-N/A} dB |"
        echo "| SSIM | ${latest_ssim:-N/A} |"
        echo "| Loss | ${latest_loss:-N/A} |"
        echo ""
        
        # 添加可视化
        if [[ -f "$run_dir/metrics.csv" ]]; then
            generate_visualization "$run_dir/metrics.csv" "$report_file"
        fi
        
        echo "## 📁 文件列表"
        echo ""
        ls -lh "$run_dir" | tail -n +2 | awk '{printf "- %s (%s)\n", $9, $5}'
        echo ""
        
        echo "## 📝 日志摘要"
        echo ""
        echo "\`\`\`"
        # 从最新的日志文件提取最后30行
        local latest_log
        latest_log=$(ls -t "$run_dir"/train_autodl*.log 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            tail -30 "$latest_log" 2>/dev/null
        else
            echo "无日志文件"
        fi
        echo "\`\`\`"
        
    } > "$report_file"
    
    echo "$report_file"
}

#===============================================================================
# 主收集流程
#===============================================================================

collect() {
    local run_dir="$LOCAL_LOG_DIR/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"
    
    log_info "开始收集结果到: $run_dir"
    
    # 1. 下载日志
    local log_count
    log_count=$(download_logs "$run_dir")
    log_info "已下载 $log_count 个日志文件"
    
    # 2. 下载检查点
    download_checkpoints "$run_dir"
    
    # 3. 解析指标
    local metrics_str=""
    if [[ -f "$run_dir/metrics.csv" ]]; then
        metrics_str=$(parse_metrics_csv "$run_dir/metrics.csv")
        log_info "解析metrics: $metrics_str"
    else
        # 从日志提取
        local latest_log
        latest_log=$(ls -t "$run_dir"/train_autodl*.log 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            local psnr ssim loss epoch
            psnr=$(extract_metrics "$latest_log" psnr)
            ssim=$(extract_metrics "$latest_log" ssim)
            loss=$(extract_metrics "$latest_log" loss)
            epoch=$(extract_metrics "$latest_log" epoch)
            metrics_str="best_psnr:$psnr|best_epoch:$epoch|best_ssim:$ssim|latest_epoch:$epoch|latest_loss:$loss|latest_psnr:$psnr|latest_ssim:$ssim"
        fi
    fi
    
    # 4. 生成JSON
    local json_file="$run_dir/metrics.json"
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        
        if [[ -n "$metrics_str" ]]; then
            local best_psnr best_epoch best_ssim
            best_psnr=$(echo "$metrics_str" | grep -oE 'best_psnr:[^|]+' | cut -d':' -f2)
            best_epoch=$(echo "$metrics_str" | grep -oE 'best_epoch:[^|]+' | cut -d':' -f2)
            best_ssim=$(echo "$metrics_str" | grep -oE 'best_ssim:[^|]+' | cut -d':' -f2)
            
            echo "  \"best_psnr\": \"$best_psnr\","
            echo "  \"best_ssim\": \"$best_ssim\","
            echo "  \"best_epoch\": \"$best_epoch\","
        fi
        
        echo "  \"log_count\": $log_count,"
        echo "  \"files\": ["
        
        local first=true
        for f in "$run_dir"/*; do
            [[ -f "$f" ]] || continue
            [[ "$first" == "true" ]] || echo ","
            first=false
            local fname
            fname=$(basename "$f")
            echo -n "    \"$fname\""
        done
        echo ""
        
        echo "  ]"
        echo "}"
    } > "$json_file"
    
    # 5. 生成报告
    local report_file
    report_file=$(generate_report "$run_dir" "$metrics_str")
    
    # 6. 显示结果
    echo ""
    echo -e "${GREEN}✅ 收集完成!${NC}"
    echo ""
    echo "保存位置: $run_dir/"
    echo ""
    echo "文件:"
    ls -lh "$run_dir" | tail -n +2 | awk '{printf "  %-30s %s\n", $9, $5}'
    echo ""
    
    if [[ -f "$json_file" ]]; then
        echo "指标:"
        cat "$json_file" | sed 's/^/  /'
        echo ""
    fi
    
    echo "查看报告: cat $report_file"
}

#===============================================================================
# 入口
#===============================================================================

# 显示帮助
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示帮助"
    echo "  -t, --test     测试SSH连接"
    echo ""
    exit 0
fi

# 测试模式
if [[ "$1" == "-t" || "$1" == "--test" ]]; then
    echo "测试SSH连接..."
    if ssh_test; then
        exit 0
    else
        exit 1
    fi
fi

# 执行收集
collect
