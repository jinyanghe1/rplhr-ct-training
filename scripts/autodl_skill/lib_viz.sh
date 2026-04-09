#!/bin/bash
#===============================================================================
# 可视化工具库 - 提供训练结果的可视化功能
# 用法: source $0
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#===============================================================================
# 生成CSV数据的PSNR趋势图 (使用gnuplot或ASCII)
#===============================================================================

generate_psnr_plot() {
    local csv_file="$1"
    local output_file="${2:-psnr_plot.txt}"
    
    if [[ ! -f "$csv_file" ]]; then
        echo "错误: CSV文件不存在" >&2
        return 1
    fi
    
    # 检查gnuplot是否可用
    if command -v gnuplot &>/dev/null; then
        # 使用gnuplot生成图表
        gnuplot << EOF 2>/dev/null
set terminal png size 800,400
set output '${output_file%.txt}.png'
set title 'PSNR Training Progress'
set xlabel 'Epoch'
set ylabel 'PSNR (dB)'
set grid
set datafile separator ','
plot '$csv_file' using 1:4 with lines title 'PSNR' lw 2, \
     '' using 1:4 smooth bezier title 'Trend' lw 1 lt 2
EOF
        echo "图表已保存: ${output_file%.txt}.png"
    else
        # 使用ASCII图表
        generate_ascii_trend "$csv_file" PSNR 4 > "$output_file"
        echo "ASCII图表已保存: $output_file"
    fi
}

#===============================================================================
# 生成ASCII趋势图
#===============================================================================

generate_ascii_trend() {
    local csv_file="$1"
    local metric_name="${2:-Value}"
    local column="${3:-4}"
    local width="${4:-50}"
    
    local values
    values=$(tail -n +2 "$csv_file" | cut -d',' -f"$column" | grep -v '^$')
    
    if [[ -z "$values" ]]; then
        echo "无数据"
        return 1
    fi
    
    local max_val min_val
    max_val=$(echo "$values" | sort -n | tail -1)
    min_val=$(echo "$values" | sort -n | head -1)
    
    # 添加一些边距
    local range
    range=$(echo "scale=4; $max_val - $min_val" | bc 2>/dev/null || echo "1")
    
    echo "$metric_name 趋势图"
    echo "范围: $min_val - $max_val"
    echo ""
    
    local count=0
    tail -n +2 "$csv_file" | cut -d',' -f1,"$column" | while IFS=',' read -r epoch value; do
        [[ -z "$value" ]] && continue
        
        # 计算条形长度
        local normalized
        normalized=$(echo "scale=4; ($value - $min_val) / $range" | bc 2>/dev/null || echo "0")
        local bar_len
        bar_len=$(echo "scale=0; $normalized * $width / 1" | bc 2>/dev/null || echo "0")
        
        printf "%4s |" "$epoch"
        for ((i=0; i<bar_len; i++)); do echo -n "█"; done
        for ((i=bar_len; i<width; i++)); do echo -n " "; done
        echo "| $value"
        
        count=$((count + 1))
        [[ $count -ge 30 ]] && break  # 限制显示行数
    done
}

#===============================================================================
# 生成对比柱状图
#===============================================================================

generate_comparison_chart() {
    local data_file="$1"  # 格式: name|value
    local metric_name="${2:-Score}"
    local width="${3:-40}"
    
    echo "$metric_name 对比"
    echo ""
    
    # 获取最大值
    local max_val
    max_val=$(cut -d'|' -f2 "$data_file" | sort -n | tail -1)
    
    while IFS='|' read -r name value; do
        [[ -z "$name" ]] && continue
        
        local bar_len
        bar_len=$(echo "scale=0; $value * $width / $max_val" | bc 2>/dev/null || echo "0")
        
        printf "%-15s |" "$name"
        for ((i=0; i<bar_len; i++)); do echo -n "█"; done
        echo " $value"
    done < "$data_file"
}

#===============================================================================
# 生成HTML报告
#===============================================================================

generate_html_report() {
    local run_dir="$1"
    local output_file="${2:-report.html}"
    
    local metrics_json
    metrics_json=$(find "$run_dir" -name "metrics.json" -type f | head -1)
    
    {
        echo "<!DOCTYPE html>"
        echo "<html>"
        echo "<head>"
        echo "<title>RPLHR-CT 训练报告</title>"
        echo "<style>"
        echo "body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }"
        echo ".container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }"
        echo "h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }"
        echo "h2 { color: #555; margin-top: 30px; }"
        echo ".metric { display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #f9f9f9; border-radius: 4px; }"
        echo ".metric-label { font-size: 12px; color: #888; }"
        echo ".metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }"
        echo "table { width: 100%; border-collapse: collapse; margin: 20px 0; }"
        echo "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }"
        echo "th { background: #4CAF50; color: white; }"
        echo "tr:hover { background: #f5f5f5; }"
        echo "pre { background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; }"
        echo "</style>"
        echo "</head>"
        echo "<body>"
        echo "<div class='container'>"
        echo "<h1>🧠 RPLHR-CT 训练报告</h1>"
        echo "<p>生成时间: $(date '+%Y-%m-%d %H:%M:%S')</p>"
        
        # 指标部分
        if [[ -f "$metrics_json" ]]; then
            echo "<h2>📊 关键指标</h2>"
            echo "<div>"
            
            local best_psnr best_ssim
            best_psnr=$(grep -oE '"best_psnr": "[^"]+"' "$metrics_json" | cut -d'"' -f4)
            best_ssim=$(grep -oE '"best_ssim": "[^"]+"' "$metrics_json" | cut -d'"' -f4)
            
            echo "<div class='metric'>"
            echo "<div class='metric-label'>Best PSNR</div>"
            echo "<div class='metric-value'>${best_psnr:-N/A}</div>"
            echo "</div>"
            
            echo "<div class='metric'>"
            echo "<div class='metric-label'>Best SSIM</div>"
            echo "<div class='metric-value'>${best_ssim:-N/A}</div>"
            echo "</div>"
            
            echo "</div>"
        fi
        
        # 文件列表
        echo "<h2>📁 文件</h2>"
        echo "<table>"
        echo "<tr><th>文件名</th><th>大小</th><th>修改时间</th></tr>"
        
        for f in "$run_dir"/*; do
            [[ -f "$f" ]] || continue
            local fname fsize ftime
            fname=$(basename "$f")
            fsize=$(ls -lh "$f" | awk '{print $5}')
            ftime=$(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1 || stat -f '%Sm' "$f" 2>/dev/null)
            echo "<tr><td>$fname</td><td>$fsize</td><td>$ftime</td></tr>"
        done
        
        echo "</table>"
        
        # 日志预览
        local log_file
        log_file=$(find "$run_dir" -name "*.log" -type f | head -1)
        if [[ -f "$log_file" ]]; then
            echo "<h2>📝 日志预览 (最后20行)</h2>"
            echo "<pre>"
            tail -20 "$log_file" | sed 's/</\&lt;/g; s/>/\&gt;/g'
            echo "</pre>"
        fi
        
        echo "</div>"
        echo "</body>"
        echo "</html>"
    } > "$output_file"
    
    echo "HTML报告已生成: $output_file"
}

#===============================================================================
# 生成Markdown表格
#===============================================================================

generate_markdown_table() {
    local data_file="$1"
    local headers="$2"  # 格式: "Col1|Col2|Col3"
    
    echo "| $(echo "$headers" | tr '|' '|') |"
    
    # 生成分隔符
    local cols
    cols=$(echo "$headers" | tr '|' '\n' | wc -l)
    printf "|"
    for ((i=0; i<cols; i++)); do printf " --- |"; done
    echo ""
    
    # 数据行
    while IFS='|' read -r -a row; do
        printf "|"
        for cell in "${row[@]}"; do
            printf " %s |" "$cell"
        done
        echo ""
    done < "$data_file"
}

#===============================================================================
# 主可视化函数
#===============================================================================

visualize_training() {
    local run_dir="$1"
    local output_dir="${2:-$run_dir/visualization}"
    
    mkdir -p "$output_dir"
    
    local csv_file
    csv_file=$(find "$run_dir" -name "metrics.csv" -type f | head -1)
    
    if [[ -f "$csv_file" ]]; then
        # 生成ASCII图表
        generate_ascii_trend "$csv_file" PSNR 4 > "$output_dir/psnr_trend.txt"
        generate_ascii_trend "$csv_file" Loss 3 > "$output_dir/loss_trend.txt"
        
        # 尝试生成gnuplot图表
        if command -v gnuplot &>/dev/null; then
            generate_psnr_plot "$csv_file" "$output_dir/psnr_plot.txt"
        fi
        
        echo "可视化文件已保存到: $output_dir/"
        ls -la "$output_dir/"
    else
        echo "未找到metrics.csv文件"
        return 1
    fi
}
