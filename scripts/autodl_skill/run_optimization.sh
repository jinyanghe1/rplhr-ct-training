#!/bin/bash
#===============================================================================
# Claude Code 优化 Agent 包装脚本
# 供 crontab 调用，每小时运行一次
#===============================================================================

PROJECT_DIR="/Users/hejinyang/毕业设计_0306/RPLHR-CT-main"
PROMPT_FILE="$PROJECT_DIR/scripts/autodl_skill/optimization_prompt.txt"
LOG_DIR="$PROJECT_DIR/scripts/autodl_skill/logs"
LOG_FILE="$LOG_DIR/claude_optimization_$(date +\%Y\%m).log"

# crontab 环境 PATH 不包含 ~/.local/bin，需要手动指定
export PATH="/Users/hejinyang/.local/bin:$PATH"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始模型优化..." >> "$LOG_FILE"

# 设置 CLAUDECODE="" 允许嵌套运行 Claude Code
# 使用 --dangerously-skip-permissions 跳过权限检查
CLAUDECODE="" /Users/hejinyang/.local/bin/claude -p "$(cat "$PROMPT_FILE")" \
    --dangerously-skip-permissions \
    --output-format text \
    >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 优化完成" >> "$LOG_FILE"