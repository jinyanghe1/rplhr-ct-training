#!/bin/bash
#===============================================================================
# 设置 Crontab 定时任务
#
# 伪代码:
# 定时 60mins
# 运行 Claude (yolo mode since no one is monitoring)
# -> prompt (完成修改代码训练闭环)
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# crontab 条目
CRON_ENTRY="# AutoDL MLOps - 每小时运行一次训练闭环
# 运行时间: 每小时第 30 分钟
30 * * * * cd $PROJECT_DIR && ./scripts/autodl_skill/run_training.sh 10 >> ./scripts/autodl_skill/logs/cron_$(date +\%Y\%m).log 2>&1

# Claude Code Yolo Mode - 每小时自动检查并微调
# 使用 --dangerously-skip-permanent 避免阻塞
0 * * * * claudec --dangerously-skip-permanent << 'CLAUDE_EOF'
请检查 AutoDL 训练状态，如果训练完成，请分析日志并记录训练心得到 logs/ 目录。
如果发现改进空间，请适当调整代码参数（学习率、batch size等），但不要修改模型架构。
完成分析后，将修改提交到 GitHub（如果有必要）。
CLAUDE_EOF
"

install_crontab() {
    echo -e "${BLUE}[INFO]${NC} 安装 crontab..."

    # 备份现有 crontab
    crontab -l > "$SCRIPT_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true

    # 添加新条目
    (echo "$CRON_ENTRY"; crontab -l 2>/dev/null) | crontab -

    echo -e "${GREEN}[SUCCESS]${NC} Crontab 已安装!"
    echo ""
    echo "当前 crontab:"
    crontab -l
}

uninstall_crontab() {
    echo -e "${YELLOW}[WARN]${NC} 移除 crontab..."
    # 移除 AutoDL 相关条目
    crontab -l 2>/dev/null | grep -v "AutoDL MLOps" | grep -v "Claude Code Yolo" | crontab -
    echo -e "${GREEN}[SUCCESS]${NC} Crontab 已清理"
}

show_status() {
    echo ""
    echo "=============================================="
    echo "  Crontab 状态"
    echo "=============================================="
    crontab -l 2>/dev/null || echo "无 crontab 配置"
    echo ""
}

case "$1" in
    install)
        install_crontab
        ;;
    uninstall)
        uninstall_crontab
        ;;
    status)
        show_status
        ;;
    *)
        echo "用法: $0 {install|uninstall|status}"
        echo ""
        echo "  install   - 安装 crontab (每小时运行训练)"
        echo "  uninstall - 移除 crontab"
        echo "  status    - 查看当前 crontab"
        echo ""
        show_status
        ;;
esac
