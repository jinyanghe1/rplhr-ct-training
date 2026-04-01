#!/bin/bash
#===============================================================================
# 实验对比报告生成
# 用法: ./compare_experiments.sh
#===============================================================================

EXPERIMENTS_FILE="$(cd "$(dirname "$0")" && pwd)/EXPERIMENTS.md"
ROADMAP_FILE="$(cd "$(dirname "$0")" && pwd)/ROADMAP.md"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_highlight() { echo -e "${BLUE}$1${NC}"; }

echo ""
log_highlight "╔══════════════════════════════════════════════════╗"
log_highlight "║           RPLHR-CT 实验对比报告                  ║"
log_highlight "╚══════════════════════════════════════════════════╝"
echo ""

# 统计实验数量
COMPLETED=$(grep -c "状态.*完成\|状态.*✅" "$EXPERIMENTS_FILE" 2>/dev/null || echo "0")
FAILED=$(grep -c "状态.*失败\|状态.*❌" "$EXPERIMENTS_FILE" 2>/dev/null || echo "0")
PENDING=$(grep -c "状态.*待执行\|状态.*⏳" "$EXPERIMENTS_FILE" 2>/dev/null || echo "0")

echo "📊 实验统计:"
echo "  ✅ 完成: $COMPLETED"
echo "  ❌ 失败: $FAILED"
echo "  ⏳ 待执行: $PENDING"
echo ""

# 显示基线结果
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_highlight "🏆 基线实验 (A0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 提取基线信息
BASELINE_PSNR=$(grep -A20 "## 🎯 基线实验" "$EXPERIMENTS_FILE" 2>/dev/null | grep "最佳结果" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "N/A")
BASELINE_SSIM=$(grep -A20 "## 🎯 基线实验" "$EXPERIMENTS_FILE" 2>/dev/null | grep "最佳结果" | grep -oE "0\.[0-9]+" | head -1 || echo "N/A")

echo "  PSNR: ${BASELINE_PSNR} dB"
echo "  SSIM: ${BASELINE_SSIM}"
echo ""

# 显示待执行实验队列
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_highlight "⏳ 待执行实验队列"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 提取待执行实验
PENDING_EXPS=$(grep -B2 "状态.*⏳" "$EXPERIMENTS_FILE" 2>/dev/null | grep "## EXP_ID" | sed 's/## EXP_ID: //')

if [ -n "$PENDING_EXPS" ]; then
    echo "$PENDING_EXPS" | head -5 | while read exp; do
        echo "  • $exp"
    done
else
    echo "  暂无待执行实验"
fi
echo ""

# 显示Roadmap中的消融矩阵
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_highlight "📋 消融实验矩阵 (来自ROADMAP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 提取Loss Function消融表格
echo ""
echo "Loss Function 消融:"
grep -A10 "实验组 A:" "$ROADMAP_FILE" 2>/dev/null | grep "^|" | tail -6

echo ""
echo "完整矩阵请查看: $ROADMAP_FILE"
echo ""

# 显示下一步行动
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_highlight "📝 下一步行动"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 提取TODO列表中的高优先级任务
echo ""
echo "P0 - 立即执行:"
grep -A3 "P0 - 立即执行" "$ROADMAP_FILE" 2>/dev/null | grep "^- \[ \]" | head -3 | sed 's/^- \[ \]/  •/'

echo ""
echo "P1 - 本周完成:"
grep -A3 "P1 - 本周完成" "$ROADMAP_FILE" 2>/dev/null | grep "^- \[ \]" | head -3 | sed 's/^- \[ \]/  •/'

echo ""
log_highlight "══════════════════════════════════════════════════"
echo "$(date '+%Y-%m-%d %H:%M:%S') 报告生成完成"
echo ""
echo "快捷命令:"
echo "  • 创建新实验: ./create_experiment.sh A1 \"EAGLELoss3D\" A0"
echo "  • 查看详细记录: cat EXPERIMENTS.md"
echo "  • 查看完整路线图: cat ROADMAP.md"
echo ""
