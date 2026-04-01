#!/bin/bash
#===============================================================================
# 创建新实验记录
# 用法: ./create_experiment.sh <实验ID> <实验名称> <基线ID>
# 示例: ./create_experiment.sh A1 "EAGLELoss3D" A0
#===============================================================================

EXPERIMENTS_FILE="$(cd "$(dirname "$0")" && pwd)/EXPERIMENTS.md"

if [ $# -lt 2 ]; then
    echo "用法: $0 <实验ID> <实验名称> [基线ID]"
    echo "示例: $0 A1 \"EAGLELoss3D\" A0"
    exit 1
fi

EXP_ID="$1"
EXP_NAME="$2"
BASELINE="${3:-A0}"
DATE=$(date '+%Y-%m-%d')

# 创建实验记录
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

echo "✅ 实验记录已创建: EXP_ID=$EXP_ID"
echo "📄 文件: $EXPERIMENTS_FILE"
echo ""
echo "下一步:"
echo "1. 编辑 EXPERIMENTS.md 填写配置详情"
echo "2. 执行实验"
echo "3. 更新实验结果"
