#!/bin/bash
# 从阿里云盘下载训练数据

set -e

WORK_DIR="/root/autodl-tmp/RPLHR-CT"
DATA_DIR="$WORK_DIR/data"
mkdir -p "$DATA_DIR"

echo "========================================"
echo "从阿里云盘下载数据集"
echo "========================================"

# 检查是否已登录 aliyunpan
if ! aliyunpan whoami &> /dev/null; then
    echo "请先登录阿里云盘:"
    echo "  aliyunpan login"
    echo "按照提示完成登录"
    exit 1
fi

# 设置下载目录
aliyunpan config set -savedir "$DATA_DIR"

cd "$DATA_DIR"

# 定义文件列表
FILES=(
    "RPLHR-CT-Dataset/train_5mm.tar.gz"
    "RPLHR-CT-Dataset/train_1mm.tar.gz"
    "RPLHR-CT-Dataset/val_5mm.tar.gz"
    "RPLHR-CT-Dataset/val_1mm.tar.gz"
)

# 下载文件
for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    if [ -f "$filename" ]; then
        echo "✓ $filename 已存在，跳过下载"
    else
        echo "下载 $filename..."
        aliyunpan download "$file"
    fi
done

# 解压数据
echo ""
echo "解压数据集..."

if [ ! -d "train/5mm" ]; then
    echo "解压 train_5mm.tar.gz..."
    tar -xzf train_5mm.tar.gz
fi

if [ ! -d "train/1mm" ]; then
    echo "解压 train_1mm.tar.gz..."
    tar -xzf train_1mm.tar.gz
fi

if [ ! -d "val/5mm" ]; then
    echo "解压 val_5mm.tar.gz..."
    tar -xzf val_5mm.tar.gz
fi

if [ ! -d "val/1mm" ]; then
    echo "解压 val_1mm.tar.gz..."
    tar -xzf val_1mm.tar.gz
fi

# 验证数据完整性
echo ""
echo "========================================"
echo "数据验证"
echo "========================================"

train_5mm_count=$(find $DATA_DIR/train/5mm -name "*.nii.gz" 2>/dev/null | wc -l)
train_1mm_count=$(find $DATA_DIR/train/1mm -name "*.nii.gz" 2>/dev/null | wc -l)
val_5mm_count=$(find $DATA_DIR/val/5mm -name "*.nii.gz" 2>/dev/null | wc -l)
val_1mm_count=$(find $DATA_DIR/val/1mm -name "*.nii.gz" 2>/dev/null | wc -l)

echo "训练集 5mm 病例数: $train_5mm_count"
echo "训练集 1mm 病例数: $train_1mm_count"
echo "验证集 5mm 病例数: $val_5mm_count"
echo "验证集 1mm 病例数: $val_1mm_count"

if [ "$train_5mm_count" -eq 0 ] || [ "$train_1mm_count" -eq 0 ]; then
    echo "警告: 训练集数据为空，请检查解压是否成功"
    exit 1
fi

echo ""
echo "========================================"
echo "数据下载完成！"
echo "数据目录: $DATA_DIR"
echo "========================================"
