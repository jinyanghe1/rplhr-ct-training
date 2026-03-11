#!/bin/bash
# 本地数据打包脚本
# 在本地 Mac/Linux 运行，将数据打包后上传到阿里云盘

echo "========================================"
echo "RPLHR-CT 数据打包脚本"
echo "========================================"

# 设置数据目录（根据实际情况修改）
DATA_DIR="${1:-../data}"
OUTPUT_DIR="${2:-./packaged_data}"

if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    echo "用法: bash prepare_data_local.sh [数据目录] [输出目录]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 打包训练集
echo "[1/4] 打包训练集 5mm..."
if [ -d "$DATA_DIR/train/5mm" ]; then
    tar -czf "$OUTPUT_DIR/train_5mm.tar.gz" -C "$DATA_DIR" train/5mm/
    echo "✓ train_5mm.tar.gz"
else
    echo "✗ train/5mm 不存在"
fi

echo "[2/4] 打包训练集 1mm..."
if [ -d "$DATA_DIR/train/1mm" ]; then
    tar -czf "$OUTPUT_DIR/train_1mm.tar.gz" -C "$DATA_DIR" train/1mm/
    echo "✓ train_1mm.tar.gz"
else
    echo "✗ train/1mm 不存在"
fi

# 打包验证集
echo "[3/4] 打包验证集 5mm..."
if [ -d "$DATA_DIR/val/5mm" ]; then
    tar -czf "$OUTPUT_DIR/val_5mm.tar.gz" -C "$DATA_DIR" val/5mm/
    echo "✓ val_5mm.tar.gz"
else
    echo "✗ val/5mm 不存在"
fi

echo "[4/4] 打包验证集 1mm..."
if [ -d "$DATA_DIR/val/1mm" ]; then
    tar -czf "$OUTPUT_DIR/val_1mm.tar.gz" -C "$DATA_DIR" val/1mm/
    echo "✓ val_1mm.tar.gz"
else
    echo "✗ val/1mm 不存在"
fi

# 生成校验文件
echo ""
echo "生成校验文件..."
cd "$OUTPUT_DIR"
md5sum *.tar.gz > checksum.md5
cat checksum.md5

# 显示文件大小
echo ""
echo "========================================"
echo "打包完成！"
echo "========================================"
ls -lh *.tar.gz

echo ""
echo "下一步:"
echo "  1. 在阿里云盘创建 RPLHR-CT-Dataset 文件夹"
echo "  2. 上传以上 .tar.gz 文件到该文件夹"
echo "  3. 上传完成后，在 AutoDL 运行 download_data.sh"
echo ""
