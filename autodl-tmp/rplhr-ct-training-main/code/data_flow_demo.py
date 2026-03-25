#!/usr/bin/env python3
"""
RPLHR-CT 数据流演示脚本

这个脚本演示了数据加载和处理的核心流程，帮助理解：
1. 如何读取 nii.gz 文件
2. 训练时的数据采样流程
3. 验证时的切块-拼接流程

运行方式:
    python data_flow_demo.py --data_path ../data/
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from copy import deepcopy

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_volume_info(name, volume):
    """打印体积信息"""
    print(f"  {name}:")
    print(f"    Shape: {volume.shape}")
    print(f"    Dtype: {volume.dtype}")
    print(f"    Range: [{volume.min():.2f}, {volume.max():.2f}]")


def demo_read_niigz(data_path):
    """演示 1: 直接读取 nii.gz 文件"""
    print("\n" + "="*70)
    print("演示 1: 直接读取 nii.gz 文件")
    print("="*70)
    
    train_1mm_path = os.path.join(data_path, 'train/1mm/')
    train_5mm_path = os.path.join(data_path, 'train/5mm/')
    
    # 检查路径是否存在
    if not os.path.exists(train_1mm_path):
        print(f"错误: 路径不存在 {train_1mm_path}")
        print("请确保已解压数据到正确的位置")
        return None, None
    
    # 获取病例列表
    cases = [f.replace('.nii.gz', '') for f in os.listdir(train_1mm_path) if f.endswith('.nii.gz')]
    if not cases:
        print("错误: 未找到 nii.gz 文件")
        return None, None
    
    print(f"\n找到 {len(cases)} 个病例")
    print(f"示例病例: {cases[:5]}")
    
    # 读取第一个病例
    case_name = cases[0]
    print(f"\n读取病例: {case_name}")
    
    # 读取 1mm (HR)
    hr_path = os.path.join(train_1mm_path, f'{case_name}.nii.gz')
    hr_img = sitk.ReadImage(hr_path)
    hr_array = sitk.GetArrayFromImage(hr_img)
    
    # 读取 5mm (LR)
    lr_path = os.path.join(train_5mm_path, f'{case_name}.nii.gz')
    lr_img = sitk.ReadImage(lr_path)
    lr_array = sitk.GetArrayFromImage(lr_img)
    
    print_volume_info("HR (1mm)", hr_array)
    print_volume_info("LR (5mm)", lr_array)
    
    # 验证比例关系
    z_ratio = hr_array.shape[0] / lr_array.shape[0]
    print(f"\n  Z 方向比例: {z_ratio:.1f}x (理论值: 5x)")
    
    return lr_array, hr_array


def demo_train_sampling(lr_vol, hr_vol, c_z=4, c_y=256, c_x=256):
    """演示 2: 训练时的随机采样"""
    print("\n" + "="*70)
    print("演示 2: 训练时的随机采样 (Random Crop)")
    print("="*70)
    
    if lr_vol is None or hr_vol is None:
        print("跳过 (数据未加载)")
        return
    
    print(f"\n配置: c_z={c_z}, c_y={c_y}, c_x={c_x}")
    print(f"原始体积: LR {lr_vol.shape}, HR {hr_vol.shape}")
    
    # 模拟随机采样 (类似 get_train_img)
    np.random.seed(42)  # 为了可重复
    
    z, y, x = lr_vol.shape
    z_s = np.random.randint(0, z - c_z)
    y_s = np.random.randint(0, y - c_y)
    x_s = np.random.randint(0, x - c_x)
    z_e = z_s + c_z
    y_e = y_s + c_y
    x_e = x_s + c_x
    
    print(f"\n随机裁剪位置:")
    print(f"  Z: [{z_s}:{z_e}]")
    print(f"  Y: [{y_s}:{y_e}]")
    print(f"  X: [{x_s}:{x_e}]")
    
    # 裁剪 LR
    lr_crop = lr_vol[z_s:z_e, y_s:y_e, x_s:x_e]
    
    # 计算对应的 HR 裁剪位置
    # 注意：这里的映射关系是关键
    hr_z_s = z_s * 5 + 3
    hr_z_e = (z_e - 1) * 5 - 2  # 或者简化为 (z_s + c_z - 1) * 5 - 2
    
    print(f"\n对应的 HR 裁剪位置:")
    print(f"  Z: [{hr_z_s}:{hr_z_e}] (基于 5:1 比例)")
    print(f"  Y: [{y_s}:{y_e}]")
    print(f"  X: [{x_s}:{x_e}]")
    
    hr_crop = hr_vol[hr_z_s:hr_z_e, y_s:y_e, x_s:x_e]
    
    print(f"\n裁剪结果:")
    print_volume_info("LR Crop", lr_crop)
    print_volume_info("HR Crop", hr_crop)
    
    # 验证比例
    z_ratio = hr_crop.shape[0] / lr_crop.shape[0]
    print(f"\n  实际 Z 比例: {z_ratio:.1f}x")


def demo_val_patching(lr_vol, hr_vol, vc_z=4, vc_y=256, vc_x=256):
    """演示 3: 验证时的切块策略"""
    print("\n" + "="*70)
    print("演示 3: 验证时的切块策略 (Sliding Window)")
    print("="*70)
    
    if lr_vol is None or hr_vol is None:
        print("跳过 (数据未加载)")
        return
    
    print(f"\n配置: vc_z={vc_z}, vc_y={vc_y}, vc_x={vc_x}")
    print(f"重叠: Z方向重叠 {vc_z - 2} 层")
    
    z, y, x = lr_vol.shape
    
    # 计算切块位置 (类似 get_val_img)
    z_s = 0
    z_split = []
    while z_s + vc_z < z:
        z_split.append(z_s)
        z_s += (vc_z - 2)  # 重叠 2 层
    if z - vc_z > z_split[-1]:
        z_split.append(z - vc_z)
    
    y_split = np.arange(y // vc_y) * vc_y
    x_split = np.arange(x // vc_x) * vc_x
    
    num_patches = len(z_split) * len(y_split) * len(x_split)
    
    print(f"\n切块规划:")
    print(f"  Z 位置: {len(z_split)} 个 {z_split[:5]}...")
    print(f"  Y 位置: {len(y_split)} 个 {y_split}")
    print(f"  X 位置: {len(x_split)} 个 {x_split}")
    print(f"  总计: {num_patches} 个小块")
    
    # 模拟前几个切块
    print(f"\n前 3 个切块示例:")
    count = 0
    for z_pos in z_split[:2]:
        for y_pos in y_split[:2]:
            for x_pos in x_split[:1]:
                if count >= 3:
                    break
                patch = lr_vol[z_pos:z_pos+vc_z, y_pos:y_pos+vc_y, x_pos:x_pos+vc_x]
                print(f"  Patch {count}: shape={patch.shape}, pos=({z_pos}, {y_pos}, {x_pos})")
                count += 1
    
    # 演示拼接概念
    print(f"\n拼接后的预测体积:")
    print(f"  Shape: ({z*5}, {y}, {x})  # Z方向5倍上采样")
    print(f"  与 HR GT {hr_vol.shape} 对比计算 PSNR")


def demo_directory_structure(data_path):
    """演示 4: 检查目录结构"""
    print("\n" + "="*70)
    print("演示 4: 数据目录结构检查")
    print("="*70)
    
    required_dirs = [
        'train/1mm', 'train/5mm',
        'val/1mm', 'val/5mm',
        'test/1mm', 'test/5mm'
    ]
    
    print(f"\n数据根目录: {data_path}")
    print("\n目录状态:")
    
    for subdir in required_dirs:
        full_path = os.path.join(data_path, subdir)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        
        if exists:
            files = [f for f in os.listdir(full_path) if f.endswith('.nii.gz')]
            print(f"  {status} {subdir:20s} ({len(files)} files)")
        else:
            print(f"  {status} {subdir:20s} (missing)")


def main():
    parser = argparse.ArgumentParser(description='RPLHR-CT 数据流演示')
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='数据目录路径 (默认: ../data/)')
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_path = os.path.abspath(args.data_path)
    
    print("\n" + "="*70)
    print("RPLHR-CT 数据流演示")
    print("="*70)
    print(f"\n数据路径: {data_path}")
    
    # 演示 4: 目录结构
    demo_directory_structure(data_path)
    
    # 演示 1: 读取数据
    lr_vol, hr_vol = demo_read_niigz(data_path)
    
    # 演示 2: 训练采样
    demo_train_sampling(lr_vol, hr_vol)
    
    # 演示 3: 验证切块
    demo_val_patching(lr_vol, hr_vol)
    
    print("\n" + "="*70)
    print("演示完成!")
    print("="*70)
    print("""
总结:
1. 数据格式: 直接使用 .nii.gz，通过 SimpleITK 读取
2. 训练流程: 随机裁剪 4×256×256 (5mm) → 网络 → 16×256×256 SR
3. 验证流程: 切块 → 逐块推理 → 拼接 → 计算 PSNR/SSIM
4. 数据配对: 相同文件名的 5mm 和 1mm 构成一对

更多信息请查看:
  - DATA_PIPELINE_EXPLAINED.md (详细说明)
  - DATA_FLOW_VISUAL.md (可视化流程)
""")


if __name__ == '__main__':
    main()
