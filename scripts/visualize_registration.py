#!/usr/bin/env python3
"""
配准结果可视化脚本
生成厚扫和薄扫的叠加对比图，用于验证配准质量
"""

import os
import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_nifti(filepath):
    """读取NIfTI文件"""
    return sitk.ReadImage(str(filepath))


def create_checkerboard(img1, img2, block_size=10):
    """创建棋盘格叠加图"""
    arr1 = sitk.GetArrayFromImage(img1)
    arr2 = sitk.GetArrayFromImage(img2)
    
    # 确保尺寸一致
    min_shape = tuple(min(a, b) for a, b in zip(arr1.shape, arr2.shape))
    arr1 = arr1[:min_shape[0], :min_shape[1], :min_shape[2]]
    arr2 = arr2[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # 创建棋盘格掩码
    checkerboard = np.zeros_like(arr1)
    for i in range(0, arr1.shape[0], block_size * 2):
        for j in range(0, arr1.shape[1], block_size * 2):
            checkerboard[i:i+block_size, j:j+block_size] = 1
            if i + block_size < arr1.shape[0] and j + block_size < arr1.shape[1]:
                checkerboard[i+block_size:i+block_size*2, j+block_size:j+block_size*2] = 1
    
    # 应用掩码
    result = np.where(checkerboard, arr1, arr2)
    return result


def create_overlay_image(thick_arr, thin_arr, alpha=0.5):
    """创建透明叠加图"""
    # 归一化到0-1
    thick_norm = (thick_arr - thick_arr.min()) / (thick_arr.max() - thick_arr.min() + 1e-8)
    thin_norm = (thin_arr - thin_arr.min()) / (thin_arr.max() - thin_arr.min() + 1e-8)
    
    # 创建RGB图像
    overlay = np.zeros((*thick_norm.shape, 3))
    overlay[..., 0] = thick_norm  # 厚扫 - 红色通道
    overlay[..., 1] = thin_norm   # 薄扫 - 绿色通道
    overlay[..., 2] = (thick_norm + thin_norm) / 2  # 蓝色通道
    
    return np.clip(overlay, 0, 1)


def visualize_patient(thick_path, thin_path, output_path, patient_id, n_slices=5):
    """
    可视化单个患者的配准结果
    
    参数:
        thick_path: 厚扫文件路径
        thin_path: 薄扫文件路径
        output_path: 输出图像路径
        patient_id: 患者ID
        n_slices: 显示的切片数量
    """
    thick_img = read_nifti(thick_path)
    thin_img = read_nifti(thin_path)
    
    thick_arr = sitk.GetArrayFromImage(thick_img)
    thin_arr = sitk.GetArrayFromImage(thin_img)
    
    # 选择中央切片
    z_mid = thick_arr.shape[0] // 2
    z_indices = np.linspace(z_mid - n_slices//2, z_mid + n_slices//2, n_slices, dtype=int)
    z_indices = np.clip(z_indices, 0, thick_arr.shape[0] - 1)
    
    # 创建子图
    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes.reshape(1, -1)
    
    for i, z_idx in enumerate(z_indices):
        # 计算对应的薄扫切片 (假设Z比例)
        z_ratio = thin_arr.shape[0] / thick_arr.shape[0]
        z_idx_thin = int(z_idx * z_ratio)
        z_idx_thin = min(z_idx_thin, thin_arr.shape[0] - 1)
        
        thick_slice = thick_arr[z_idx]
        thin_slice = thin_arr[z_idx_thin]
        
        # 列1: 厚扫
        axes[i, 0].imshow(thick_slice, cmap='gray', vmin=-1000, vmax=1000)
        axes[i, 0].set_title(f'厚扫 (Z={z_idx})' if i == 0 else f'Z={z_idx}')
        axes[i, 0].axis('off')
        
        # 列2: 薄扫
        axes[i, 1].imshow(thin_slice, cmap='gray', vmin=-1000, vmax=1000)
        axes[i, 1].set_title(f'薄扫 (Z={z_idx_thin})' if i == 0 else f'Z={z_idx_thin}')
        axes[i, 1].axis('off')
        
        # 列3: 差值图
        # 将薄扫重采样到厚扫大小
        from scipy.ndimage import zoom
        zoom_factor = thick_slice.shape[0] / thin_slice.shape[0]
        thin_resized = zoom(thin_slice, zoom_factor, order=1)
        
        if thin_resized.shape != thick_slice.shape:
            # 裁剪或填充
            min_h = min(thin_resized.shape[0], thick_slice.shape[0])
            min_w = min(thin_resized.shape[1], thick_slice.shape[1])
            diff = np.zeros_like(thick_slice)
            diff[:min_h, :min_w] = thick_slice[:min_h, :min_w] - thin_resized[:min_h, :min_w]
        else:
            diff = thick_slice - thin_resized
        
        axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=-500, vmax=500)
        axes[i, 2].set_title('差值 (厚-薄)' if i == 0 else '')
        axes[i, 2].axis('off')
        
        # 列4: 叠加图
        overlay = create_overlay_image(thick_slice, thin_resized)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('叠加 (红=厚, 绿=薄)' if i == 0 else '')
        axes[i, 3].axis('off')
    
    plt.suptitle(f'患者 {patient_id} 配准可视化', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  可视化图像已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='配准结果可视化')
    parser.add_argument('--data-dir', default='RPLHR-CT-main/data/thick-thin-layer-paired/registered',
                       help='配准后数据目录')
    parser.add_argument('--output-dir', default='RPLHR-CT-main/data/thick-thin-layer-paired/registration_vis',
                       help='可视化输出目录')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--patients', nargs='+', help='指定患者ID')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载报告
    report_path = data_dir / 'registration_report.json'
    if not report_path.exists():
        print(f"错误: 找不到报告文件 {report_path}")
        print("请先运行 register_ct_pairs.py")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # 确定要可视化的患者
    patients_to_vis = []
    for result in report['results']:
        if args.patients and result['patient_id'] not in args.patients:
            continue
        if args.split != 'all' and result['split'] != args.split:
            continue
        patients_to_vis.append(result)
    
    print(f"将为 {len(patients_to_vis)} 个患者生成可视化图像")
    
    for result in patients_to_vis:
        patient_id = result['patient_id']
        split = result['split']
        
        thick_path = result.get('output_thick_path') or result['thick_path']
        thin_path = result.get('output_thin_path') or result['thin_path']
        
        # 处理相对路径
        if not Path(thick_path).is_absolute():
            thick_path = Path(args.data_dir) / split / 'thick' / f'{patient_id}.nii.gz'
            thin_path = Path(args.data_dir) / split / 'thin' / f'{patient_id}.nii.gz'
        
        output_path = output_dir / f'{patient_id}_{split}.png'
        
        try:
            visualize_patient(thick_path, thin_path, output_path, patient_id)
        except Exception as e:
            print(f"  错误: 无法可视化 {patient_id}: {e}")
    
    print(f"\n所有可视化图像已保存到: {output_dir}")


if __name__ == '__main__':
    main()
