#!/usr/bin/env python3
"""
快速配准脚本 - 只做XY spacing对齐，跳过耗时的刚体配准
"""

import os
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

DATA_ROOT = Path("RPLHR-CT-main/data/thick-thin-layer-paired/cleaned_final")
OUTPUT_ROOT = Path("RPLHR-CT-main/data/thick-thin-layer-paired/registered")


def align_xy_spacing(thin_img, thick_img):
    """将薄扫的XY spacing对齐到厚扫"""
    thick_spacing = thick_img.GetSpacing()
    thin_spacing = thin_img.GetSpacing()
    
    xy_diff = abs(thin_spacing[0] - thick_spacing[0]) + abs(thin_spacing[1] - thick_spacing[1])
    
    if xy_diff < 0.01:
        return thin_img, False
    
    # 创建新的spacing：保持Z不变，XY使用厚扫的spacing
    new_spacing = (thick_spacing[0], thick_spacing[1], thin_spacing[2])
    
    # 计算新尺寸
    thin_size = thin_img.GetSize()
    new_size = [
        int(round(thin_size[0] * thin_spacing[0] / new_spacing[0])),
        int(round(thin_size[1] * thin_spacing[1] / new_spacing[1])),
        thin_size[2]
    ]
    
    # 重采样
    resampled = sitk.Resample(
        thin_img,
        new_size,
        sitk.Transform(),
        sitk.sitkBSpline,
        thin_img.GetOrigin(),
        new_spacing,
        thin_img.GetDirection(),
        -1024,
        thin_img.GetPixelID()
    )
    
    return resampled, True


def process_patient(patient_id, split):
    """处理单个患者"""
    thick_path = DATA_ROOT / split / 'thick' / f'{patient_id}.nii.gz'
    thin_path = DATA_ROOT / split / 'thin' / f'{patient_id}.nii.gz'
    
    if not thick_path.exists() or not thin_path.exists():
        return None
    
    # 读取图像
    thick_img = sitk.ReadImage(str(thick_path))
    thin_img = sitk.ReadImage(str(thin_path))
    
    # XY spacing对齐
    thin_aligned, was_resampled = align_xy_spacing(thin_img, thick_img)
    
    # 保存结果
    out_thin_dir = OUTPUT_ROOT / split / 'thin'
    out_thick_dir = OUTPUT_ROOT / split / 'thick'
    out_thin_dir.mkdir(parents=True, exist_ok=True)
    out_thick_dir.mkdir(parents=True, exist_ok=True)
    
    out_thin_path = out_thin_dir / f'{patient_id}.nii.gz'
    out_thick_path = out_thick_dir / f'{patient_id}.nii.gz'
    
    sitk.WriteImage(thin_aligned, str(out_thin_path))
    
    # 复制厚扫（如果不存在）
    if not out_thick_path.exists():
        shutil.copy(thick_path, out_thick_path)
    
    return {
        'patient_id': patient_id,
        'split': split,
        'xy_resampled': was_resampled,
        'original_thin_spacing': list(thin_img.GetSpacing()),
        'aligned_thin_spacing': list(thin_aligned.GetSpacing()),
    }


def main():
    # 加载清单
    with open(DATA_ROOT / 'split_manifest.json', 'r') as f:
        manifest = json.load(f)
    
    all_patients = []
    for split in ['train', 'val', 'test']:
        for entry in manifest.get(split, []):
            pid = entry if isinstance(entry, str) else entry.get('patient_id')
            all_patients.append((pid, split))
    
    print(f"共 {len(all_patients)} 个患者需要处理")
    print("=" * 50)
    
    results = []
    for i, (pid, split) in enumerate(all_patients, 1):
        result = process_patient(pid, split)
        if result:
            results.append(result)
            status = "重采样" if result['xy_resampled'] else "跳过"
            print(f"[{i:2d}/{len(all_patients)}] {pid} ({split}): {status}")
    
    # 保存报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'xy_resampled_count': sum(r['xy_resampled'] for r in results),
        'results': results
    }
    
    with open(OUTPUT_ROOT / 'quick_registration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("=" * 50)
    print(f"处理完成! 共 {len(results)} 个患者")
    print(f"XY重采样: {report['xy_resampled_count']} 个")
    print(f"报告: {OUTPUT_ROOT / 'quick_registration_report.json'}")


if __name__ == '__main__':
    main()
