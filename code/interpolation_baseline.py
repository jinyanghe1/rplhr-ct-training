#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插值基线: bicubic / trilinear / nearest z-axis 上采样

对5mm CT volume沿z轴插值到与1mm GT相同分辨率，计算PSNR/SSIM/MSE。
用于论文消融表中的插值对照实验。

用法 (AutoDL):
    cd /root/autodl-tmp/rplhr-ct-training-main/code
    python interpolation_baseline.py run \
        --path_key=SRM --subset=val --ratio=5 \
        --output_dir=../interp_baseline_results

    python interpolation_baseline.py run \
        --path_key=dataset01_xuanwu --subset=val --ratio=4 \
        --normalize_ct_input=True \
        --output_dir=../interp_baseline_xuanwu
"""

import os
import sys
import json
import csv
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom as ndimage_zoom
from tqdm import tqdm

# ---- metric helpers (standalone, no torch dependency for CPU-only usage) ----
import math

def cal_psnr(img1, img2, pixel_max=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 40.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def cal_mse(img1, img2):
    return float(np.mean((img1 - img2) ** 2))

def cal_ssim_numpy(img1, img2):
    """Compute mean SSIM using skimage (CPU). Falls back to slice-level torch SSIM if unavailable."""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2, data_range=1.0)
    except ImportError:
        pass
    # Fallback: torch-based per-slice SSIM
    try:
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.non_model import cal_ssim_volume
        return cal_ssim_volume(img1, img2, device=torch.device('cpu'), batch_size=16)
    except Exception:
        return float('nan')


def _auto_normalize_ct_pair(lr_data, hr_data):
    """Mirror of in_model._auto_normalize_ct_pair for standalone usage."""
    lr_max = np.max(lr_data)
    hr_max = np.max(hr_data)
    if lr_max <= 10.0 and hr_max <= 10.0:
        return lr_data, hr_data

    combined_p01 = min(np.percentile(lr_data, 0.1), np.percentile(hr_data, 0.1))
    if combined_p01 < -500:
        lr_data = lr_data + 1024.0
        hr_data = hr_data + 1024.0

    lr_data = np.clip(lr_data, 0, 4096) / 4096.0
    hr_data = np.clip(hr_data, 0, 4096) / 4096.0
    return lr_data.astype(np.float32), hr_data.astype(np.float32)


def interpolate_z(volume, target_z, method='cubic'):
    """
    沿z轴插值3D体积到目标z尺寸。

    Args:
        volume: (Z_lr, Y, X) numpy array
        target_z: 目标z维度
        method: 'nearest', 'linear', 'cubic'
    Returns:
        (target_z, Y, X) numpy array
    """
    z_lr = volume.shape[0]
    if z_lr == target_z:
        return volume.copy()

    zoom_factor = target_z / z_lr

    order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
    order = order_map.get(method, 3)

    # zoom only along z-axis
    result = ndimage_zoom(volume, (zoom_factor, 1.0, 1.0), order=order, mode='nearest')

    # ensure exact target_z (rounding may cause ±1 difference)
    if result.shape[0] > target_z:
        result = result[:target_z]
    elif result.shape[0] < target_z:
        pad = np.zeros((target_z - result.shape[0], *result.shape[1:]), dtype=result.dtype)
        result = np.concatenate([result, pad], axis=0)

    return result


def list_paired_cases(path_img, subset, high_res='1mm', low_res='5mm'):
    """Return sorted case names that exist in both high/low resolution folders."""
    high_dir = os.path.join(path_img, subset, high_res)
    low_dir = os.path.join(path_img, subset, low_res)

    if not os.path.isdir(high_dir) or not os.path.isdir(low_dir):
        return []

    high_cases = {f[:-7] for f in os.listdir(high_dir) if f.endswith('.nii.gz')}
    low_cases = {f[:-7] for f in os.listdir(low_dir) if f.endswith('.nii.gz')}
    return sorted(high_cases & low_cases)


def run(path_key=None, path_img=None, subset='val', ratio=5,
        normalize_ct_input=False, output_dir=None,
        methods='nearest,linear,cubic', save_nifti=True,
        crop_margin=3):
    """
    运行插值基线实验。

    Args:
        path_key: 数据集配置key (e.g. SRM, dataset01_xuanwu)
        path_img: 直接指定数据路径 (覆盖path_key)
        subset: val 或 test
        ratio: 超分倍率 (用于评估裁剪)
        normalize_ct_input: 是否归一化CT数据 (宣武数据需要True)
        output_dir: 输出目录
        methods: 逗号分隔的插值方法列表
        save_nifti: 是否保存插值后的NIfTI文件
        crop_margin: z方向裁剪边距 (与模型评估对齐)
    """
    # Resolve data path
    if path_img is None:
        if path_key is None:
            print("Error: must provide --path_key or --path_img")
            sys.exit(1)
        dict_path = '../config/%s_dict.json' % path_key
        with open(dict_path, 'r') as f:
            data_info = json.load(f)
        path_img = data_info['path_img']

    if output_dir is None:
        output_dir = '../interp_baseline_%s_%s' % (path_key or 'custom', subset)

    os.makedirs(output_dir, exist_ok=True)
    method_list = [m.strip() for m in methods.split(',')]

    # Find paired cases
    cases = list_paired_cases(path_img, subset)
    if len(cases) == 0:
        print(f"Error: no paired cases found in {path_img}/{subset}/")
        sys.exit(1)
    print(f"Found {len(cases)} paired {subset} cases")

    # Results storage
    all_results = {m: [] for m in method_list}
    csv_rows = []

    for case_name in tqdm(cases, desc='Processing'):
        lr_path = os.path.join(path_img, subset, '5mm', case_name + '.nii.gz')
        hr_path = os.path.join(path_img, subset, '1mm', case_name + '.nii.gz')

        lr_sitk = sitk.ReadImage(lr_path)
        hr_sitk = sitk.ReadImage(hr_path)
        lr_vol = sitk.GetArrayFromImage(lr_sitk).astype(np.float32)
        hr_vol = sitk.GetArrayFromImage(hr_sitk).astype(np.float32)

        if normalize_ct_input:
            lr_vol, hr_vol = _auto_normalize_ct_pair(lr_vol, hr_vol)

        z_hr = hr_vol.shape[0]

        # Trim edges to match model evaluation: y[ratio:-ratio]
        trim = ratio
        if z_hr <= 2 * trim:
            print(f"  Warning: {case_name} z_hr={z_hr} too small for trim={trim}, skipping")
            continue
        hr_trimmed = hr_vol[trim:-trim]

        for method in method_list:
            interp_vol = interpolate_z(lr_vol, z_hr, method=method)
            interp_trimmed = interp_vol[trim:-trim]

            # Clip to [0, 1] for normalized data
            if normalize_ct_input or np.max(hr_trimmed) <= 2.0:
                interp_trimmed = np.clip(interp_trimmed, 0, 1)

            psnr = cal_psnr(interp_trimmed, hr_trimmed)
            mse = cal_mse(interp_trimmed, hr_trimmed)
            ssim_val = cal_ssim_numpy(interp_trimmed, hr_trimmed)

            all_results[method].append({
                'case': case_name,
                'psnr': psnr,
                'ssim': ssim_val,
                'mse': mse,
            })

            csv_rows.append({
                'case': case_name,
                'method': method,
                'psnr': f'{psnr:.4f}',
                'ssim': f'{ssim_val:.6f}',
                'mse': f'{mse:.8f}',
                'z_lr': lr_vol.shape[0],
                'z_hr': z_hr,
            })

            # Save interpolated NIfTI
            if save_nifti:
                nifti_dir = os.path.join(output_dir, 'nifti', method)
                os.makedirs(nifti_dir, exist_ok=True)
                out_img = sitk.GetImageFromArray(interp_vol)
                out_img.CopyInformation(hr_sitk)
                sitk.WriteImage(out_img, os.path.join(nifti_dir, case_name + '.nii.gz'))

            tqdm.write(f"  {case_name} [{method}] PSNR={psnr:.2f} SSIM={ssim_val:.4f} MSE={mse:.6f}")

    # Write CSV
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'method', 'psnr', 'ssim', 'mse', 'z_lr', 'z_hr'])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    print("\n" + "=" * 60)
    print("插值基线结果汇总")
    print("=" * 60)
    summary_rows = []
    for method in method_list:
        results = all_results[method]
        if len(results) == 0:
            continue
        mean_psnr = np.mean([r['psnr'] for r in results])
        mean_ssim = np.mean([r['ssim'] for r in results])
        mean_mse = np.mean([r['mse'] for r in results])
        print(f"  {method:>8s}: PSNR={mean_psnr:.4f}  SSIM={mean_ssim:.6f}  MSE={mean_mse:.8f}  (n={len(results)})")
        summary_rows.append({
            'method': method,
            'mean_psnr': f'{mean_psnr:.4f}',
            'mean_ssim': f'{mean_ssim:.6f}',
            'mean_mse': f'{mean_mse:.8f}',
            'n_cases': len(results),
        })

    # Write summary CSV
    summary_path = os.path.join(output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'mean_psnr', 'mean_ssim', 'mean_mse', 'n_cases'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nResults saved to: {output_dir}")
    print(f"  metrics.csv: per-case results")
    print(f"  summary.csv: method averages")
    if save_nifti:
        print(f"  nifti/: interpolated volumes")


if __name__ == '__main__':
    import fire
    fire.Fire()
