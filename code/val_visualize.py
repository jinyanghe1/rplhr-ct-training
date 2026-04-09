# -*- coding: utf-8 -*-
"""
Phase B: 验证集可视化脚本
生成 LR | SR | GT 对比图, 矢状面, 差值图, z-profile 曲线
输出到 验证集结果_可视化/
"""
import os
import random
import csv

import torch
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset import val_Dataset
from net import model_TransSR

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['1', 'true', 'yes', 'y', 'on']:
            return True
        if value in ['0', 'false', 'no', 'n', 'off']:
            return False
    return bool(value)


def _make_axial_comparison(lr_vol, sr_vol, gt_vol, case_name, psnr, ssim_val, output_dir, z_idx=None):
    """生成轴向切片对比图: LR | SR | GT + 差值图"""
    if z_idx is None:
        z_idx = sr_vol.shape[0] // 2

    # LR needs upsampling for display
    lr_z_idx = z_idx // 4  # ratio=4
    if lr_z_idx >= lr_vol.shape[0]:
        lr_z_idx = lr_vol.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{case_name}  PSNR={psnr:.2f}dB  SSIM={ssim_val:.4f}', fontsize=16, fontweight='bold')

    # Row 1: LR, SR, GT
    axes[0, 0].imshow(lr_vol[lr_z_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'LR (5mm) z={lr_z_idx}', fontsize=13)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sr_vol[z_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'SR (1mm) z={z_idx}', fontsize=13)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt_vol[z_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'GT (1mm) z={z_idx}', fontsize=13)
    axes[0, 2].axis('off')

    # Row 2: Difference maps
    diff_sr_gt = np.abs(sr_vol[z_idx].astype(np.float64) - gt_vol[z_idx].astype(np.float64))
    axes[1, 0].imshow(diff_sr_gt, cmap='hot', vmin=0, vmax=0.15)
    axes[1, 0].set_title('|SR - GT| Error Map', fontsize=13)
    axes[1, 0].axis('off')

    # Zoomed region (center crop 128x128)
    h, w = sr_vol[z_idx].shape
    cy, cx = h // 2, w // 2
    crop_size = min(64, h // 4, w // 4)
    sr_crop = sr_vol[z_idx, cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    gt_crop = gt_vol[z_idx, cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]

    axes[1, 1].imshow(sr_crop, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('SR (zoom center)', fontsize=13)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gt_crop, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('GT (zoom center)', fontsize=13)
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{case_name}_axial.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def _make_sagittal_comparison(lr_vol, sr_vol, gt_vol, case_name, output_dir, x_idx=None):
    """生成矢状面对比图 (体现 z 方向超分效果)"""
    if x_idx is None:
        x_idx = sr_vol.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{case_name} — Sagittal (x={x_idx})', fontsize=14, fontweight='bold')

    # LR sagittal
    lr_sag = lr_vol[:, :, min(x_idx, lr_vol.shape[2]-1)]
    axes[0].imshow(lr_sag, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f'LR (5mm) [{lr_sag.shape[0]}×{lr_sag.shape[1]}]', fontsize=12)
    axes[0].set_ylabel('z (slice)')
    axes[0].set_xlabel('y')

    sr_sag = sr_vol[:, :, x_idx]
    axes[1].imshow(sr_sag, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f'SR (1mm) [{sr_sag.shape[0]}×{sr_sag.shape[1]}]', fontsize=12)
    axes[1].set_xlabel('y')

    gt_sag = gt_vol[:, :, x_idx]
    axes[2].imshow(gt_sag, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[2].set_title(f'GT (1mm) [{gt_sag.shape[0]}×{gt_sag.shape[1]}]', fontsize=12)
    axes[2].set_xlabel('y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{case_name}_sagittal.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def _make_z_profile(sr_vol, gt_vol, case_name, output_dir):
    """生成 z-profile 曲线 (固定 ROI 的 z 方向强度变化)"""
    h, w = sr_vol.shape[1], sr_vol.shape[2]
    cy, cx = h // 2, w // 2
    roi_size = 16

    sr_roi = sr_vol[:, cy-roi_size:cy+roi_size, cx-roi_size:cx+roi_size].mean(axis=(1, 2))
    gt_roi = gt_vol[:, cy-roi_size:cy+roi_size, cx-roi_size:cx+roi_size].mean(axis=(1, 2))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(gt_roi, label='GT', linewidth=1.5, color='green')
    ax.plot(sr_roi, label='SR', linewidth=1.5, color='blue', linestyle='--')
    ax.set_xlabel('Z slice index')
    ax.set_ylabel('Mean intensity (ROI center 32×32)')
    ax.set_title(f'{case_name} — Z-Profile (center ROI)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{case_name}_zprofile.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def visualize(**kwargs):
    """
    对验证集进行推理并生成可视化对比图

    Usage:
        python val_visualize.py visualize --path_key dataset01_xuanwu --net_idx step4_eagle \\
            --output_dir ../../验证集结果_可视化/
    """
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    output_dir = str(kwargs.pop('output_dir', '../../验证集结果_可视化/'))
    model_path = kwargs.pop('model_path', None)
    compute_ssim = _to_bool(kwargs.pop('compute_ssim', True))
    ssim_batch_size = int(kwargs.pop('ssim_batch_size', 32))
    ssim_stride = int(kwargs.pop('ssim_stride', 1))

    opt.load_config('../config/default.txt')
    config_dict = opt._spec(kwargs)

    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # Load model
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    if model_path and os.path.isfile(model_path):
        use_model_path = model_path
    else:
        best_path = os.path.join(save_model_folder, 'best.pkl')
        if os.path.isfile(best_path):
            use_model_path = best_path
        else:
            save_model_list = sorted([f for f in os.listdir(save_model_folder) if f.endswith('.pkl')])
            use_model_path = os.path.join(save_model_folder, save_model_list[0])

    print(f'Loading model: {use_model_path}')
    save_dict = torch.load(use_model_path, map_location=torch.device('cpu'), weights_only=False)

    if 'config_dict' in save_dict:
        loaded_config = save_dict['config_dict']
        loaded_config.pop('path_img', None)
        loaded_config['mode'] = 'test'
        opt._spec(loaded_config)

    # Device
    GLOBAL_SEED = 2022
    non_model.seed_everything(GLOBAL_SEED)
    device = non_model.resolve_device(opt.gpu_idx)
    print('Use device:', device)

    # Dataset
    val_list, _, _ = non_model.list_paired_cases(opt.path_img, 'val')
    if len(val_list) == 0:
        raise RuntimeError(f'No paired val cases found under path_img={opt.path_img}')
    val_set = val_Dataset(val_list)
    val_batch = Data.DataLoader(dataset=val_set, batch_size=1, shuffle=False,
                                num_workers=opt.test_num_workers)
    print(f'Loaded {len(val_set.img_list)} val cases')

    # Model
    load_net = save_dict['net']
    load_model_dict = load_net.state_dict() if hasattr(load_net, 'state_dict') else load_net
    net = model_TransSR.TVSRN()
    net.load_state_dict(load_model_dict, strict=False)
    del save_dict
    net = net.to(device).eval()

    val_ratio = getattr(opt, 'ratio', 4)
    crop_margin = getattr(opt, 'crop_margin', 3)

    metrics_rows = []

    with torch.no_grad():
        for i, return_list in tqdm(enumerate(val_batch), total=len(val_batch)):
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]

            x_np = x.squeeze().data.numpy()
            y_np = y.squeeze().data.numpy()

            y_pre = np.zeros_like(y_np)
            pos_list_np = pos_list.data.numpy()[0]

            for pos_idx, pos in enumerate(pos_list_np):
                tmp_x = x_np[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                tmp_x_t = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                tmp_y_pre = net(tmp_x_t, ratio=val_ratio)
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                D = y_for_psnr.shape[0]
                pos_z_s = val_ratio * tmp_pos_z + crop_margin
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x
                y_pre[pos_z_s:pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

            # Trim boundary
            sr_vol = y_pre[val_ratio:-val_ratio]
            gt_vol = y_np[val_ratio:-val_ratio]

            # LR volume: subsample from x_np (first patch position covers the volume)
            # x_np shape: [num_patches, c_z, c_y, c_x]
            # For LR display, read the original 5mm volume
            lr_path = os.path.join(opt.path_img, 'val', '5mm', case_name + '.nii.gz')
            if os.path.isfile(lr_path):
                lr_img = sitk.ReadImage(lr_path)
                lr_vol = sitk.GetArrayFromImage(lr_img).astype(np.float32)
                # Normalize to [0,1] same as GT
                if lr_vol.max() > 1.5:
                    lr_vol = np.clip(lr_vol / 4095.0, 0, 1)
            else:
                # Fallback: subsample GT by 4x in z
                lr_vol = gt_vol[::val_ratio]

            # Metrics
            psnr = non_model.cal_psnr(sr_vol, gt_vol)
            if compute_ssim:
                ssim_v = non_model.cal_ssim_volume(gt_vol, sr_vol, device=device,
                                                    batch_size=ssim_batch_size, stride=ssim_stride)
            else:
                ssim_v = 0.0

            metrics_rows.append({
                'case': case_name, 'psnr': psnr, 'ssim': ssim_v,
                'sr_shape': str(sr_vol.shape), 'gt_shape': str(gt_vol.shape),
            })
            print(f'  {case_name}: PSNR={psnr:.2f}, SSIM={ssim_v:.4f}')

            # Generate visualizations
            _make_axial_comparison(lr_vol, sr_vol, gt_vol, case_name, psnr, ssim_v, output_dir)
            _make_sagittal_comparison(lr_vol, sr_vol, gt_vol, case_name, output_dir)
            _make_z_profile(sr_vol, gt_vol, case_name, output_dir)

    # Summary
    mean_psnr = np.mean([r['psnr'] for r in metrics_rows])
    mean_ssim = np.mean([r['ssim'] for r in metrics_rows])
    print(f'\n========== Summary ==========')
    print(f'Mean PSNR: {mean_psnr:.4f}')
    print(f'Mean SSIM: {mean_ssim:.6f}')
    print(f'Cases: {len(metrics_rows)}')

    # Save CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'psnr', 'ssim', 'sr_shape', 'gt_shape'])
        writer.writeheader()
        writer.writerows(metrics_rows)
        writer.writerow({'case': 'MEAN', 'psnr': mean_psnr, 'ssim': mean_ssim})
    print(f'Metrics saved: {csv_path}')

    # Summary montage
    _make_summary_figure(metrics_rows, output_dir)


def _make_summary_figure(metrics_rows, output_dir):
    """生成总体指标汇总图"""
    cases = [r['case'] for r in metrics_rows]
    psnrs = [r['psnr'] for r in metrics_rows]
    ssims = [r['ssim'] for r in metrics_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(cases)*1.2), 8))
    fig.suptitle('Validation Set Metrics Summary', fontsize=14, fontweight='bold')

    x = np.arange(len(cases))
    ax1.bar(x, psnrs, color='steelblue', alpha=0.8)
    ax1.axhline(y=np.mean(psnrs), color='red', linestyle='--', label=f'Mean={np.mean(psnrs):.2f}')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(x, ssims, color='darkorange', alpha=0.8)
    ax2.axhline(y=np.mean(ssims), color='red', linestyle='--', label=f'Mean={np.mean(ssims):.4f}')
    ax2.set_ylabel('SSIM')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary figure saved: {save_path}')


if __name__ == '__main__':
    import fire
    fire.Fire()
