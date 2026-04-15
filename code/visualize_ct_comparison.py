#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT三方向对比可视化: 带正确窗位调整

生成多列对比图: LR(5mm) | Bicubic | Trilinear | TVSRN(baseline) | TVSRN(+z_attn) | GT(1mm)
三行: axial(横断面) / sagittal(矢状面) / coronal(冠状面)
尤其强调z轴超分效果 (sagittal和coronal方向)

窗位说明:
- 脑窗 (Brain): W=80, L=40 → 脑实质
- 骨窗 (Bone): W=2000, L=500 → 颅骨
- 软组织窗 (Soft Tissue): W=400, L=40 → 软组织

用法 (AutoDL):
    cd /root/autodl-tmp/rplhr-ct-training-main/code
    python visualize_ct_comparison.py generate \
        --path_key=dataset01_xuanwu --subset=val --ratio=4 \
        --normalize_ct_input=True \
        --interp_dir=../interp_baseline_xuanwu/nifti \
        --model_dirs='{"baseline":"../val_output/dataset01_xuanwu/xuanwu_finetune","z_attn":"../val_output/dataset01_xuanwu/xuanwu_zattn_finetune"}' \
        --output_dir=../viz_comparison_xuanwu \
        --num_cases=3
"""

import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ===========================================================================
# CT windowing
# ===========================================================================

WINDOW_PRESETS = {
    'brain':       {'W': 80,   'L': 40,   'label': 'Brain (W=80, L=40)'},
    'bone':        {'W': 2000, 'L': 500,  'label': 'Bone (W=2000, L=500)'},
    'soft_tissue': {'W': 400,  'L': 40,   'label': 'Soft Tissue (W=400, L=40)'},
    'lung':        {'W': 1500, 'L': -600, 'label': 'Lung (W=1500, L=-600)'},
    'abdomen':     {'W': 350,  'L': 50,   'label': 'Abdomen (W=350, L=50)'},
}


def apply_ct_window(data, window_width, window_level):
    """
    Apply CT window to data and map to [0, 255].

    Args:
        data: numpy array in HU values
        window_width: W
        window_level: L
    Returns:
        numpy array in [0, 255] uint8
    """
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    result = np.clip((data - lower) / (upper - lower), 0, 1)
    return (result * 255).astype(np.uint8)


def normalized_to_hu(data, normalize_method='auto'):
    """
    将[0,1]归一化数据反映射回HU。

    公开RPLHR-CT数据: (HU + 1024) / 4096 → HU = data * 4096 - 1024
    """
    return data * 4096.0 - 1024.0


def apply_window_to_normalized(data, window_name, already_hu=False):
    """
    对归一化[0,1]数据应用CT窗位。

    Args:
        data: numpy array, 归一化[0,1] 或 HU values
        window_name: 窗位名称 (brain, bone, soft_tissue, etc.)
        already_hu: 数据是否已经是HU值
    Returns:
        uint8 image [0, 255]
    """
    preset = WINDOW_PRESETS[window_name]
    W, L = preset['W'], preset['L']

    if not already_hu:
        hu_data = normalized_to_hu(data)
    else:
        hu_data = data

    return apply_ct_window(hu_data, W, L)


# ===========================================================================
# Normalization helper (standalone)
# ===========================================================================

def _auto_normalize_ct_pair(lr_data, hr_data):
    """Mirror of in_model._auto_normalize_ct_pair."""
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


# ===========================================================================
# Paired case discovery
# ===========================================================================

def list_paired_cases(path_img, subset):
    high_dir = os.path.join(path_img, subset, '1mm')
    low_dir = os.path.join(path_img, subset, '5mm')
    if not os.path.isdir(high_dir) or not os.path.isdir(low_dir):
        return []
    high_cases = {f[:-7] for f in os.listdir(high_dir) if f.endswith('.nii.gz')}
    low_cases = {f[:-7] for f in os.listdir(low_dir) if f.endswith('.nii.gz')}
    return sorted(high_cases & low_cases)


# ===========================================================================
# Visualization
# ===========================================================================

def extract_slices(vol, z_idx=None, y_idx=None, x_idx=None):
    """Extract axial/sagittal/coronal slices at given indices."""
    if z_idx is None:
        z_idx = vol.shape[0] // 2
    if y_idx is None:
        y_idx = vol.shape[1] // 2
    if x_idx is None:
        x_idx = vol.shape[2] // 2

    axial = vol[min(z_idx, vol.shape[0]-1), :, :]
    # For sagittal/coronal, z-axis is vertical — need to handle different z dims
    coronal = vol[:, min(y_idx, vol.shape[1]-1), :]
    sagittal = vol[:, :, min(x_idx, vol.shape[2]-1)]
    return axial, coronal, sagittal


def make_comparison_figure(volumes_dict, case_name, window_name,
                           z_idx=None, y_idx=None, x_idx=None,
                           output_path=None, figsize_scale=3.5):
    """
    生成一张多列×3行对比图。

    Args:
        volumes_dict: OrderedDict of {label: (volume, z_dim)} 
                      volume is 3D numpy array in [0,1] normalized
        case_name: 用于标题
        window_name: CT窗位名称
        z_idx, y_idx, x_idx: 切片位置 (None=自动选中间)
        output_path: 保存路径
    """
    preset = WINDOW_PRESETS[window_name]
    n_cols = len(volumes_dict)

    fig, axes = plt.subplots(3, n_cols, figsize=(figsize_scale * n_cols, figsize_scale * 3.2))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    row_labels = ['Axial (横断面)', 'Coronal (冠状面)', 'Sagittal (矢状面)']

    for col_idx, (label, vol) in enumerate(volumes_dict.items()):
        # Determine indices using GT dimensions
        if z_idx is None:
            _z = vol.shape[0] // 2
        else:
            _z = min(z_idx, vol.shape[0] - 1)
        if y_idx is None:
            _y = vol.shape[1] // 2
        else:
            _y = y_idx
        if x_idx is None:
            _x = vol.shape[2] // 2
        else:
            _x = x_idx

        axial, coronal, sagittal = extract_slices(vol, _z, _y, _x)

        slices = [axial, coronal, sagittal]
        for row_idx, (sl, row_label) in enumerate(zip(slices, row_labels)):
            windowed = apply_window_to_normalized(sl, window_name)
            axes[row_idx, col_idx].imshow(windowed, cmap='gray', aspect='auto')
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

            if row_idx == 0:
                axes[row_idx, col_idx].set_title(label, fontsize=10, fontweight='bold')
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(row_label, fontsize=9)

    fig.suptitle(f'{case_name} — {preset["label"]}', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()


def make_zoomed_comparison(volumes_dict, case_name, window_name,
                           plane='sagittal', slice_idx=None,
                           roi=None, output_path=None):
    """
    生成单方向局部放大对比图（突出z轴超分效果）。

    Args:
        volumes_dict: {label: volume}
        plane: 'axial', 'coronal', 'sagittal'
        slice_idx: 切片索引
        roi: (y_start, y_end, x_start, x_end) 或 None (自动选中心区域)
    """
    n_cols = len(volumes_dict)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7),
                              gridspec_kw={'height_ratios': [1, 1]})
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    preset = WINDOW_PRESETS[window_name]

    for col_idx, (label, vol) in enumerate(volumes_dict.items()):
        if plane == 'axial':
            idx = slice_idx if slice_idx is not None else vol.shape[0] // 2
            sl = vol[min(idx, vol.shape[0]-1), :, :]
        elif plane == 'coronal':
            idx = slice_idx if slice_idx is not None else vol.shape[1] // 2
            sl = vol[:, min(idx, vol.shape[1]-1), :]
        elif plane == 'sagittal':
            idx = slice_idx if slice_idx is not None else vol.shape[2] // 2
            sl = vol[:, :, min(idx, vol.shape[2]-1)]
        else:
            raise ValueError(f"Unknown plane: {plane}")

        windowed = apply_window_to_normalized(sl, window_name)

        # Full view
        axes[0, col_idx].imshow(windowed, cmap='gray', aspect='auto')
        axes[0, col_idx].set_title(label, fontsize=10, fontweight='bold')
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])

        # ROI zoom
        h, w = windowed.shape
        if roi is None:
            # Auto: center 40% crop
            rh, rw = int(h * 0.3), int(w * 0.3)
            ys, ye = h // 2 - rh // 2, h // 2 + rh // 2
            xs, xe = w // 2 - rw // 2, w // 2 + rw // 2
        else:
            ys, ye, xs, xe = roi

        roi_img = windowed[ys:ye, xs:xe]
        axes[1, col_idx].imshow(roi_img, cmap='gray', aspect='auto')
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])

        # Draw ROI box on full view
        from matplotlib.patches import Rectangle
        rect = Rectangle((xs, ys), xe - xs, ye - ys,
                          linewidth=1.5, edgecolor='red', facecolor='none')
        axes[0, col_idx].add_patch(rect)

    fig.suptitle(f'{case_name} — {plane} zoomed — {preset["label"]}',
                 fontsize=11, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


# ===========================================================================
# Main entry
# ===========================================================================

def generate(path_key=None, path_img=None, subset='val', ratio=4,
             normalize_ct_input=False,
             interp_dir=None,
             model_dirs=None,
             output_dir=None,
             windows='brain,bone,soft_tissue',
             num_cases=3,
             z_idx=None, y_idx=None, x_idx=None):
    """
    生成CT三方向对比可视化图。

    Args:
        path_key: 数据集配置key
        path_img: 直接指定数据路径
        subset: val 或 test
        ratio: 超分倍率
        normalize_ct_input: 是否归一化CT数据
        interp_dir: 插值结果目录 (含 nearest/, linear/, cubic/ 子目录)
        model_dirs: JSON字符串, 模型输出目录 {"label": "path"}
        output_dir: 输出目录
        windows: 逗号分隔的窗位列表
        num_cases: 可视化的case数量
        z_idx, y_idx, x_idx: 切片位置 (None=自动)
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
        output_dir = '../viz_comparison_%s' % (path_key or 'custom')
    os.makedirs(output_dir, exist_ok=True)

    # Parse model_dirs
    model_outputs = {}
    if model_dirs:
        if isinstance(model_dirs, str):
            model_outputs = json.loads(model_dirs)
        else:
            model_outputs = model_dirs

    window_list = [w.strip() for w in windows.split(',')]

    # Find cases
    cases = list_paired_cases(path_img, subset)
    if len(cases) == 0:
        print(f"Error: no paired cases in {path_img}/{subset}/")
        sys.exit(1)

    cases = cases[:num_cases]
    print(f"Visualizing {len(cases)} cases with windows: {window_list}")

    for case_name in tqdm(cases, desc='Generating visualizations'):
        # Load LR and GT
        lr_path = os.path.join(path_img, subset, '5mm', case_name + '.nii.gz')
        hr_path = os.path.join(path_img, subset, '1mm', case_name + '.nii.gz')

        lr_vol = sitk.GetArrayFromImage(sitk.ReadImage(lr_path)).astype(np.float32)
        hr_vol = sitk.GetArrayFromImage(sitk.ReadImage(hr_path)).astype(np.float32)

        if normalize_ct_input:
            lr_vol, hr_vol = _auto_normalize_ct_pair(lr_vol, hr_vol)

        # Upsample LR in z for display (nearest for quick reference)
        from scipy.ndimage import zoom as ndimage_zoom
        lr_upsampled = ndimage_zoom(lr_vol, (hr_vol.shape[0] / lr_vol.shape[0], 1.0, 1.0),
                                     order=0, mode='nearest')
        if lr_upsampled.shape[0] > hr_vol.shape[0]:
            lr_upsampled = lr_upsampled[:hr_vol.shape[0]]

        # Build volumes dict (ordered)
        from collections import OrderedDict
        volumes = OrderedDict()
        volumes['5mm LR'] = lr_upsampled

        # Interpolation results
        if interp_dir:
            for method in ['cubic', 'linear']:
                nifti_path = os.path.join(interp_dir, method, case_name + '.nii.gz')
                if os.path.exists(nifti_path):
                    vol = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path)).astype(np.float32)
                    if normalize_ct_input and np.max(vol) > 10:
                        vol = np.clip(vol, 0, 4096) / 4096.0
                    label = 'Bicubic' if method == 'cubic' else 'Trilinear'
                    volumes[label] = vol

        # Model outputs
        for label, model_dir in model_outputs.items():
            nifti_path = os.path.join(model_dir, case_name + '_pre.nii.gz')
            if os.path.exists(nifti_path):
                vol = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path)).astype(np.float32)
                # Model output may have different z-dim (trimmed)
                # Pad to match GT if needed
                if vol.shape[0] < hr_vol.shape[0]:
                    pad_total = hr_vol.shape[0] - vol.shape[0]
                    pad_top = pad_total // 2
                    pad_bot = pad_total - pad_top
                    vol = np.pad(vol, ((pad_top, pad_bot), (0, 0), (0, 0)), mode='edge')
                elif vol.shape[0] > hr_vol.shape[0]:
                    vol = vol[:hr_vol.shape[0]]
                volumes[label] = vol

        volumes['1mm GT'] = hr_vol

        # Generate for each window
        for window_name in window_list:
            if window_name not in WINDOW_PRESETS:
                print(f"  Warning: unknown window '{window_name}', skipping")
                continue

            # 3-plane comparison
            out_path = os.path.join(output_dir, f'{case_name}_{window_name}_3plane.png')
            make_comparison_figure(
                volumes, case_name, window_name,
                z_idx=z_idx, y_idx=y_idx, x_idx=x_idx,
                output_path=out_path
            )

            # Sagittal zoomed (best shows z-axis SR quality)
            out_path_zoom = os.path.join(output_dir, f'{case_name}_{window_name}_sagittal_zoom.png')
            make_zoomed_comparison(
                volumes, case_name, window_name,
                plane='sagittal', output_path=out_path_zoom
            )

            # Coronal zoomed
            out_path_zoom_cor = os.path.join(output_dir, f'{case_name}_{window_name}_coronal_zoom.png')
            make_zoomed_comparison(
                volumes, case_name, window_name,
                plane='coronal', output_path=out_path_zoom_cor
            )

    print(f"\nVisualization saved to: {output_dir}")
    print(f"  {len(cases)} cases × {len(window_list)} windows × 3 figures = {len(cases) * len(window_list) * 3} images")


if __name__ == '__main__':
    import fire
    fire.Fire()
