#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集三方向截面可视化脚本

在AutoDL上运行，生成超分结果 vs 低分辨率输入 vs Ground Truth 在三个方向（轴位/冠状/矢状）的截面对比图。

用法:
    cd /root/autodl-tmp/rplhr-ct-training-main/code
    python visualize_3plane.py val \
        --path_key=SRM \
        --net_idx=pretrain_ratio5 \
        --output_dir=../val_viz_3plane \
        --num_samples=5

注意:
    - 需要已训练好的模型 (model/{path_key}/{net_idx}/best_model.pkl)
    - 需要验证数据集
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import SimpleITK as sitk
import torch
from tqdm import tqdm

from config import opt
from utils import non_model
from make_dataset import val_Dataset
from net import model_TransSR

import warnings
warnings.filterwarnings("ignore")

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_3plane_comparison(sr_vol, lr_vol, gt_vol, case_name, output_dir, 
                                 slice_indices=None, psnr_val=None, ssim_val=None):
    """
    生成三个方向的截面对比图
    
    Args:
        sr_vol: 超分结果 (Z, Y, X)
        lr_vol: 低分辨率输入 (Z, Y, X) 
        gt_vol: Ground Truth (Z, Y, X)
        case_name: 样本名
        output_dir: 输出目录
        slice_indices: 三个方向的切片索引 [z, y, x]
        psnr_val: PSNR值
        ssim_val: SSIM值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动选择切片位置（体积的中间位置）
    if slice_indices is None:
        slice_indices = [
            sr_vol.shape[0] // 2,  # Z方向中间
            sr_vol.shape[1] // 2,  # Y方向中间
            sr_vol.shape[2] // 2,  # X方向中间
        ]
    
    z_idx, y_idx, x_idx = slice_indices
    
    # 三个方向的切片
    planes = [
        ('Axial (Z轴)', sr_vol[z_idx, :, :], lr_vol[z_idx, :, :], gt_vol[z_idx, :, :], f'z={z_idx}'),
        ('Coronal (Y轴)', sr_vol[:, y_idx, :], lr_vol[:, y_idx, :], gt_vol[:, y_idx, :], f'y={y_idx}'),
        ('Sagittal (X轴)', sr_vol[:, :, x_idx], lr_vol[:, :, x_idx], gt_vol[:, :, x_idx], f'x={x_idx}'),
    ]
    
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.05)
    
    col_titles = ['Low Resolution (Input)', 'Super-Resolution (Output)', 'Ground Truth']
    
    for row_idx, (plane_name, sr_slice, lr_slice, gt_slice, slice_info) in enumerate(planes):
        # 确定显示范围
        vmin = min(sr_slice.min(), lr_slice.min(), gt_slice.min())
        vmax = max(sr_slice.max(), lr_slice.max(), gt_slice.max())
        
        for col_idx, (data, title) in enumerate(zip(
            [lr_slice, sr_slice, gt_slice], col_titles
        )):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(f'{title}\n{plane_name} ({slice_info})', fontsize=10)
            ax.axis('off')
            
            # 最后一行加colorbar
            if row_idx == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 添加指标信息
    metric_text = f'{case_name}'
    if psnr_val is not None:
        metric_text += f'  |  PSNR: {psnr_val:.2f} dB'
    if ssim_val is not None:
        metric_text += f'  |  SSIM: {ssim_val:.4f}'
    
    fig.suptitle(metric_text, fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(output_dir, f'{case_name}_3plane_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def visualize_difference_map(sr_vol, gt_vol, case_name, output_dir, slice_indices=None):
    """
    生成三个方向的差异图
    
    Args:
        sr_vol: 超分结果 (Z, Y, X)
        gt_vol: Ground Truth (Z, Y, X)
        case_name: 样本名
        output_dir: 输出目录
        slice_indices: 三个方向的切片索引 [z, y, x]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if slice_indices is None:
        slice_indices = [
            sr_vol.shape[0] // 2,
            sr_vol.shape[1] // 2,
            sr_vol.shape[2] // 2,
        ]
    
    z_idx, y_idx, x_idx = slice_indices
    
    # 计算差异
    diff = np.abs(sr_vol - gt_vol)
    
    planes = [
        ('Axial', diff[z_idx, :, :], gt_vol[z_idx, :, :], sr_vol[z_idx, :, :]),
        ('Coronal', diff[:, y_idx, :], gt_vol[:, y_idx, :], sr_vol[:, y_idx, :]),
        ('Sagittal', diff[:, :, x_idx], gt_vol[:, :, x_idx], sr_vol[:, :, x_idx]),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'{case_name} - Difference Map', fontsize=14, fontweight='bold', y=0.98)
    
    for row_idx, (plane_name, diff_slice, gt_slice, sr_slice) in enumerate(planes):
        # GT
        axes[row_idx, 0].imshow(gt_slice, cmap='gray', aspect='auto')
        axes[row_idx, 0].set_title(f'GT - {plane_name}')
        axes[row_idx, 0].axis('off')
        
        # SR
        axes[row_idx, 1].imshow(sr_slice, cmap='gray', aspect='auto')
        axes[row_idx, 1].set_title(f'SR - {plane_name}')
        axes[row_idx, 1].axis('off')
        
        # Difference
        im = axes[row_idx, 2].imshow(diff_slice, cmap='hot', aspect='auto')
        axes[row_idx, 2].set_title(f'|SR-GT| - {plane_name}')
        axes[row_idx, 2].axis('off')
        plt.colorbar(im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
    
    save_path = os.path.join(output_dir, f'{case_name}_3plane_difference.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def val(**kwargs):
    """主验证函数 - 生成三方向截面对比可视化"""
    # Stage 1: 配置
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/default.txt')
    config_dict = opt._spec(kwargs)
    
    output_dir = kwargs.get('output_dir', '../val_viz_3plane')
    num_samples = int(kwargs.get('num_samples', 10))
    
    # Stage 2: 加载模型
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    save_output_folder = '../val_output/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找模型文件
    save_model_list = sorted(os.listdir(save_model_folder))
    use_model = [each for each in save_model_list if each.endswith('pkl')][0]
    use_model_path = save_model_folder + use_model
    
    print(f'Loading model: {use_model_path}')
    save_dict = torch.load(use_model_path, map_location='cpu')
    
    # 恢复配置
    config_dict = save_dict.get('config_dict', {})
    config_dict.pop('path_img', None)
    config_dict['mode'] = 'test'
    opt._spec(config_dict)
    
    # Stage 3: 初始化模型
    device = non_model.resolve_device(opt.gpu_idx)
    print(f'Use device: {device}')
    
    load_net = save_dict['net']
    load_model_dict = load_net.state_dict()
    
    net = model_TransSR.TVSRN()
    net.load_state_dict(load_model_dict, strict=False)
    net = net.to(device)
    net = net.eval()
    
    # 获取模型信息
    best_psnr = save_dict.get('psnr', 'N/A')
    best_epoch = save_dict.get('epoch', 'N/A')
    print(f'Model epoch: {best_epoch}, PSNR: {best_psnr}')
    
    del save_dict
    
    # Stage 4: 加载验证数据
    val_dir = os.path.join(opt.path_img, 'val', '1mm')
    val_list = [each.split('.')[0] for each in sorted(os.listdir(val_dir))]
    val_set = val_Dataset(val_list)
    val_batch = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=1, shuffle=False,
        num_workers=2
    )
    
    print(f'Validation samples: {len(val_list)}')
    print(f'Output directory: {output_dir}')
    
    # Stage 5: 逐样本推理+可视化
    with torch.no_grad():
        psnr_list = []
        ssim_list = []
        
        for i, return_list in tqdm(enumerate(val_batch)):
            if i >= num_samples:
                break
                
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]
            
            x_np = x.squeeze().data.numpy()
            y_np = y.squeeze().data.numpy()
            
            # 构建超分结果
            ratio = getattr(opt, 'ratio', 5)
            crop_margin = getattr(opt, 'crop_margin', 3)
            
            y_pre = np.zeros_like(y_np)
            pos_list_np = pos_list.data.numpy()[0]
            
            # 同时构建低分辨率上采样结果（用于对比）
            # 将LR输入简单插值到HR尺寸
            from scipy.ndimage import zoom as scipy_zoom
            lr_upsampled = np.zeros_like(y_np)
            
            for pos_idx, pos in enumerate(pos_list_np):
                tmp_x = x_np[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos
                
                # SR inference
                tmp_x_tensor = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                
                # 使用ratio参数
                if hasattr(net, 'forward') and 'ratio' in net.forward.__code__.co_varnames:
                    tmp_y_pre = net(tmp_x_tensor, ratio=ratio)
                else:
                    tmp_y_pre = net(tmp_x_tensor)
                
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()
                
                D = y_for_psnr.shape[0]
                pos_z_s = ratio * tmp_pos_z + crop_margin
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x
                
                y_pre[pos_z_s:pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr
                
                # LR上采样（简单最近邻插值用于对比）
                lr_block = tmp_x  # (c_z, c_y, c_x) 低分辨率块
                # 在z方向上采样
                zoom_factors = [ratio, 1, 1]
                lr_upsampled_block = scipy_zoom(lr_block, zoom_factors, order=0)
                
                # 裁剪到与SR输出相同尺寸
                if lr_upsampled_block.shape[0] > D:
                    lr_upsampled_block = lr_upsampled_block[:D]
                elif lr_upsampled_block.shape[0] < D:
                    pad = D - lr_upsampled_block.shape[0]
                    lr_upsampled_block = np.pad(lr_upsampled_block, ((0,pad),(0,0),(0,0)))
                
                lr_upsampled[pos_z_s:pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = lr_upsampled_block
                
                del tmp_y_pre, tmp_x_tensor
            
            # 裁剪边缘
            margin = crop_margin + 2
            y_pre_eval = y_pre[margin:-margin]
            y_eval = y_np[margin:-margin]
            lr_eval = lr_upsampled[margin:-margin]
            
            # 计算指标
            psnr = non_model.cal_psnr(y_pre_eval, y_eval)
            psnr_list.append(psnr)
            
            pid_ssim_list = []
            for z_idx, z_layer in enumerate(y_pre_eval):
                mask_layer = y_eval[z_idx]
                tmp_ssim = non_model.cal_ssim(mask_layer, z_layer, device=device)
                pid_ssim_list.append(tmp_ssim)
            ssim_val = np.mean(pid_ssim_list)
            ssim_list.append(ssim_val)
            
            print(f'\n{case_name}: PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}')
            
            # 生成三方向截面对比
            visualize_3plane_comparison(
                sr_vol=y_pre_eval, 
                lr_vol=lr_eval, 
                gt_vol=y_eval,
                case_name=case_name, 
                output_dir=output_dir,
                psnr_val=psnr,
                ssim_val=ssim_val
            )
            
            # 生成差异图
            visualize_difference_map(
                sr_vol=y_pre_eval,
                gt_vol=y_eval,
                case_name=case_name,
                output_dir=output_dir
            )
    
    # 输出统计
    print('\n' + '=' * 60)
    print(f'Average PSNR: {np.mean(psnr_list):.2f} dB')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')
    print(f'Output directory: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    import fire
    fire.Fire()
