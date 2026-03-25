# -*- coding: utf-8 -*-
"""
Full validation script with detailed metrics and visualization.
Generates:
1. Table of metrics for each validation sample
2. Comparison visualization (input, output, ground truth)
3. Summary statistics
Includes: PSNR, MSE, SSIM, LPIPS_alex, LPIPS_vgg
"""
import os
import random
import json
import csv
import time

import torch
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset import val_Dataset
from net import model_TransSR

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import LPIPS
import lpips

def visualize_comparison(lr_img, sr_img, hr_img, case_name, save_path, slice_idx=None):
    """
    Create visualization comparing LR input, SR output, and HR ground truth.
    """
    if slice_idx is None:
        slice_idx = sr_img.shape[0] // 2
    
    sr_slice = sr_img[slice_idx]
    hr_slice = hr_img[slice_idx]
    
    lr_z = slice_idx // 5
    if lr_z >= lr_img.shape[0]:
        lr_z = lr_img.shape[0] - 1
    lr_slice = lr_img[lr_z]
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(lr_slice, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Input (5mm)\nZ={lr_z}', fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(sr_slice, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'SR Output (1mm)\nZ={slice_idx}', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(hr_slice, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'Ground Truth (1mm)\nZ={slice_idx}', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Case: {case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_error_map(sr_img, hr_img, case_name, save_path, slice_idx=None):
    """
    Create error map visualization.
    """
    if slice_idx is None:
        slice_idx = sr_img.shape[0] // 2
    
    sr_slice = sr_img[slice_idx]
    hr_slice = hr_img[slice_idx]
    error = np.abs(sr_slice - hr_slice)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(sr_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('SR Output')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(hr_slice, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    im3 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=error.max())
    axes[2].set_title(f'Absolute Error (MSE: {np.mean(error**2):.6f})')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Error Map - Case: {case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_lpips_for_volume(sr_vol, hr_vol, loss_fn, device):
    """
    Compute LPIPS for a 3D volume by averaging over all slices.
    Args:
        sr_vol: numpy array (D, H, W) - super resolution output
        hr_vol: numpy array (D, H, W) - ground truth
        loss_fn: LPIPS loss function
        device: torch device
    Returns:
        mean LPIPS score
    """
    lpips_scores = []
    
    for z_idx in range(sr_vol.shape[0]):
        sr_slice = sr_vol[z_idx]
        hr_slice = hr_vol[z_idx]
        
        # Convert to torch tensor and add batch/channel dimensions: (1, 1, H, W)
        sr_tensor = torch.from_numpy(sr_slice).float().unsqueeze(0).unsqueeze(0).to(device)
        hr_tensor = torch.from_numpy(hr_slice).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # LPIPS expects 3-channel images, so we need to repeat
        sr_tensor = sr_tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        hr_tensor = hr_tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        
        with torch.no_grad():
            lpips_val = loss_fn(sr_tensor, hr_tensor)
            lpips_scores.append(lpips_val.item())
    
    return np.mean(lpips_scores)

def val(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/default.txt')
    config_dict = opt._spec(kwargs)

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    save_output_folder = '../val_output/%s/%s/' % (opt.path_key, str(opt.net_idx))
    save_viz_folder = '../val_viz/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(save_output_folder, exist_ok=True)
    os.makedirs(save_viz_folder, exist_ok=True)

    # stage 3 - Load best model
    save_model_list = sorted(os.listdir(save_model_folder))
    use_model = [each for each in save_model_list if each.endswith('pkl') and 'best' in each]
    if len(use_model) == 0:
        use_model = [each for each in save_model_list if each.endswith('pkl')][0]
    else:
        use_model = use_model[0]
    use_model_path = save_model_folder + use_model
    
    print('=' * 70)
    print('Loading model:', use_model)
    print('=' * 70)
    
    config_dict = non_model.update_kwargs(use_model_path, kwargs)
    opt._spec(config_dict)
    print('load config done')

    # stage 4 Dataloader Setting
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    GLOBAL_SEED = 2022
    non_model.seed_everything(GLOBAL_SEED)

    ###### Device ######
    device = non_model.resolve_device(opt.gpu_idx)
    print('Use device:', device)

    save_model_path = save_model_folder + use_model
    save_dict = torch.load(save_model_path, map_location=torch.device('cpu'), weights_only=False)
    config_dict = save_dict['config_dict']
    config_dict.pop('path_img')
    config_dict['mode'] = 'test'
    opt._spec(config_dict)

    # val set
    val_dir = os.path.join(opt.path_img, 'val', '1mm')
    val_list = [each.split('.')[0] for each in sorted(os.listdir(val_dir))]
    val_set = val_Dataset(val_list)
    val_data_num = len(val_set.img_list)
    val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
                                num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
    print('load val data done, num =', val_data_num)

    load_net = save_dict['net']
    load_model_dict = load_net.state_dict()

    net = model_TransSR.TVSRN()
    net.load_state_dict(load_model_dict, strict=False)

    del save_dict
    net = net.to(device)
    net = net.eval()

    # Initialize LPIPS models
    print('Initializing LPIPS models...')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    print('LPIPS models initialized.')

    # Results storage
    results = []
    
    # Timing benchmark
    total_inference_time = 0
    
    with torch.no_grad():
        for i, return_list in tqdm(enumerate(val_batch)):
            case_start_time = time.time()
            
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]

            # Get LR input (5mm) for visualization
            lr_path = os.path.join(opt.path_img, 'val', '5mm', case_name + '.nii.gz')
            lr_img = sitk.GetArrayFromImage(sitk.ReadImage(lr_path))

            x = x.squeeze().data.numpy()
            y = y.squeeze().data.numpy()

            y_pre = np.zeros_like(y)
            pos_list = pos_list.data.numpy()[0]

            for pos_idx, pos in enumerate(pos_list):
                tmp_x = x[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                tmp_y_pre = net(tmp_x)
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                D = y_for_psnr.shape[0]
                pos_z_s = 5 * tmp_pos_z + 3
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x

                y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s + opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

            del tmp_y_pre, tmp_x

            y_pre = y_pre[5:-5]
            y = y[5:-5]
            
            # Save predicted output
            save_name_pre = save_output_folder + '%s_pre.nii.gz' % case_name
            output_pre = sitk.GetImageFromArray(y_pre)
            sitk.WriteImage(output_pre, save_name_pre)

            # Calculate traditional metrics
            psnr = non_model.cal_psnr(y_pre, y)
            mse = non_model.cal_mse(y_pre, y)
            
            # Calculate SSIM per slice and average
            pid_ssim_list = []
            for z_idx, z_layer in enumerate(y_pre):
                mask_layer = y[z_idx]
                tmp_ssim = non_model.cal_ssim(mask_layer, z_layer, device=device)
                pid_ssim_list.append(tmp_ssim)
            ssim_val = np.mean(pid_ssim_list)

            # Calculate LPIPS metrics
            lpips_alex = compute_lpips_for_volume(y_pre, y, loss_fn_alex, device)
            lpips_vgg = compute_lpips_for_volume(y_pre, y, loss_fn_vgg, device)

            case_inference_time = time.time() - case_start_time
            total_inference_time += case_inference_time

            results.append({
                'case_name': case_name,
                'psnr': float(psnr),
                'mse': float(mse),
                'ssim': float(ssim_val),
                'lpips_alex': float(lpips_alex),
                'lpips_vgg': float(lpips_vgg),
                'inference_time': float(case_inference_time)
            })
            
            # Generate visualizations
            viz_path = save_viz_folder + '%s_comparison.png' % case_name
            visualize_comparison(lr_img, y_pre, y, case_name, viz_path)
            
            error_path = save_viz_folder + '%s_error.png' % case_name
            visualize_error_map(y_pre, y, case_name, error_path)
            
            print(f"Case {case_name}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, "
                  f"LPIPS_alex={lpips_alex:.4f}, LPIPS_vgg={lpips_vgg:.4f}, "
                  f"Time={case_inference_time:.2f}s")

    # Calculate summary statistics
    psnr_list = [r['psnr'] for r in results]
    mse_list = [r['mse'] for r in results]
    ssim_list = [r['ssim'] for r in results]
    lpips_alex_list = [r['lpips_alex'] for r in results]
    lpips_vgg_list = [r['lpips_vgg'] for r in results]
    time_list = [r['inference_time'] for r in results]
    
    summary = {
        'mean_psnr': float(np.mean(psnr_list)),
        'std_psnr': float(np.std(psnr_list)),
        'mean_mse': float(np.mean(mse_list)),
        'std_mse': float(np.std(mse_list)),
        'mean_ssim': float(np.mean(ssim_list)),
        'std_ssim': float(np.std(ssim_list)),
        'mean_lpips_alex': float(np.mean(lpips_alex_list)),
        'std_lpips_alex': float(np.std(lpips_alex_list)),
        'mean_lpips_vgg': float(np.mean(lpips_vgg_list)),
        'std_lpips_vgg': float(np.std(lpips_vgg_list)),
        'best_psnr': float(np.max(psnr_list)),
        'worst_psnr': float(np.min(psnr_list)),
        'avg_inference_time': float(np.mean(time_list)),
        'total_inference_time': float(np.sum(time_list))
    }
    
    # Save results to CSV
    csv_path = save_output_folder + 'validation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case Name', 'PSNR (dB)', 'MSE', 'SSIM', 'LPIPS_alex', 'LPIPS_vgg', 'Time (s)'])
        for r in results:
            writer.writerow([r['case_name'], r['psnr'], r['mse'], r['ssim'], 
                           r['lpips_alex'], r['lpips_vgg'], r['inference_time']])
        writer.writerow([])
        writer.writerow(['Summary Statistics', '', '', '', '', '', ''])
        writer.writerow(['Metric', 'Mean', 'Std', '', '', '', ''])
        writer.writerow(['PSNR (dB)', summary['mean_psnr'], summary['std_psnr'], '', '', '', ''])
        writer.writerow(['MSE', summary['mean_mse'], summary['std_mse'], '', '', '', ''])
        writer.writerow(['SSIM', summary['mean_ssim'], summary['std_ssim'], '', '', '', ''])
        writer.writerow(['LPIPS_alex', summary['mean_lpips_alex'], summary['std_lpips_alex'], '', '', '', ''])
        writer.writerow(['LPIPS_vgg', summary['mean_lpips_vgg'], summary['std_lpips_vgg'], '', '', '', ''])
        writer.writerow(['Avg Inference Time (s)', summary['avg_inference_time'], '', '', '', '', ''])
    
    # Save results to JSON
    json_path = save_output_folder + 'validation_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'per_sample': results,
            'summary': summary
        }, f, indent=2)
    
    # Print summary
    print('\n' + '=' * 70)
    print('VALIDATION RESULTS SUMMARY (with LPIPS)')
    print('=' * 70)
    print(f'Number of samples: {len(results)}')
    print(f'PSNR: {summary["mean_psnr"]:.4f} ± {summary["std_psnr"]:.4f} dB')
    print(f'  (Best: {summary["best_psnr"]:.4f}, Worst: {summary["worst_psnr"]:.4f})')
    print(f'MSE:  {summary["mean_mse"]:.6f} ± {summary["std_mse"]:.6f}')
    print(f'SSIM: {summary["mean_ssim"]:.4f} ± {summary["std_ssim"]:.4f}')
    print(f'LPIPS (Alex): {summary["mean_lpips_alex"]:.4f} ± {summary["std_lpips_alex"]:.4f}')
    print(f'LPIPS (VGG):  {summary["mean_lpips_vgg"]:.4f} ± {summary["std_lpips_vgg"]:.4f}')
    print()
    print(f'Average inference time per case: {summary["avg_inference_time"]:.2f}s')
    print(f'Total inference time: {summary["total_inference_time"]:.2f}s')
    print('=' * 70)
    print(f'Results saved to: {save_output_folder}')
    print(f'Visualizations saved to: {save_viz_folder}')
    print('=' * 70)

if __name__ == '__main__':
    import fire

    fire.Fire()
