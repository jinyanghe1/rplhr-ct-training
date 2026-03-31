import numpy as np
import SimpleITK as sitk
import cv2
import os
import sys
import json
import random
import torch
import torch.nn.functional as F
import csv

from builtins import range
import math
from math import exp

from config import opt

################################## For config ##################################
def read_kwargs(kwargs):
    ####### Common Setting #######
    if 'path_key' not in kwargs:
        print('Error: no path key')
        sys.exit()
    else:
        dict_path = '../config/%s_dict.json' % kwargs['path_key']
        with open(dict_path, 'r') as f:
            data_info_dict = json.load(f)
        # Allow CLI path_img override for quick experiments.
        if 'path_img' not in kwargs or kwargs['path_img'] in [None, '']:
            kwargs['path_img'] = data_info_dict['path_img']

    if 'net_idx' not in kwargs:
        print('Error: no net idx')
        sys.exit()

    ####### Special Setting #######
    # cycle learning
    if 'cycle_r' in kwargs and int(kwargs['cycle_r']) > 0:
        if 'Tmax' not in kwargs:
            kwargs['Tmax'] = 100
        kwargs['cos_lr'] = True
        kwargs['epoch'] = int(kwargs['cycle_r']) * 2 * kwargs['Tmax']
        kwargs['gap_epoch'] = kwargs['epoch'] + 1
        kwargs['optim'] = 'SGD'

    # optim set
    if 'optim' in kwargs and kwargs['optim'] == 'SGD':
        if 'wd' not in kwargs:
            kwargs['wd'] = 0.00001
        if 'lr' not in kwargs:
            kwargs['lr'] = 0.01

    return kwargs, data_info_dict

def update_kwargs(init_model_path, kwargs):
    save_dict = torch.load(init_model_path, map_location=torch.device('cpu'))
    config_dict = save_dict['config_dict']
    del save_dict

    config_dict.pop('gpu_idx', None)
    config_dict['mode'] = 'test'

    if 'val_bs' in kwargs:
        config_dict['val_bs'] = kwargs['val_bs']

    return config_dict

def resolve_device(gpu_idx=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{int(gpu_idx)}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clear_device_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def list_paired_cases(path_img, subset, high_res='1mm', low_res='5mm'):
    """
    Return sorted case names that exist in both high/low resolution folders.
    """
    high_dir = os.path.join(path_img, subset, high_res)
    low_dir = os.path.join(path_img, subset, low_res)

    if not os.path.isdir(high_dir) or not os.path.isdir(low_dir):
        return [], [], []

    high_cases = {f[:-7] for f in os.listdir(high_dir) if f.endswith('.nii.gz')}
    low_cases = {f[:-7] for f in os.listdir(low_dir) if f.endswith('.nii.gz')}

    common_cases = sorted(high_cases & low_cases)
    high_only = sorted(high_cases - low_cases)
    low_only = sorted(low_cases - high_cases)
    return common_cases, high_only, low_only

def save_metric_history(metric_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metric_history, f, indent=2)

    csv_path = os.path.join(save_dir, 'metrics.csv')
    if len(metric_history) == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_psnr', 'val_ssim', 'epoch_sec', 'val_sec'])
        return

    fieldnames = ['epoch', 'lr', 'train_loss', 'val_psnr', 'val_ssim', 'epoch_sec', 'val_sec']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for each in metric_history:
            writer.writerow(each)

def plot_metric_history(metric_history, save_dir):
    if len(metric_history) == 0:
        return

    try:
        mpl_cache_dir = '/tmp/matplotlib-cache'
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ.setdefault('MPLCONFIGDIR', mpl_cache_dir)
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'Warning: matplotlib is not available, skip plotting. {e}')
        return

    try:
        epochs = [m['epoch'] for m in metric_history]
        train_loss = [m['train_loss'] for m in metric_history]
        val_psnr = [m['val_psnr'] for m in metric_history]
        val_ssim = [m['val_ssim'] for m in metric_history]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(epochs, train_loss, marker='o', linewidth=1.8)
        axes[0].set_ylabel('Train Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Training Metrics')

        axes[1].plot(epochs, val_psnr, marker='o', linewidth=1.8)
        axes[1].set_ylabel('Val PSNR')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, val_ssim, marker='o', linewidth=1.8)
        axes[2].set_ylabel('Val SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'metrics.png')
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f'Warning: failed to generate metrics plot. {e}')

################################## For Metric ##################################
def cal_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 40
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

_SSIM_MODULE_CACHE = {}

def ssim(img1, img2, window_size=11, size_average=True):
    key = (
        int(window_size),
        bool(size_average),
        str(img1.device),
        str(img1.dtype),
        img1.size(1),
    )
    ssim_module = _SSIM_MODULE_CACHE.get(key)
    if ssim_module is None:
        ssim_module = SSIM(window_size=window_size, size_average=size_average)
        _SSIM_MODULE_CACHE[key] = ssim_module
    return ssim_module(img1, img2)

def cal_ssim(img1, img2, cuda_use=None, device=None):
    img1 = torch.from_numpy(np.ascontiguousarray(img1)).unsqueeze(0).unsqueeze(0).float()
    img2 = torch.from_numpy(np.ascontiguousarray(img2)).unsqueeze(0).unsqueeze(0).float()

    if device is None and cuda_use is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{int(cuda_use)}')

    if device is not None:
        img1 = img1.to(device)
        img2 = img2.to(device)

    return ssim(img1, img2).data.cpu().numpy()

def cal_ssim_volume(img1, img2, cuda_use=None, device=None, batch_size=32, stride=1, window_size=11):
    """
    Compute mean SSIM over a 3D volume by batching 2D slices.
    When stride == 1, this matches averaging per-slice SSIM.
    """
    if device is None and cuda_use is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{int(cuda_use)}')

    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if stride <= 0:
        raise ValueError('stride must be a positive integer')

    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)

    if img1.shape != img2.shape:
        raise ValueError(f'img1 and img2 must have same shape, got {img1.shape} vs {img2.shape}')

    # Keep backward compatibility for 2D usage.
    if img1.ndim == 2:
        return float(cal_ssim(img1, img2, cuda_use=cuda_use, device=device))
    if img1.ndim != 3:
        raise ValueError(f'Only 2D/3D arrays are supported, got ndim={img1.ndim}')

    if stride > 1:
        img1 = img1[::stride]
        img2 = img2[::stride]

    total_ssim = 0.0
    total_slices = 0

    for start in range(0, img1.shape[0], batch_size):
        end = min(start + batch_size, img1.shape[0])

        batch_img1 = torch.from_numpy(np.ascontiguousarray(img1[start:end])).unsqueeze(1)
        batch_img2 = torch.from_numpy(np.ascontiguousarray(img2[start:end])).unsqueeze(1)

        if device is not None:
            batch_img1 = batch_img1.to(device, non_blocking=True)
            batch_img2 = batch_img2.to(device, non_blocking=True)

        batch_ssim = ssim(batch_img1, batch_img2, window_size=window_size, size_average=False)
        total_ssim += batch_ssim.sum().item()
        total_slices += batch_ssim.numel()

    if total_slices == 0:
        return 0.0
    return total_ssim / total_slices
