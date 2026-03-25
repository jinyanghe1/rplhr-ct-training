# -*- coding: utf-8 -*-
"""
支持混合Ratio的训练脚本

特点:
- 支持不同case有不同的超分比例 (如4x, 5x混合)
- 支持灵活的路径命名 (thick/thin 或 5mm/1mm)
- 支持自动检测ratio或从配置文件读取

使用方法:
    python train_mixed.py train --path_key DATASET --gpu_idx 0 --net_idx TVSRN --lr_dir thick --hr_dir thin
"""
import os
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset import train_Dataset, val_Dataset
from net import model_TransSR

import numpy as np
from tqdm import tqdm

import cv2
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))

import json
import csv
from datetime import datetime

# 导入混合ratio支持
from utils import in_model_mixed as in_model


def get_case_list_mixed(img_path, subset='train', hr_dir='thin'):
    """
    获取病例列表（支持多种目录命名）
    
    Args:
        img_path: 数据根目录
        subset: 'train', 'val', 或 'test'
        hr_dir: 高分辨率目录名
    
    Returns:
        list: 病例名称列表
    """
    # 尝试多种可能的目录名
    possible_hr_names = [hr_dir, '1mm', 'thin', 'HR']
    
    for hr_name in possible_hr_names:
        hr_path = os.path.join(img_path, subset, hr_name)
        if os.path.exists(hr_path):
            cases = [f.replace('.nii.gz', '') for f in sorted(os.listdir(hr_path)) 
                     if f.endswith('.nii.gz')]
            if cases:
                print(f"从 {subset}/{hr_name} 加载 {len(cases)} 个cases")
                return cases
    
    raise FileNotFoundError(f"找不到HR数据目录: {img_path}/{subset}/(尝试了: {possible_hr_names})")


def calculate_val_position(tmp_pos_z, ratio, offset=3):
    """
    计算验证时的输出位置（支持任意ratio）
    
    Args:
        tmp_pos_z: 输入Z位置
        ratio: 超分比例
        offset: 偏移量
    
    Returns:
        int: 输出Z位置
    """
    return ratio * tmp_pos_z + offset


def train(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/default.txt')
    config_dict = opt._spec(kwargs)

    ###### random setting ######
    GLOBAL_SEED = 2022
    non_model.seed_everything(GLOBAL_SEED)

    ###### Device ######
    device = non_model.resolve_device(opt.gpu_idx)
    print('Use device:', device)
    
    # 打印混合ratio模式信息
    print('=' * 70)
    print('混合Ratio训练模式 (Mixed Ratio Training)')
    print('=' * 70)
    
    # 获取路径配置
    lr_dir = getattr(opt, 'lr_dir', 'thick')
    hr_dir = getattr(opt, 'hr_dir', 'thin')
    print(f'LR目录: {lr_dir}, HR目录: {hr_dir}')
    
    # 检查ratio配置
    ratio_config_path = os.path.join(opt.path_img, 'case_ratio_config.json')
    if os.path.exists(ratio_config_path):
        print(f'Ratio配置文件: {ratio_config_path}')
        with open(ratio_config_path, 'r') as f:
            ratio_config = json.load(f)
        ratio_counts = {}
        for r in ratio_config.values():
            ratio_counts[r] = ratio_counts.get(r, 0) + 1
        print('Ratio分布:')
        for r, count in sorted(ratio_counts.items()):
            print(f'  {r}x: {count} cases')
    else:
        print(f'使用全局ratio: {opt.ratio}x')
        print(f'(如需为每个case指定不同ratio，请创建: {ratio_config_path})')
    print('=' * 70)

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(save_model_folder, exist_ok=True)
    
    # Create output folder for training logs
    save_log_folder = '../train_log/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(save_log_folder, exist_ok=True)

    ###### network ######
    net = model_TransSR.TVSRN().to(device)

    ###### optim ######
    lr = opt.lr
    if opt.optim == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=lr, weight_decay=opt.wd, momentum=0.9)
        print('================== SGD lr = %.6f ==================' % lr)

    elif opt.optim == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                      lr=lr, weight_decay=opt.wd)
        print('================== Adam lr = %.6f ==================' % lr)

    elif opt.optim == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                      lr=lr, weight_decay=opt.wd)
        print('================== AdamW lr = %.6f ==================' % lr)

    if opt.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Tmax, \
                                                       eta_min=opt.lr / opt.lr_gap)
    elif opt.Tmin == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=opt.patience, threshold=0.000001)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                               patience=opt.patience, threshold=0.000001)

    ###### loss ######
    print('Use %s loss'%opt.loss_f)
    train_criterion = nn.L1Loss()

    ###### Dataloader Setting ######
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    # 使用混合ratio兼容的病例列表获取
    train_list = get_case_list_mixed(opt.path_img, 'train', hr_dir)
    val_list = get_case_list_mixed(opt.path_img, 'val', hr_dir)

    train_set = train_Dataset(train_list)
    train_data_num = len(train_set.img_list)
    train_batch = Data.DataLoader(dataset=train_set, batch_size=opt.train_bs, shuffle=True, \
                                  num_workers=opt.num_workers, worker_init_fn=worker_init_fn, \
                                  drop_last=True)
    print('load train data done, num =', train_data_num)

    val_set = val_Dataset(val_list)
    val_data_num = len(val_set.img_list)
    val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
                                num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
    print('load val data done, num =', val_data_num)

    ###### Task based metric ######
    best_net = None
    epoch_save = 0
    best_metric = 0
    lr_change = 0
    
    # Initialize training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_psnr': [],
        'val_mse': [],
        'val_ssim': [],
        'lr': []
    }

    ###### Start Training ######
    for e in range(opt.epoch):
        tmp_epoch = e+opt.start_epoch
        tmp_lr = optimizer.__getstate__()['param_groups'][0]['lr']
        print('================= Epoch %s lr=%.6f =================' % (tmp_epoch, tmp_lr))

        if tmp_epoch > epoch_save + opt.gap_epoch or lr_change == 4:
            break

        # Train
        train_loss = 0
        net = net.train()

        for i, return_list in tqdm(enumerate(train_batch)):
            case_name, x, y = return_list
            x = x.float().to(device, non_blocking=True)
            label = y.float().to(device, non_blocking=True)

            y_pre = net(x)
            loss = train_criterion(y_pre, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del y_pre, label, x

        non_model.clear_device_cache(device)
        train_loss = train_loss / len(train_batch)
        
        # Run validation every epoch (removed gap_val skip logic)
        net = net.eval()
        with torch.no_grad():
            psnr_list = []
            mse_list = []
            ssim_list = []

            for i, return_list in tqdm(enumerate(val_batch)):
                case_name, x, y, pos_list = return_list
                case_name = case_name[0]
                x = x.squeeze().data.numpy()
                y = y.squeeze().data.numpy()

                if e == 0 and i == 0:
                    print('thin size:', y.shape)

                y_pre = np.zeros_like(y)
                pos_list = pos_list.data.numpy()[0]
                
                # 获取该case的ratio（用于动态计算位置）
                case_ratio = in_model.get_case_ratio(case_name, opt.path_img, 'val')
                offset = case_ratio // 2  # 自动计算偏移量

                for pos_idx, pos in enumerate(pos_list):
                    tmp_x = x[pos_idx]
                    tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                    tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                    tmp_y_pre = net(tmp_x)
                    tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                    y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                    D = y_for_psnr.shape[0]
                    # 使用动态ratio计算位置（关键修改！）
                    pos_z_s = calculate_val_position(tmp_pos_z, case_ratio, offset)
                    pos_y_s = tmp_pos_y
                    pos_x_s = tmp_pos_x

                    y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

                del tmp_y_pre, tmp_x

                # Calculate metrics on the full volume (excluding borders)
                # 根据ratio动态调整边界
                border = max(case_ratio, 5)  # 至少去掉5层边界
                y_pre_eval = y_pre[border:-border]
                y_eval = y[border:-border]
                
                psnr = non_model.cal_psnr(y_pre_eval, y_eval)
                mse = non_model.cal_mse(y_pre_eval, y_eval)
                
                # Calculate SSIM per slice and average
                pid_ssim_list = []
                for z_idx, z_layer in enumerate(y_pre_eval):
                    mask_layer = y_eval[z_idx]
                    tmp_ssim = non_model.cal_ssim(mask_layer, z_layer, device=device)
                    pid_ssim_list.append(tmp_ssim)
                ssim_val = np.mean(pid_ssim_list)
                
                psnr_list.append(psnr)
                mse_list.append(mse)
                ssim_list.append(ssim_val)

        non_model.clear_device_cache(device)

        psnr_val = np.array(psnr_list).mean()
        mse_val = np.array(mse_list).mean()
        ssim_val = np.array(ssim_list).mean()
        
        # Update history
        history['epoch'].append(tmp_epoch)
        history['train_loss'].append(train_loss)
        history['val_psnr'].append(psnr_val)
        history['val_mse'].append(mse_val)
        history['val_ssim'].append(ssim_val)
        history['lr'].append(tmp_lr)
        
        # Print comprehensive metrics
        print('=' * 70)
        print('Epoch %s Summary:' % tmp_epoch)
        print('  Train Loss: %.6f' % train_loss)
        print('  Val PSNR:   %.4f dB' % psnr_val)
        print('  Val MSE:    %.6f' % mse_val)
        print('  Val SSIM:   %.4f' % ssim_val)
        print('  LR:         %.6f' % tmp_lr)
        print('=' * 70)

        # Save best model (always keep the best)
        if psnr_val > best_metric:
            best_metric = psnr_val
            epoch_save = tmp_epoch
            save_dict = {}
            save_dict['net'] = net
            save_dict['config_dict'] = config_dict
            save_dict['history'] = history
            save_dict['epoch'] = tmp_epoch
            save_dict['psnr'] = psnr_val
            save_dict['mse'] = mse_val
            save_dict['ssim'] = ssim_val
            
            # Save as best model
            best_model_path = save_model_folder + 'best_model.pkl'
            torch.save(save_dict, best_model_path)
            
            print('>>> Best Model Saved! (PSNR: %.4f)' % psnr_val)
            del save_dict
        
        # Save checkpoint every 10 epochs (if it's the best so far)
        if tmp_epoch % 10 == 0 and psnr_val > best_metric:
            save_dict = {}
            save_dict['net'] = net
            save_dict['config_dict'] = config_dict
            save_dict['history'] = history
            save_dict['epoch'] = tmp_epoch
            save_dict['psnr'] = psnr_val
            save_dict['mse'] = mse_val
            save_dict['ssim'] = ssim_val
            
            checkpoint_path = save_model_folder + 'checkpoint_epoch_%03d.pkl' % tmp_epoch
            torch.save(save_dict, checkpoint_path)
            print('>>> Checkpoint Saved (every 10 epochs): epoch %d' % tmp_epoch)
            del save_dict

        # Save history to CSV
        history_csv_path = save_log_folder + 'training_history.csv'
        with open(history_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val PSNR', 'Val MSE', 'Val SSIM', 'LR'])
            for i in range(len(history['epoch'])):
                writer.writerow([
                    history['epoch'][i],
                    history['train_loss'][i],
                    history['val_psnr'][i],
                    history['val_mse'][i],
                    history['val_ssim'][i],
                    history['lr'][i]
                ])

        if opt.cos_lr == True:
            scheduler.step()
        elif opt.Tmin == True:
            scheduler.step(train_loss)
        else:
            scheduler.step(best_metric)

        before_lr = optimizer.__getstate__()['param_groups'][0]['lr']
        if before_lr != tmp_lr:
            lr_change += 1

        non_model.clear_device_cache(device)
    
    # Training completed - save final history
    print('\n' + '=' * 70)
    print('Training Completed!')
    print('Best PSNR: %.4f at epoch %d' % (best_metric, epoch_save))
    print('=' * 70)
    
    # Save final history as JSON
    history_json_path = save_log_folder + 'training_history.json'
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print('Training history saved to:', save_log_folder)

if __name__ == '__main__':
    import fire

    fire.Fire()
