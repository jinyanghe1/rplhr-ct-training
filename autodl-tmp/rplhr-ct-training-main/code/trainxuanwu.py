# -*- coding: utf-8 -*-
"""
宣武数据集训练脚本 (Xuanwu Dataset Training Script)

特性:
- 适配宣武数据集格式 (thin/thick 替代 1mm/5mm)
- 使用保守数据增强方案
- 训练50个epoch
- 输出到指定目录并标明宣武数据集

Author: Auto-generated
Dataset: 宣武数据集 (Xuanwu Dataset)
Date: 2026-03-26
"""

import os
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset_xuanwu import train_Dataset, val_Dataset
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

# 导入保守增强配置
from utils.augmentation_config import CONSERVATIVE_AUG


def train(**kwargs):
    # ==================== 宣武数据集标识 ====================
    DATASET_NAME = "XUANWU"
    DATASET_DESC = "宣武数据集 (Xuanwu Dataset)"
    print("=" * 70)
    print(f"🎯 当前使用数据集: {DATASET_DESC}")
    print(f"🎯 Dataset Name: {DATASET_NAME}")
    print("=" * 70)
    # =====================================================

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

    # ==================== 宣武数据集路径适配 ====================
    # 使用 thin/thick 替代 1mm/5mm
    train_dir = os.path.join(opt.path_img, 'train', 'thin')
    val_dir = os.path.join(opt.path_img, 'val', 'thin')
    
    if not os.path.exists(train_dir):
        print(f"❌ 错误: 训练目录不存在: {train_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"❌ 错误: 验证目录不存在: {val_dir}")
        return
    
    train_list = [each.split('.')[0] for each in sorted(os.listdir(train_dir))]
    val_list = [each.split('.')[0] for each in sorted(os.listdir(val_dir))]
    # ==========================================================

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
    
    # ==================== 数据增强配置信息 ====================
    if hasattr(opt, 'use_augmentation') and opt.use_augmentation:
        print("=" * 70)
        print("🔄 数据增强已启用 (Geometry Only)")
        print(f"   基础概率: {getattr(opt, 'aug_prob', 0.5)}")
        print(f"   数据归一化: {getattr(opt, 'normalize_ct', False)}")
        print(f"   窗宽窗位: {getattr(opt, 'window_center', 40)}/{getattr(opt, 'window_width', 400)}")
        print("=" * 70)
    else:
        print("⚠️  数据增强未启用")
    # ========================================================

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
    print(f"\n🚀 开始训练 - 总共 {opt.epoch} 个 epoch\n")
    
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

                # 首先对 y (原始HR) 进行插值，匹配模型输出的尺寸
                # 宣武数据集实际比例4:1 -> 模型期望比例5:1
                # 计算插值后的目标尺寸
                actual_ratio = 4  # 宣武数据集实际比例
                model_ratio = 5   # 模型期望比例
                z_zoom_factor = model_ratio / actual_ratio
                target_z = int(y.shape[0] * z_zoom_factor)
                
                # 根据插值后的尺寸初始化 y_pre
                y_pre = np.zeros((target_z, y.shape[1], y.shape[2]), dtype=y.dtype)
                
                pos_list = pos_list.data.numpy()[0]

                for pos_idx, pos in enumerate(pos_list):
                    tmp_x = x[pos_idx]
                    tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                    tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                    tmp_y_pre = net(tmp_x)
                    tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                    y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                    D = y_for_psnr.shape[0]
                    ratio = getattr(opt, 'ratio', 5)  # 使用模型期望比例5
                    pos_z_s = ratio * tmp_pos_z + 3  # 与原始代码一致
                    pos_y_s = tmp_pos_y
                    pos_x_s = tmp_pos_x

                    y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

                del tmp_y_pre, tmp_x

                # 对 y (原始HR) 进行插值，匹配 y_pre (模型输出) 的尺寸
                if y_pre.shape[0] != y.shape[0]:
                    from scipy.ndimage import zoom
                    zoom_factor_z = y_pre.shape[0] / y.shape[0]
                    zoom_factors = [zoom_factor_z, 1.0, 1.0]
                    y = zoom(y, zoom_factors, order=3)
                
                # 修复 #2: 对HR进行归一化，与训练时保持一致
                if getattr(opt, 'normalize_ct', False):
                    # 使用与训练相同的窗宽窗位进行归一化
                    window_center = getattr(opt, 'window_center', 40)
                    window_width = getattr(opt, 'window_width', 400)
                    min_val = window_center - window_width / 2
                    max_val = window_center + window_width / 2
                    y = np.clip((y - min_val) / (max_val - min_val + 1e-8), 0, 1)
                
                # Calculate metrics on the full volume (excluding borders)
                y_pre_eval = y_pre[5:-5]
                y_eval = y[5:-5]
                
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
            save_dict['dataset'] = DATASET_NAME  # 标记数据集
            
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
            save_dict['dataset'] = DATASET_NAME  # 标记数据集
            
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
    print(f'Dataset: {DATASET_DESC}')
    print('Best PSNR: %.4f at epoch %d' % (best_metric, epoch_save))
    print('=' * 70)
    
    # Save final history as JSON
    history_json_path = save_log_folder + 'training_history.json'
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save dataset info
    dataset_info = {
        'dataset_name': DATASET_NAME,
        'dataset_desc': DATASET_DESC,
        'train_samples': train_data_num,
        'val_samples': val_data_num,
        'best_psnr': best_metric,
        'best_epoch': epoch_save,
        'augmentation': getattr(opt, 'use_augmentation', False),
        'aug_config': 'CONSERVATIVE' if getattr(opt, 'use_augmentation', False) else 'NONE'
    }
    dataset_info_path = save_log_folder + 'dataset_info.json'
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print('Training history saved to:', save_log_folder)

if __name__ == '__main__':
    import fire

    fire.Fire()
