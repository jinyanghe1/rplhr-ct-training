#!/usr/bin/env python3
"""
支持 DICOM 格式的训练脚本

使用方法:
1. 将此文件放在 code/ 目录下
2. 修改 make_dataset.py 中的导入:
   - 将: from utils import in_model
   - 改为: from utils import in_model_dicom as in_model
3. 运行: python train_dicom.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN

主要修改:
- 支持从 DICOM 文件夹读取病例列表
- 支持混合模式 (DICOM 和 NIfTI 共存)
"""

# -*- coding: utf-8 -*-
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


def get_case_list(data_path, subset='train', resolution='1mm'):
    """
    获取病例列表 (支持 DICOM 文件夹和 NIfTI 文件)
    
    Args:
        data_path: 数据根目录
        subset: 'train', 'val', 或 'test'
        resolution: '1mm' 或 '5mm'
    
    Returns:
        list: 病例名称列表
    """
    target_dir = os.path.join(data_path, subset, resolution)
    
    if not os.path.exists(target_dir):
        print(f'警告: 目录不存在 {target_dir}')
        return []
    
    case_list = []
    
    for item in sorted(os.listdir(target_dir)):
        item_path = os.path.join(target_dir, item)
        
        # 情况 1: DICOM 文件夹
        if os.path.isdir(item_path):
            # 检查是否包含 DICOM 文件
            dicom_files = [f for f in os.listdir(item_path) 
                          if f.endswith('.dcm') or f.endswith('.DCM')]
            if len(dicom_files) > 0:
                case_list.append(item)  # 使用文件夹名作为病例名
        
        # 情况 2: NIfTI 文件
        elif item.endswith('.nii.gz'):
            case_name = item.replace('.nii.gz', '')
            case_list.append(case_name)
    
    return case_list


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

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(save_model_folder, exist_ok=True)

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

    # 使用新的函数获取病例列表 (支持 DICOM)
    train_list = get_case_list(opt.path_img, 'train', '1mm')
    val_list = get_case_list(opt.path_img, 'val', '1mm')

    print(f'训练病例列表 (前5个): {train_list[:5]}')
    print(f'验证病例列表 (前5个): {val_list[:5]}')

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
        
        pass_flag = False

        # gap val
        if opt.gap_val != 0 and e % opt.gap_val != 0:
            pass_flag = True
            
        if pass_flag:
            print('epoch %s, train_loss: %.4f' % (tmp_epoch, train_loss))
            continue

        net = net.eval()
        with torch.no_grad():
            psnr_list = []

            for i, return_list in tqdm(enumerate(val_batch)):
                case_name, x, y, pos_list = return_list
                case_name = case_name[0]
                x = x.squeeze().data.numpy()
                y = y.squeeze().data.numpy()

                if e == 0 and i == 0:
                    print('thin size:', y.shape)

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

                    y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

                del tmp_y_pre, tmp_x

                psnr = non_model.cal_psnr(y_pre[5:-5], y[5:-5])
                psnr_list.append(psnr)

        non_model.clear_device_cache(device)

        psnr_val = np.array(psnr_list).mean()
        print('epoch %s, train_loss: %.4f, psnr_val:, %.4f'%(tmp_epoch, train_loss, psnr_val))

        if psnr_val > best_metric:
            best_metric = psnr_val
            epoch_save = tmp_epoch
            save_dict = {}
            save_dict['net'] = net
            save_dict['config_dict'] = config_dict
            torch.save(save_dict, save_model_folder + \
                        '%s_train_loss_%.4f_val_psnr_%.4f.pkl' %
                           (str(tmp_epoch).rjust(3,'0'), train_loss, psnr_val))

            del save_dict
            print('====================== model save ========================')

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

if __name__ == '__main__':
    import fire

    fire.Fire()
