# -*- coding: utf-8 -*-
"""
宣武数据集训练脚本 (Ratio=4 架构适配版本)

关键修改：
1. 使用 xuanwu_ratio4.txt 配置 (c_z=6, ratio=4)
2. 使用 thick/thin 数据路径 (而非 1mm/5mm)
3. 训练时对齐 y (24层) 到 y_pre (16层)
4. 验证时使用正确的位置计算
"""
import os
import random
import time

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
import lpips
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

def train(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    compute_val_ssim = _to_bool(kwargs.pop('compute_val_ssim', True))
    val_ssim_batch_size = int(kwargs.pop('val_ssim_batch_size', 32))
    val_ssim_stride = int(kwargs.pop('val_ssim_stride', 1))
    checkpoint_root = kwargs.pop('checkpoint_root', '../checkpoints')
    archive_every_epoch = int(kwargs.pop('archive_every_epoch', 0))
    if val_ssim_batch_size <= 0:
        raise ValueError('val_ssim_batch_size must be > 0')
    if val_ssim_stride <= 0:
        raise ValueError('val_ssim_stride must be > 0')
    if archive_every_epoch < 0:
        raise ValueError('archive_every_epoch must be >= 0')

    # 加载宣武数据集专用配置
    opt.load_config('../config/xuanwu_ratio4.txt')
    config_dict = opt._spec(kwargs)

    ###### random setting ######
    GLOBAL_SEED = 2022
    non_model.seed_everything(GLOBAL_SEED)

    ###### Device ######
    device = non_model.resolve_device(opt.gpu_idx)
    print('Use device:', device)
    print('Config: ratio=%d, c_z=%d' % (opt.ratio, opt.c_z))

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(save_model_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, str(opt.path_key), str(opt.net_idx))
    os.makedirs(checkpoint_dir, exist_ok=True)
    archive_dir = os.path.join(checkpoint_dir, 'epoch_archives')
    os.makedirs(archive_dir, exist_ok=True)

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

    # Perceptual loss (LPIPS)
    use_perceptual_loss = getattr(opt, 'use_perceptual_loss', False)
    perceptual_alpha = getattr(opt, 'perceptual_alpha', 0.1)
    perceptual_loss_fn = None
    if use_perceptual_loss:
        print(f'Initializing LPIPS perceptual loss (alpha={perceptual_alpha})...')
        perceptual_loss_fn = lpips.LPIPS(net='alex').to(device)

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

    # 使用 thick/thin 作为高分辨率/低分辨率标识 (xuanwu数据集)
    train_list, train_only_1mm, train_only_5mm = non_model.list_paired_cases(
        opt.path_img, 'train', high_res='thin', low_res='thick'
    )
    val_list, val_only_1mm, val_only_5mm = non_model.list_paired_cases(
        opt.path_img, 'val', high_res='thin', low_res='thick'
    )
    if len(train_only_1mm) > 0 or len(train_only_5mm) > 0:
        print(f'Warning: train set has unmatched cases, skipped. thin_only={len(train_only_1mm)}, thick_only={len(train_only_5mm)}')
    if len(val_only_1mm) > 0 or len(val_only_5mm) > 0:
        print(f'Warning: val set has unmatched cases, skipped. thin_only={len(val_only_1mm)}, thick_only={len(val_only_5mm)}')
    if len(train_list) == 0 or len(val_list) == 0:
        raise RuntimeError(f'No paired cases found under path_img={opt.path_img}')

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
    metric_history = []
    print(f'checkpoints will be saved to: {checkpoint_dir}')
    if compute_val_ssim:
        print(f'val SSIM enabled, batch_size={val_ssim_batch_size}, stride={val_ssim_stride}')
    else:
        print('val SSIM disabled by compute_val_ssim=False')
    if archive_every_epoch > 0:
        print(f'periodic archive enabled: every {archive_every_epoch} epochs -> {archive_dir}')
    else:
        print('periodic archive disabled')

    # 计算位置映射参数
    # 模型输出层数: (c_z-1)*ratio+1 - 裁剪层数
    # 对于 c_z=6, ratio=4: out_z=21, 裁剪5层 -> 16层
    model_output_z = (opt.c_z - 1) * opt.ratio + 1
    if opt.ratio == 4 and opt.c_z == 6:
        crop_front = 2
        crop_back = 3
    else:
        crop_front = 3
        crop_back = 3
    actual_output_z = model_output_z - crop_front - crop_back
    print(f'Model internal out_z={model_output_z}, actual output={actual_output_z} after cropping')

    ###### Start Training ######
    for e in range(opt.epoch):
        epoch_start_time = time.time()
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
            y = y.float().to(device, non_blocking=True)

            y_pre = net(x)  # 输出是 actual_output_z 层 (如 16层)

            # ========== 对齐 y 到 y_pre 的尺寸 ==========
            # y 是 24层 (c_z*ratio=6*4), y_pre 是 16层
            # 取 y 的中间16层进行对齐
            if y.shape[2] != y_pre.shape[2]:
                offset = (y.shape[2] - y_pre.shape[2]) // 2
                y = y[:, :, offset:offset+y_pre.shape[2], :, :]

            loss = train_criterion(y_pre, y)

            # Perceptual loss
            if perceptual_loss_fn is not None:
                # LPIPS expects 3-channel input, expand from 1 to 3 channels
                y_pre_3ch = y_pre.repeat(1, 3, 1, 1, 1)
                y_3ch = y.repeat(1, 3, 1, 1, 1)
                loss_percep = perceptual_loss_fn(y_pre_3ch, y_3ch).mean()
                loss = loss + perceptual_alpha * loss_percep

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del y_pre, y, x

        non_model.clear_device_cache(device)
        train_loss = train_loss / len(train_batch)

        psnr_val = None
        ssim_val = None
        val_elapsed = None
        pass_flag = False

        # gap val
        if opt.gap_val != 0 and e % opt.gap_val != 0:
            pass_flag = True

        if pass_flag:
            print('epoch %s, train_loss: %.4f' % (tmp_epoch, train_loss))
        else:
            val_start_time = time.time()
            net = net.eval()
            with torch.no_grad():
                psnr_list = []
                ssim_list = [] if compute_val_ssim else None

                for i, return_list in tqdm(enumerate(val_batch)):
                    case_name, x, y, pos_list = return_list
                    case_name = case_name[0]
                    x = x.squeeze().data.numpy()
                    y = y.squeeze().data.numpy()

                    if e == 0 and i == 0:
                        print('thin size:', y.shape)  # 应该是 16*4=64 层

                    y_pre = np.zeros_like(y)
                    pos_list = pos_list.data.numpy()[0]

                    for pos_idx, pos in enumerate(pos_list):
                        tmp_x = x[pos_idx]
                        tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                        tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)

                        # TTA: Test Time Augmentation
                        if getattr(opt, 'use_tta', False):
                            # Original
                            tmp_y_pre = net(tmp_x)
                            # Horizontal flip
                            tmp_x_hflip = torch.flip(tmp_x, [3])
                            tmp_y_pre_hflip = net(tmp_x_hflip)
                            tmp_y_pre_hflip = torch.flip(tmp_y_pre_hflip, [3])
                            # Vertical flip
                            tmp_x_vflip = torch.flip(tmp_x, [4])
                            tmp_y_pre_vflip = net(tmp_x_vflip)
                            tmp_y_pre_vflip = torch.flip(tmp_y_pre_vflip, [4])
                            # Average
                            tmp_y_pre = (tmp_y_pre + tmp_y_pre_hflip + tmp_y_pre_vflip) / 3.0
                        else:
                            tmp_y_pre = net(tmp_x)

                        tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                        y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                        D = y_for_psnr.shape[0]  # 模型输出层数 (16)
                        # 位置映射: 输入thick位置 * ratio = 对应thin位置
                        # crop_front/crop_back是模型内部裁剪，不影响位置映射
                        pos_z_s = opt.ratio * tmp_pos_z
                        pos_y_s = tmp_pos_y
                        pos_x_s = tmp_pos_x

                        y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

                    # 对齐评估区域 (去掉边界，避免边界效应)
                    # 移除前后各 crop_front+crop_back 层
                    border = crop_front + crop_back
                    y_pre_valid = y_pre[border:-border]
                    y_valid = y[border:-border]

                    # 确保尺寸匹配
                    min_z = min(y_pre_valid.shape[0], y_valid.shape[0])
                    y_pre_valid = y_pre_valid[:min_z]
                    y_valid = y_valid[:min_z]

                    psnr = non_model.cal_psnr(y_pre_valid, y_valid)
                    psnr_list.append(psnr)

                    if compute_val_ssim:
                        case_ssim = non_model.cal_ssim_volume(
                            y_valid,
                            y_pre_valid,
                            device=device,
                            batch_size=val_ssim_batch_size,
                            stride=val_ssim_stride,
                        )
                        ssim_list.append(case_ssim)

            non_model.clear_device_cache(device)

            psnr_val = float(np.array(psnr_list).mean())
            val_elapsed = time.time() - val_start_time
            if compute_val_ssim:
                ssim_val = float(np.array(ssim_list).mean())
                print('epoch %s, train_loss: %.4f, psnr_val: %.4f, ssim_val: %.6f, val_sec: %.1f' %
                      (tmp_epoch, train_loss, psnr_val, ssim_val, val_elapsed))
            else:
                print('epoch %s, train_loss: %.4f, psnr_val: %.4f, val_sec: %.1f' %
                      (tmp_epoch, train_loss, psnr_val, val_elapsed))

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

        epoch_elapsed = time.time() - epoch_start_time
        metric_history.append({
            'epoch': int(tmp_epoch),
            'lr': float(tmp_lr),
            'train_loss': float(train_loss),
            'val_psnr': psnr_val,
            'val_ssim': ssim_val,
            'epoch_sec': float(epoch_elapsed),
            'val_sec': float(val_elapsed) if val_elapsed is not None else None,
        })
        non_model.save_metric_history(metric_history, checkpoint_dir)
        non_model.plot_metric_history(metric_history, checkpoint_dir)
        print('metrics updated:', os.path.join(checkpoint_dir, 'metrics.csv'))

        if archive_every_epoch > 0 and tmp_epoch % archive_every_epoch == 0:
            archive_path = os.path.join(archive_dir, f'epoch_{str(tmp_epoch).rjust(3, "0")}.pth')
            archive_dict = {
                'epoch': tmp_epoch,
                'net': net,
                'config_dict': config_dict,
                'metric_history': metric_history,
            }
            torch.save(archive_dict, archive_path)
            print('periodic archive saved:', archive_path)

        if pass_flag:
            continue

if __name__ == '__main__':
    import fire

    fire.Fire()