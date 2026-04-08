# -*- coding: utf-8 -*-
import os
import random
import time

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
    resume_from = kwargs.pop('resume_from', None)
    freeze_mode = str(kwargs.pop('freeze_mode', 'none')).strip().lower()
    if val_ssim_batch_size <= 0:
        raise ValueError('val_ssim_batch_size must be > 0')
    if val_ssim_stride <= 0:
        raise ValueError('val_ssim_stride must be > 0')
    if archive_every_epoch < 0:
        raise ValueError('archive_every_epoch must be >= 0')

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
    checkpoint_dir = os.path.join(checkpoint_root, str(opt.path_key), str(opt.net_idx))
    os.makedirs(checkpoint_dir, exist_ok=True)
    archive_dir = os.path.join(checkpoint_dir, 'epoch_archives')
    os.makedirs(archive_dir, exist_ok=True)

    ###### network ######
    net = model_TransSR.TVSRN().to(device)

    # Resume from checkpoint if specified
    if resume_from and os.path.isfile(resume_from):
        print(f'Resuming from: {resume_from}')
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if 'net' in ckpt:
            loaded_net = ckpt['net']
            if hasattr(loaded_net, 'state_dict'):
                net.load_state_dict(loaded_net.state_dict())
            else:
                net.load_state_dict(loaded_net)
        else:
            net.load_state_dict(ckpt)
        del ckpt
        print('Checkpoint loaded successfully')

    # ========== Freeze layers for small-data finetune ==========
    # freeze_mode options:
    #   'none'            — all params trainable (default)
    #   'encoder'         — freeze Encoder only (58K params, 0.7%)
    #   'encoder_mask'    — freeze Encoder + x_patch_mask (6.35M, 73.2%)
    #   'lp_encoder_mask' — freeze LP + Encoder + x_patch_mask (6.88M, 79.2%)
    #   'max_freeze'      — freeze LP + Encoder + x_patch_mask + Decoder_I1 (8.48M, 97.7%)
    if freeze_mode != 'none':
        freeze_names = set()
        if freeze_mode in ('encoder', 'encoder_mask', 'lp_encoder_mask', 'max_freeze'):
            freeze_names.add('Encoder')
        if freeze_mode in ('encoder_mask', 'lp_encoder_mask', 'max_freeze'):
            freeze_names.add('x_patch_mask')
        if freeze_mode in ('lp_encoder_mask', 'max_freeze'):
            freeze_names.add('LP')
        if freeze_mode == 'max_freeze':
            freeze_names.add('Decoder_I')

        frozen_count = 0
        for name, param in net.named_parameters():
            # Use startswith to match e.g. 'Decoder_I' → 'Decoder_I1.layers...'
            should_freeze = any(name == fn or name.startswith(fn + '.') or name.startswith(fn)
                                for fn in freeze_names)
            if should_freeze:
                param.requires_grad = False
                frozen_count += param.numel()

        total_count = sum(p.numel() for p in net.parameters())
        trainable_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'[freeze_mode={freeze_mode}] Frozen: {frozen_count:,} params ({100*frozen_count/total_count:.1f}%)')
        print(f'[freeze_mode={freeze_mode}] Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)')
        # Log per-module status
        for name, param in net.named_parameters():
            if not name.startswith(('Encoder.', 'LP.', 'Decoder_T', 'Decoder_I', 'conv_')):
                print(f'  {name}: {"FROZEN" if not param.requires_grad else "trainable"} ({param.numel():,})')

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

    train_list, train_only_1mm, train_only_5mm = non_model.list_paired_cases(opt.path_img, 'train')
    val_list, val_only_1mm, val_only_5mm = non_model.list_paired_cases(opt.path_img, 'val')
    if len(train_only_1mm) > 0 or len(train_only_5mm) > 0:
        print(f'Warning: train set has unmatched cases, skipped. 1mm_only={len(train_only_1mm)}, 5mm_only={len(train_only_5mm)}')
    if len(val_only_1mm) > 0 or len(val_only_5mm) > 0:
        print(f'Warning: val set has unmatched cases, skipped. 1mm_only={len(val_only_1mm)}, 5mm_only={len(val_only_5mm)}')
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
            label = y.float().to(device, non_blocking=True)

            y_pre = net(x, ratio=opt.ratio)
            # ========== 统一 Loss 计算域 ==========
            # 确保输出和GT在同一物理分辨率
            if y_pre.shape != label.shape:
                # 在 z 维度上对齐
                min_z = min(y_pre.shape[2], label.shape[2])
                diff_pred = y_pre.shape[2] - min_z
                diff_label = label.shape[2] - min_z
                y_pre = y_pre[:, :, diff_pred//2:diff_pred//2+min_z]
                label = label[:, :, diff_label//2:diff_label//2+min_z]
            loss = train_criterion(y_pre, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del y_pre, label, x

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
                        print('thin size:', y.shape)

                    y_pre = np.zeros_like(y)
                    pos_list = pos_list.data.numpy()[0]

                    for pos_idx, pos in enumerate(pos_list):
                        tmp_x = x[pos_idx]
                        tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                        tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                        tmp_y_pre = net(tmp_x, ratio=opt.ratio)
                        tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                        y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                        D = y_for_psnr.shape[0]
                        crop_margin = getattr(opt, 'crop_margin', 3)
                        pos_z_s = opt.ratio * tmp_pos_z + crop_margin
                        pos_y_s = tmp_pos_y
                        pos_x_s = tmp_pos_x

                        y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

                    y_pre_valid = y_pre[opt.ratio:-opt.ratio]
                    y_valid = y[opt.ratio:-opt.ratio]
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
    
