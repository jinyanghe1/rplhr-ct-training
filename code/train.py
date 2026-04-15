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
from scipy.signal import windows as scipy_windows
from tqdm import tqdm

import cv2
import warnings
warnings.filterwarnings("ignore")


def _z_blend_weight(D, blend_type='gaussian'):
    """Build 1D z-blending weight for overlap stitching."""
    if blend_type == 'gaussian':
        w = scipy_windows.gaussian(D, std=D / 4.0)
    elif blend_type == 'triang':
        w = scipy_windows.triang(D)
    else:
        w = np.ones(D)
    w = np.maximum(w, 1e-6).astype(np.float32)
    return w[:, np.newaxis, np.newaxis]

# Phase B imports
from loss_eagle3d import EAGLELoss3D, CharbonnierLoss, L1SSIMLoss3D
from training.ema import EMA

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

    # Phase B: Early stopping
    early_stop_patience = int(kwargs.pop('early_stop_patience', 20))

    # Phase B: Warmup + Gradient Clipping
    use_warmup = _to_bool(kwargs.pop('use_warmup', False))
    warmup_epochs = int(kwargs.pop('warmup_epochs', 5))
    use_grad_clip = _to_bool(kwargs.pop('use_grad_clip', False))
    grad_clip_norm = float(kwargs.pop('grad_clip_norm', 1.0))

    # Phase B: EMA
    use_ema = _to_bool(kwargs.pop('use_ema', False))
    ema_decay = float(kwargs.pop('ema_decay', 0.995))
    ema_warmup_epochs = int(kwargs.pop('ema_warmup_epochs', 10))

    # Phase B: Data augmentation (passed through to opt for in_model.py)
    # use_augmentation, aug_prob etc. are handled via opt config

    # Phase B: Discriminative LR (lower LR for pretrained layers)
    encoder_lr_scale = float(kwargs.pop('encoder_lr_scale', 1.0))

    # Phase B: MAE auxiliary task
    use_mae = _to_bool(kwargs.pop('use_mae', False))
    mae_prob = float(kwargs.pop('mae_prob', 0.3))
    mae_weight = float(kwargs.pop('mae_weight', 0.1))
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

    ###### EMA (Phase B) ######
    ema = None
    if use_ema:
        ema = EMA(net, decay=ema_decay, device=device)
        print(f'EMA enabled: decay={ema_decay}, warmup_epochs={ema_warmup_epochs}')

    ###### optim ######
    lr = opt.lr

    # Build parameter groups (discriminative LR for pretrained layers)
    if encoder_lr_scale < 1.0 and freeze_mode == 'none':
        pretrained_params = []
        new_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(('Encoder', 'x_patch_mask', 'LP')):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        param_groups = [
            {'params': pretrained_params, 'lr': lr * encoder_lr_scale},
            {'params': new_params, 'lr': lr},
        ]
        print(f'Discriminative LR: pretrained={lr*encoder_lr_scale:.6f} ({len(pretrained_params)} tensors), '
              f'new={lr:.6f} ({len(new_params)} tensors)')
    else:
        param_groups = filter(lambda p: p.requires_grad, net.parameters())

    if opt.optim == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=opt.wd, momentum=0.9)
        print('================== SGD lr = %.6f ==================' % lr)

    elif opt.optim == 'Adam':
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=opt.wd)
        print('================== Adam lr = %.6f ==================' % lr)

    elif opt.optim == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=opt.wd)
        print('================== AdamW lr = %.6f ==================' % lr)

    if opt.cos_lr:
        if use_warmup:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(opt.Tmax - warmup_epochs, 1), eta_min=opt.lr / opt.lr_gap)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
            print(f'Scheduler: Warmup({warmup_epochs}ep) + CosineAnnealing(T_max={opt.Tmax - warmup_epochs})')
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opt.Tmax, eta_min=opt.lr / opt.lr_gap)
    elif opt.Tmin == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=opt.patience, threshold=0.000001)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                               patience=opt.patience, threshold=0.000001)

    ###### loss ######
    print('Use %s loss'%opt.loss_f)
    if opt.loss_f == 'eagle3d':
        eagle_alpha = float(getattr(opt, 'eagle_alpha', 0.1))
        train_criterion = EAGLELoss3D(alpha=eagle_alpha).to(device)
        print(f'  EAGLELoss3D alpha={eagle_alpha}')
    elif opt.loss_f == 'charbonnier':
        train_criterion = CharbonnierLoss()
        print('  CharbonnierLoss eps=1e-6')
    elif opt.loss_f == 'l1_ssim':
        l1ssim_alpha = float(getattr(opt, 'l1ssim_alpha', 0.1))
        train_criterion = L1SSIMLoss3D(alpha=l1ssim_alpha).to(device)
        print(f'  L1SSIMLoss3D alpha={l1ssim_alpha}')
    else:
        train_criterion = nn.L1Loss()
        print('  nn.L1Loss (default)')

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
    epochs_since_improvement = 0  # Phase B: early stopping counter
    print(f'checkpoints will be saved to: {checkpoint_dir}')
    if early_stop_patience > 0:
        print(f'early stopping enabled: patience={early_stop_patience} epochs')
    if use_grad_clip:
        print(f'gradient clipping enabled: max_norm={grad_clip_norm}')
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

        # Legacy stop: gap_epoch overflow OR 4 LR reductions (ReduceLROnPlateau only)
        # With cosine LR, lr changes every epoch, so skip the lr_change check
        if tmp_epoch > epoch_save + opt.gap_epoch:
            break
        if not (opt.cos_lr == True or use_warmup) and lr_change == 4:
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

            # Phase B: MAE auxiliary task
            if use_mae and random.random() < mae_prob:
                masked_x = x.clone()
                mask_idx = random.randint(0, opt.c_z - 1)
                masked_x[:, :, mask_idx] = 0  # zero out one slice
                y_mae = net(masked_x, ratio=opt.ratio)
                if y_mae.shape != label.shape:
                    min_z_mae = min(y_mae.shape[2], label.shape[2])
                    dp = y_mae.shape[2] - min_z_mae
                    dl = label.shape[2] - min_z_mae
                    y_mae = y_mae[:, :, dp//2:dp//2+min_z_mae]
                mae_loss = F.l1_loss(y_mae, label[:, :, :y_mae.shape[2]])
                loss = loss + mae_weight * mae_loss
                del y_mae, masked_x

            # backward
            optimizer.zero_grad()
            loss.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, net.parameters()), grad_clip_norm)
            optimizer.step()

            # Phase B: EMA update
            if ema is not None:
                ema.update(net)

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

            # Phase B: Apply EMA shadow for validation (after warmup)
            ema_applied = False
            if ema is not None and e >= ema_warmup_epochs:
                ema.apply_shadow(net)
                ema_applied = True

            try:
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

                        y_pre_sum = np.zeros_like(y, dtype=np.float32)
                        weight_sum = np.zeros_like(y, dtype=np.float32)
                        pos_list = pos_list.data.numpy()[0]
                        blend_type = getattr(opt, 'val_blend', 'gaussian')

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

                            wz = _z_blend_weight(D, blend_type)
                            y_pre_sum[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] += y_for_psnr * wz
                            weight_sum[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] += wz

                        y_pre = y_pre_sum / np.maximum(weight_sum, 1e-8)
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
                ema_tag = ' [EMA]' if ema_applied else ''
                if compute_val_ssim:
                    ssim_val = float(np.array(ssim_list).mean())
                    print('epoch %s, train_loss: %.4f, psnr_val: %.4f, ssim_val: %.6f, val_sec: %.1f%s' %
                          (tmp_epoch, train_loss, psnr_val, ssim_val, val_elapsed, ema_tag))
                else:
                    print('epoch %s, train_loss: %.4f, psnr_val: %.4f, val_sec: %.1f%s' %
                          (tmp_epoch, train_loss, psnr_val, val_elapsed, ema_tag))

                if psnr_val > best_metric:
                    best_metric = psnr_val
                    epoch_save = tmp_epoch
                    epochs_since_improvement = 0
                    save_dict = {}
                    save_dict['net'] = net
                    save_dict['config_dict'] = config_dict
                    if ema is not None:
                        save_dict['ema_state'] = ema.state_dict()
                    save_path = save_model_folder + \
                                '%s_train_loss_%.4f_val_psnr_%.4f.pkl' % \
                                   (str(tmp_epoch).rjust(3,'0'), train_loss, psnr_val)
                    torch.save(save_dict, save_path)
                    # Also save as best.pkl for easy resume
                    best_path = os.path.join(save_model_folder, 'best.pkl')
                    torch.save(save_dict, best_path)
                    del save_dict
                    print('====================== model save (best.pkl) ========================')
                else:
                    epochs_since_improvement += opt.gap_val if opt.gap_val > 0 else 1

            finally:
                # Phase B: Restore original params after EMA validation
                if ema_applied:
                    ema.restore(net)

            non_model.clear_device_cache(device)

        # ========== Scheduler step (FIXED: every epoch, not just validation epochs) ==========
        if opt.cos_lr == True or use_warmup:
            scheduler.step()
        elif opt.Tmin == True:
            scheduler.step(train_loss)
        else:
            if psnr_val is not None:
                scheduler.step(best_metric)

        before_lr = optimizer.__getstate__()['param_groups'][0]['lr']
        if before_lr != tmp_lr:
            lr_change += 1

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

        # Phase B: Early stopping check
        if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
            print(f'========== EARLY STOPPING at epoch {tmp_epoch} ==========')
            print(f'No improvement in {epochs_since_improvement} epochs. Best PSNR={best_metric:.4f} at epoch {epoch_save}')
            break

if __name__ == '__main__':
    import fire

    fire.Fire()
    
