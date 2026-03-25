import numpy as np
import SimpleITK as sitk
import random
import os
from copy import deepcopy
from config import opt
from builtins import range

# 导入数据增强模块
from .augmentation import CTVolumetricAugmentation, normalize_ct, clip_ct_values

################################## for data ##################################
def get_train_img(img_path, case_name):
    """
    加载训练数据并应用增强
    
    流程:
        1. 加载LR (5mm) 和 HR (1mm) NIfTI文件
        2. 随机裁剪patch
        3. 应用数据增强 (空间变换、强度变换、噪声)
        4. 可选：CT值归一化/裁剪
    
    Args:
        img_path: 数据根目录
        case_name: 病例名称
        
    Returns:
        crop_img: 增强后的LR patch
        crop_mask: 对应的HR patch
    """
    case_mask_path = os.path.join(img_path, 'train', '1mm', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'train', '5mm', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - opt.c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + opt.c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    mask_z_s = z_s * 5 + 3
    mask_z_e = (z_e - 1) * 5 - 2

    crop_mask = tmp_mask[mask_z_s: mask_z_e, y_s:y_e, x_s:x_e]

    # 保留原有的镜像增强（向后兼容）
    if opt.mirror and np.random.uniform() <= 0.3:
        crop_img = crop_img[:, :, ::-1].copy()
        crop_mask = crop_mask[:, :, ::-1].copy()
    
    # ==================== 新增：数据增强 ====================
    # 检查是否启用新的增强pipeline
    if hasattr(opt, 'use_augmentation') and opt.use_augmentation:
        # 初始化增强器
        aug = CTVolumetricAugmentation(
            prob=opt.aug_prob if hasattr(opt, 'aug_prob') else 0.5
        )
        
        # 获取增强配置（如果提供）
        aug_config = getattr(opt, 'aug_config', None)
        
        # 应用训练增强
        crop_img, crop_mask = aug.apply_train_augmentation(
            crop_img, crop_mask, aug_config=aug_config
        )
        
        # 可选：CT值裁剪（推荐）
        if hasattr(opt, 'clip_ct') and opt.clip_ct:
            crop_img, crop_mask = clip_ct_values(
                crop_img, crop_mask,
                min_hu=getattr(opt, 'min_hu', -1024),
                max_hu=getattr(opt, 'max_hu', 3071)
            )
        
        # 可选：归一化到[0,1]
        if hasattr(opt, 'normalize_ct') and opt.normalize_ct:
            crop_img, crop_mask = normalize_ct(
                crop_img, crop_mask,
                window_center=getattr(opt, 'window_center', 0),
                window_width=getattr(opt, 'window_width', 1000)
            )
    
    # ==================== 增强结束 ====================

    return crop_img, crop_mask

def get_val_img(img_path, case_name):
    case_mask_path = os.path.join(img_path, 'val', '1mm', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))
    case_img_path = os.path.join(img_path, 'val', '5mm', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    if opt.mode != 'test':
        tmp_img = tmp_img[:, 128:-128, 128:-128]
        tmp_mask = tmp_mask[:, 128:-128, 128:-128]

    z, y, x = tmp_img.shape
    z_s = 0
    z_split = []
    while z_s + opt.vc_z < z:
        z_split.append(z_s)
        z_s += (opt.vc_z - 2)

    if z - opt.vc_z > z_split[-1]:
        z_split.append(z - opt.vc_z)

    y_split = np.arange(y // opt.vc_y) * opt.vc_y
    x_split = np.arange(x // opt.vc_x) * opt.vc_x

    crop_img = []
    pos_list = []

    for z_s in z_split:
        tmp_crop_img = deepcopy(tmp_img)[z_s:z_s + opt.vc_z]
        tmp_crop_img = np.array(np.array_split(tmp_crop_img, y // opt.vc_y, axis=1))
        tmp_crop_img = np.array(np.array_split(tmp_crop_img, x // opt.vc_x, axis=3))
        tmp_crop_img = tmp_crop_img.transpose((1, 0, 2, 3, 4))
        H_num, W_num, D, H, W = tmp_crop_img.shape
        tmp_crop_img = tmp_crop_img.reshape(H_num * W_num, D, H, W)
        crop_img.append(tmp_crop_img)

    crop_img = np.array(crop_img)
    patch_num, HW_num, D, H, W = crop_img.shape
    crop_img = crop_img.reshape(patch_num * HW_num, D, H, W)

    for z_s in z_split:
        for y_s in y_split:
            for x_s in x_split:
                pos_list.append(np.array([z_s, y_s, x_s]))

    pos_list = np.array(pos_list)

    return crop_img, pos_list, tmp_mask

def get_test_img(img_path, case_name):
    case_mask_path = os.path.join(img_path, 'test', '1mm', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'test', '5mm', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    z = tmp_img.shape[0]
    z_s = 0
    z_split = []
    while z_s + opt.vc_z < z:
        z_split.append(z_s)
        z_s += (opt.vc_z - 2)

    if z - opt.vc_z > z_split[-1]:
        z_split.append(z - opt.vc_z)

    y_split = np.arange(512 // opt.vc_y) * opt.vc_y
    x_split = np.arange(512 // opt.vc_x) * opt.vc_x

    crop_img = []
    pos_list = []

    for z_s in z_split:
        tmp_crop_img = deepcopy(tmp_img)[z_s:z_s + opt.vc_z]
        tmp_crop_img = np.array(np.array_split(tmp_crop_img, 512 // opt.vc_y, axis=1))
        tmp_crop_img = np.array(np.array_split(tmp_crop_img, 512 // opt.vc_x, axis=3))
        tmp_crop_img = tmp_crop_img.transpose((1, 0, 2, 3, 4))
        H_num, W_num, D, H, W = tmp_crop_img.shape
        tmp_crop_img = tmp_crop_img.reshape(H_num * W_num, D, H, W)
        crop_img.append(tmp_crop_img)

    crop_img = np.array(crop_img)
    patch_num, HW_num, D, H, W = crop_img.shape
    crop_img = crop_img.reshape(patch_num * HW_num, D, H, W)

    for z_s in z_split:
        for y_s in y_split:
            for x_s in x_split:
                pos_list.append(np.array([z_s, y_s, x_s]))

    pos_list = np.array(pos_list)

    return crop_img, pos_list, tmp_mask
