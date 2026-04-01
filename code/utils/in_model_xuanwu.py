import numpy as np
import SimpleITK as sitk
import random
import os
from copy import deepcopy
from config import opt
from builtins import range

# 导入数据增强模块
from .augmentation import CTVolumetricAugmentation, normalize_ct, clip_ct_values

################################## for xuanwu data ##################################
def get_train_img(img_path, case_name):
    """
    加载宣武数据集训练数据（无插值版本）

    宣武数据集使用 thick (5mm) 和 thin (1.25mm) 目录结构

    流程:
        1. 加载LR (thick/5mm) 和 HR (thin/1.25mm) NIfTI文件
        2. 随机裁剪patch
        3. 应用数据增强 (空间变换、强度变换、噪声)
        4. 可选：CT值归一化/裁剪

    Args:
        img_path: 数据根目录
        case_name: 病例名称

    Returns:
        crop_img: LR patch (thick, 6层)
        crop_mask: HR patch (thin, 24层) - 注意：不进行插值！
    """
    # 宣武数据集使用 thick/thin 而非 1mm/5mm
    case_mask_path = os.path.join(img_path, 'train', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'train', 'thick', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - opt.c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + opt.c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    # 裁剪LR (thick) - 6层
    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    # 裁剪HR (thin) - 直接使用真实比例，不插值
    # c_z=6, ratio=4 → thin有c_z*ratio=24层
    thin_z_s = z_s * opt.ratio
    thin_z_e = thin_z_s + opt.c_z * opt.ratio  # 如 c_z=6 → 24层
    crop_mask = tmp_mask[thin_z_s:thin_z_e, y_s:y_e, x_s:x_e]

    # 保留原有的镜像增强（向后兼容）
    if opt.mirror and np.random.uniform() <= 0.3:
        crop_img = crop_img[:, :, ::-1].copy()
        crop_mask = crop_mask[:, :, ::-1].copy()

    # ==================== 数据增强 ====================
    if hasattr(opt, 'use_augmentation') and opt.use_augmentation:
        aug = CTVolumetricAugmentation(
            prob=opt.aug_prob if hasattr(opt, 'aug_prob') else 0.5
        )

        aug_config = getattr(opt, 'aug_config', None)

        crop_img, crop_mask = aug.apply_train_augmentation(
            crop_img, crop_mask, aug_config=aug_config
        )

        # CT值裁剪
        if hasattr(opt, 'clip_ct') and opt.clip_ct:
            crop_img, crop_mask = clip_ct_values(
                crop_img, crop_mask,
                min_hu=getattr(opt, 'min_hu', -1024),
                max_hu=getattr(opt, 'max_hu', 3071)
            )

        # 归一化到[0,1]
        if hasattr(opt, 'normalize_ct') and opt.normalize_ct:
            crop_img, crop_mask = normalize_ct(
                crop_img, crop_mask,
                window_center=getattr(opt, 'window_center', 0),
                window_width=getattr(opt, 'window_width', 1000)
            )

    return crop_img, crop_mask

def get_val_img(img_path, case_name):
    """
    加载宣武数据集验证数据（归一化版本）
    """
    case_mask_path = os.path.join(img_path, 'val', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'val', 'thick', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    # ========== 归一化处理（与训练保持一致）==========
    # CT值裁剪
    if hasattr(opt, 'clip_ct') and opt.clip_ct:
        tmp_img = np.clip(tmp_img, getattr(opt, 'min_hu', -1024), getattr(opt, 'max_hu', 3071))
        tmp_mask = np.clip(tmp_mask, getattr(opt, 'min_hu', -1024), getattr(opt, 'max_hu', 3071))

    # 归一化到[0,1]
    if hasattr(opt, 'normalize_ct') and opt.normalize_ct:
        window_center = getattr(opt, 'window_center', 0)
        window_width = getattr(opt, 'window_width', 1000)
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        tmp_img = np.clip((tmp_img - min_val) / (max_val - min_val + 1e-8), 0, 1)
        tmp_mask = np.clip((tmp_mask - min_val) / (max_val - min_val + 1e-8), 0, 1)
    # ================================================

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
    """
    加载宣武数据集测试数据（归一化版本）
    """
    case_mask_path = os.path.join(img_path, 'test', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'test', 'thick', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    # ========== 归一化处理（与训练保持一致）==========
    # CT值裁剪
    if hasattr(opt, 'clip_ct') and opt.clip_ct:
        tmp_img = np.clip(tmp_img, getattr(opt, 'min_hu', -1024), getattr(opt, 'max_hu', 3071))
        tmp_mask = np.clip(tmp_mask, getattr(opt, 'min_hu', -1024), getattr(opt, 'max_hu', 3071))

    # 归一化到[0,1]
    if hasattr(opt, 'normalize_ct') and opt.normalize_ct:
        window_center = getattr(opt, 'window_center', 0)
        window_width = getattr(opt, 'window_width', 1000)
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        tmp_img = np.clip((tmp_img - min_val) / (max_val - min_val + 1e-8), 0, 1)
        tmp_mask = np.clip((tmp_mask - min_val) / (max_val - min_val + 1e-8), 0, 1)
    # ================================================

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