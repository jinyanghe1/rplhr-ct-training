"""
宣武数据集专用数据加载模块
Xuanwu Dataset Specific Data Loading Module

适配宣武数据集的组织结构：
- thin/ 代替 1mm/
- thick/ 代替 5mm/
- 实际比例: 4:1 (thick:thin)
- 使用 ratio=5 以保持与模型架构兼容，通过插值适配尺寸

同时集成保守数据增强方案（仅训练集）
"""

import numpy as np
import SimpleITK as sitk
import random
import os
from copy import deepcopy
from config import opt
from builtins import range
from scipy.ndimage import zoom

# 导入数据增强模块
from .augmentation import CTVolumetricAugmentation, normalize_ct, clip_ct_values
from .augmentation_config import CONSERVATIVE_AUG, GEOMETRY_ONLY_AUG


def _interpolate_to_shape(img: np.ndarray, target_shape: tuple, order: int = 3) -> np.ndarray:
    """
    使用插值将图像调整到目标形状
    
    Args:
        img: 输入图像 (Z, H, W)
        target_shape: 目标形状 (Z', H', W')
        order: 插值阶数，3=三次插值，1=线性插值
        
    Returns:
        插值后的图像
    """
    if img.shape == target_shape:
        return img
    
    # 计算缩放因子
    zoom_factors = [t / s for t, s in zip(target_shape, img.shape)]
    
    # 使用scipy.ndimage.zoom进行插值
    return zoom(img, zoom_factors, order=order)


def _get_hr_target_shape(lr_z: int, ratio: int = 5) -> int:
    """
    根据LR深度和ratio计算HR目标深度
    
    模型内部: out_z = (c_z - 1) * ratio + 1
    最终输出: out_z - 6 = (c_z - 1) * ratio - 5
    
    Args:
        lr_z: LR深度
        ratio: 比例因子
        
    Returns:
        HR目标深度
    """
    return (lr_z - 1) * ratio - 5


################################## for data ##################################
def get_train_img(img_path, case_name):
    """
    加载宣武数据集训练数据并应用保守增强
    
    路径结构:
        HR (thin):  train/thin/{case_name}.nii.gz
        LR (thick): train/thick/{case_name}.nii.gz
    
    处理流程:
        1. 加载 thick (LR) 和 thin (HR)
        2. 随机裁剪patch
        3. 对HR进行插值，将比例从4:1转换为5:1
        4. 应用保守数据增强（仅训练集）
        5. 可选：CT值裁剪
    """
    # 宣武数据集使用 thin 和 thick 命名
    case_mask_path = os.path.join(img_path, 'train', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'train', 'thick', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    # 使用 ratio=5 以保持与模型架构兼容
    ratio = 5
    actual_ratio = 4  # 宣武数据集实际比例

    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - opt.c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + opt.c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    # 裁剪LR (thick)
    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    # 裁剪HR (thin) - 使用实际比例4
    thin_z_s = z_s * actual_ratio
    thin_z_e = z_e * actual_ratio
    crop_mask = tmp_mask[thin_z_s:thin_z_e, y_s:y_e, x_s:x_e]

    # 对HR进行插值，从实际比例4转换为模型期望的比例5
    # 实际HR深度: c_z * 4 = 16
    # 期望HR深度: (c_z - 1) * 5 - 5 = 10
    hr_target_z = _get_hr_target_shape(opt.c_z, ratio)
    if crop_mask.shape[0] != hr_target_z:
        target_shape = (hr_target_z, opt.c_y, opt.c_x)
        crop_mask = _interpolate_to_shape(crop_mask, target_shape, order=3)

    # 保留原有的镜像增强（向后兼容）
    if opt.mirror and np.random.uniform() <= 0.3:
        crop_img = crop_img[:, :, ::-1].copy()
        crop_mask = crop_mask[:, :, ::-1].copy()
    
    # ==================== 修复 #1: 数据归一化 ====================
    # 在增强前进行归一化，确保数据在[0,1]范围内
    if hasattr(opt, 'normalize_ct') and opt.normalize_ct:
        crop_img, crop_mask = normalize_ct(
            crop_img, crop_mask,
            window_center=getattr(opt, 'window_center', 40),   # 软组织窗
            window_width=getattr(opt, 'window_width', 400)     # 软组织窗
        )
    
    # ==================== 修复 #2: 宣武数据集仅几何增强 ====================
    # 禁用所有强度变换，仅保留几何增强（避免破坏归一化后的数值）
    if hasattr(opt, 'use_augmentation') and opt.use_augmentation:
        aug = CTVolumetricAugmentation(
            prob=opt.aug_prob if hasattr(opt, 'aug_prob') else 0.5
        )
        
        # 使用仅几何增强配置（修复关键）
        aug_config = getattr(opt, 'aug_config', GEOMETRY_ONLY_AUG)
        
        # 应用训练增强
        crop_img, crop_mask = aug.apply_train_augmentation(
            crop_img, crop_mask, aug_config=aug_config
        )
    
    # ==================== 增强结束 ====================

    return crop_img, crop_mask


def get_val_img(img_path, case_name):
    """
    加载验证数据（宣武数据集格式）
    
    注意：验证集不使用数据增强！
    """
    case_mask_path = os.path.join(img_path, 'val', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))
    case_img_path = os.path.join(img_path, 'val', 'thick', case_name + '.nii.gz')
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

    # 注意：验证集返回原始mask，不在此处做插值
    # 插值在训练脚本中根据需要处理
    return crop_img, pos_list, tmp_mask


def get_test_img(img_path, case_name):
    """加载测试数据（宣武数据集格式）"""
    case_mask_path = os.path.join(img_path, 'test', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'test', 'thick', case_name + '.nii.gz')
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
