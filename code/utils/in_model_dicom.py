#!/usr/bin/env python3
"""
支持 DICOM 格式的数据加载模块

使用方法:
1. 将此文件保存为 utils/in_model_dicom.py
2. 在 make_dataset.py 中修改导入:
   - 将: from utils import in_model
   - 改为: from utils import in_model_dicom as in_model

DICOM 目录结构:
    data/
    ├── train/
    │   ├── 5mm/
    │   │   ├── CT00000000/          # DICOM 文件夹
    │   │   │   ├── 0001.dcm
    │   │   │   ├── 0002.dcm
    │   │   │   └── ...
    │   │   └── CT00000001/
    │   └── 1mm/
    │       ├── CT00000000/
    │       └── CT00000001/
    └── val/
        └── ...
"""

import os
import numpy as np
import SimpleITK as sitk
import random
from copy import deepcopy
from config import opt
from builtins import range


def read_dicom_series(dicom_folder):
    """
    读取 DICOM 序列文件夹
    
    Args:
        dicom_folder: DICOM 文件夹路径
        
    Returns:
        numpy array: 3D 图像数据
    """
    reader = sitk.ImageSeriesReader()
    
    # 获取 DICOM 文件列表
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    
    if len(dicom_files) == 0:
        raise ValueError(f"在 {dicom_folder} 中未找到 DICOM 文件")
    
    # 读取图像
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    
    # 转换为 numpy array
    array = sitk.GetArrayFromImage(image)
    
    return array


def get_train_img(img_path, case_name):
    """训练数据加载 - DICOM 版本"""
    # 1mm (HR) - DICOM 文件夹
    case_mask_path = os.path.join(img_path, 'train/1mm/', case_name)
    if os.path.isdir(case_mask_path):
        # DICOM 文件夹
        tmp_mask = read_dicom_series(case_mask_path)
    else:
        # 尝试 NIfTI
        case_mask_path = img_path + 'train/1mm/' + case_name + '.nii.gz'
        tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    # 5mm (LR) - DICOM 文件夹
    case_img_path = os.path.join(img_path, 'train/5mm/', case_name)
    if os.path.isdir(case_img_path):
        tmp_img = read_dicom_series(case_img_path)
    else:
        case_img_path = img_path + 'train/5mm/' + case_name + '.nii.gz'
        tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - opt.c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + opt.c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    mask_z_s = z_s * opt.ratio + 3
    mask_z_e = (z_e - 1) * opt.ratio - 2

    crop_mask = tmp_mask[mask_z_s:mask_z_e, y_s:y_e, x_s:x_e]

    if opt.mirror and np.random.uniform() <= 0.3:
        crop_img = crop_img[:, :, ::-1].copy()
        crop_mask = crop_mask[:, :, ::-1].copy()

    return crop_img, crop_mask


def get_val_img(img_path, case_name):
    """验证数据加载 - DICOM 版本"""
    # 1mm (HR)
    case_mask_path = os.path.join(img_path, 'val/1mm/', case_name)
    if os.path.isdir(case_mask_path):
        tmp_mask = read_dicom_series(case_mask_path)
    else:
        case_mask_path = img_path + 'val/1mm/' + case_name + '.nii.gz'
        tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    # 5mm (LR)
    case_img_path = os.path.join(img_path, 'val/5mm/', case_name)
    if os.path.isdir(case_img_path):
        tmp_img = read_dicom_series(case_img_path)
    else:
        case_img_path = img_path + 'val/5mm/' + case_name + '.nii.gz'
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
    """测试数据加载 - DICOM 版本"""
    # 1mm (HR)
    case_mask_path = os.path.join(img_path, 'test/1mm/', case_name)
    if os.path.isdir(case_mask_path):
        tmp_mask = read_dicom_series(case_mask_path)
    else:
        case_mask_path = img_path + 'test/1mm/' + case_name + '.nii.gz'
        tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    # 5mm (LR)
    case_img_path = os.path.join(img_path, 'test/5mm/', case_name)
    if os.path.isdir(case_img_path):
        tmp_img = read_dicom_series(case_img_path)
    else:
        case_img_path = img_path + 'test/5mm/' + case_name + '.nii.gz'
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
