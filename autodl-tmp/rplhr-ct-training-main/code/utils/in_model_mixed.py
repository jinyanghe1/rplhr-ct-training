#!/usr/bin/env python3
"""
支持混合Ratio的数据加载模块

支持同一份训练数据中包含不同超分比例的数据，例如：
- 部分数据: 5mm -> 1mm (5x)
- 部分数据: 4mm -> 1mm (4x)
- 部分数据: 5mm -> 1.25mm (4x)

使用方法:
1. 在 make_dataset.py 中修改导入:
   - 将: from utils import in_model
   - 改为: from utils import in_model_mixed as in_model
   
2. 创建 case_ratio_config.json 配置文件，指定每个case的ratio
   或使用自动检测模式（从目录名推断ratio）

目录结构示例:
    data/
    ├── train/
    │   ├── thick/          # 低分辨率数据（如5mm）
    │   │   ├── case_001.nii.gz
    │   │   └── case_002.nii.gz
    │   └── thin/           # 高分辨率数据（如1mm）
    │       ├── case_001.nii.gz
    │       └── case_002.nii.gz
    └── case_ratio_config.json  # (可选) 手动指定每个case的ratio
"""

import numpy as np
import SimpleITK as sitk
import random
import os
import json
from copy import deepcopy
from config import opt
from builtins import range


def get_case_ratio(case_name, img_path=None, subset='train'):
    """
    获取指定case的超分ratio
    
    优先级:
    1. 从 case_ratio_config.json 读取（如果存在）
    2. 从全局 opt.ratio 读取（默认）
    3. 从数据shape自动计算（如果开启auto_detect）
    
    Args:
        case_name: 病例名称
        img_path: 数据根目录（用于自动检测）
        subset: 'train', 'val', 或 'test'
    
    Returns:
        int: 超分比例 (如 4 或 5)
    """
    # 1. 尝试从配置文件读取
    if img_path is not None:
        config_path = os.path.join(img_path, 'case_ratio_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                ratio_config = json.load(f)
            if case_name in ratio_config:
                return ratio_config[case_name]
    
    # 2. 从全局配置读取（回退）
    if hasattr(opt, 'ratio'):
        return opt.ratio
    
    # 3. 默认值
    return 5


def get_lr_hr_paths(img_path, subset, case_name, lr_dir='thick', hr_dir='thin'):
    """
    获取低分辨率和高分辨率数据的完整路径
    
    支持灵活的路径命名:
    - 传统命名: 5mm/1mm
    - 通用命名: thick/thin
    
    Args:
        img_path: 数据根目录
        subset: 'train', 'val', 或 'test'
        case_name: 病例名称
        lr_dir: 低分辨率目录名（默认'thick'）
        hr_dir: 高分辨率目录名（默认'thin'）
    
    Returns:
        tuple: (lr_path, hr_path)
    """
    # 尝试从opt读取路径配置
    if hasattr(opt, 'lr_dir'):
        lr_dir = opt.lr_dir
    if hasattr(opt, 'hr_dir'):
        hr_dir = opt.hr_dir
    
    # 尝试多种可能的目录命名
    possible_lr_names = [lr_dir, '5mm', 'thick', 'LR']
    possible_hr_names = [hr_dir, '1mm', 'thin', 'HR']
    
    base_path = os.path.join(img_path, subset)
    
    # 查找LR路径
    lr_path = None
    for lr_name in possible_lr_names:
        test_path = os.path.join(base_path, lr_name, f'{case_name}.nii.gz')
        if os.path.exists(test_path):
            lr_path = test_path
            break
    
    # 查找HR路径
    hr_path = None
    for hr_name in possible_hr_names:
        test_path = os.path.join(base_path, hr_name, f'{case_name}.nii.gz')
        if os.path.exists(test_path):
            hr_path = test_path
            break
    
    if lr_path is None:
        raise FileNotFoundError(f"找不到LR数据: {case_name} (尝试了: {possible_lr_names})")
    if hr_path is None:
        raise FileNotFoundError(f"找不到HR数据: {case_name} (尝试了: {possible_hr_names})")
    
    return lr_path, hr_path


def calculate_output_slices(z_s, z_e, ratio, offset=3):
    """
    计算输出切片位置（支持任意ratio）
    
    Args:
        z_s: 输入起始位置
        z_e: 输入结束位置
        ratio: 超分比例
        offset: 边缘偏移量（用于对齐）
    
    Returns:
        tuple: (out_z_s, out_z_e)
    """
    # 通用公式: 将输入位置映射到输出位置
    # 对于ratio=5: z_s * 5 + 3, (z_e - 1) * 5 - 2
    # 对于ratio=4: z_s * 4 + 2, (z_e - 1) * 4 - 1
    # 以此类推...
    
    # 计算合适的偏移量
    if offset is None:
        offset = ratio // 2  # 自动计算中心偏移
    
    out_z_s = z_s * ratio + offset
    out_z_e = (z_e - 1) * ratio - (ratio - offset - 1)
    
    return out_z_s, out_z_e


def get_train_img(img_path, case_name):
    """
    训练数据加载 - 支持混合Ratio版本
    
    关键修改:
    - 动态获取ratio（每个case可以不同）
    - 使用calculate_output_slices计算输出位置
    """
    # 获取该case的ratio
    ratio = get_case_ratio(case_name, img_path, 'train')
    
    # 获取数据路径
    lr_path, hr_path = get_lr_hr_paths(img_path, 'train', case_name)
    
    # 读取数据
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(hr_path))
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(lr_path))

    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - opt.c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + opt.c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    # 使用动态ratio计算输出切片位置
    mask_z_s, mask_z_e = calculate_output_slices(z_s, z_e, ratio)

    crop_mask = tmp_mask[mask_z_s:mask_z_e, y_s:y_e, x_s:x_e]

    if opt.mirror and np.random.uniform() <= 0.3:
        crop_img = crop_img[:, :, ::-1].copy()
        crop_mask = crop_mask[:, :, ::-1].copy()

    return crop_img, crop_mask


def get_val_img(img_path, case_name):
    """验证数据加载 - 支持混合Ratio版本"""
    # 获取该case的ratio
    ratio = get_case_ratio(case_name, img_path, 'val')
    
    # 获取数据路径
    lr_path, hr_path = get_lr_hr_paths(img_path, 'val', case_name)
    
    # 读取数据
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(hr_path))
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(lr_path))

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
    """测试数据加载 - 支持混合Ratio版本"""
    # 获取该case的ratio
    ratio = get_case_ratio(case_name, img_path, 'test')
    
    # 获取数据路径
    lr_path, hr_path = get_lr_hr_paths(img_path, 'test', case_name)
    
    # 读取数据
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(hr_path))
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(lr_path))

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


def auto_detect_ratios(img_path, subset='train', lr_dir='thick', hr_dir='thin'):
    """
    自动检测所有case的ratio（基于数据shape）
    
    使用方法:
        ratios = auto_detect_ratios('../data/', 'train')
        # 保存到配置文件
        with open('../data/case_ratio_config.json', 'w') as f:
            json.dump(ratios, f, indent=2)
    
    Returns:
        dict: {case_name: ratio, ...}
    """
    ratios = {}
    
    base_path = os.path.join(img_path, subset)
    lr_path = os.path.join(base_path, lr_dir)
    hr_path = os.path.join(base_path, hr_dir)
    
    if not os.path.exists(lr_path) or not os.path.exists(hr_path):
        print(f"警告: 路径不存在 {lr_path} 或 {hr_path}")
        return ratios
    
    lr_cases = set([f.replace('.nii.gz', '') for f in os.listdir(lr_path) if f.endswith('.nii.gz')])
    hr_cases = set([f.replace('.nii.gz', '') for f in os.listdir(hr_path) if f.endswith('.nii.gz')])
    common_cases = lr_cases & hr_cases
    
    print(f"发现 {len(common_cases)} 个匹配的case")
    
    for case_name in sorted(common_cases):
        try:
            lr_img = sitk.ReadImage(os.path.join(lr_path, f'{case_name}.nii.gz'))
            hr_img = sitk.ReadImage(os.path.join(hr_path, f'{case_name}.nii.gz'))
            
            lr_array = sitk.GetArrayFromImage(lr_img)
            hr_array = sitk.GetArrayFromImage(hr_img)
            
            # 计算Z方向比例（四舍五入到最近整数）
            z_ratio = round(hr_array.shape[0] / lr_array.shape[0])
            ratios[case_name] = z_ratio
            
            print(f"  {case_name}: LR{lr_array.shape} -> HR{hr_array.shape}, ratio={z_ratio}")
            
        except Exception as e:
            print(f"  {case_name}: 读取失败 - {e}")
    
    return ratios


if __name__ == '__main__':
    # 测试脚本
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"自动检测ratio: {data_path}")
        ratios = auto_detect_ratios(data_path, 'train')
        
        # 统计
        ratio_counts = {}
        for r in ratios.values():
            ratio_counts[r] = ratio_counts.get(r, 0) + 1
        
        print("\nRatio统计:")
        for r, count in sorted(ratio_counts.items()):
            print(f"  {r}x: {count} cases")
        
        # 保存配置
        config_path = os.path.join(data_path, 'case_ratio_config.json')
        with open(config_path, 'w') as f:
            json.dump(ratios, f, indent=2)
        print(f"\n配置已保存: {config_path}")
