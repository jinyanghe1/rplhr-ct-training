"""
宣武数据集专用 Dataset 类
Xuanwu Dataset Specific Dataset Classes

使用 in_model_xuanwu 加载数据，支持保守数据增强
"""

import numpy as np
from utils import in_model_xuanwu
from config import opt
import torch.nn.functional as F


class train_Dataset:
    """宣武数据集训练数据集类"""
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y - 使用宣武数据加载模块
        tmp_img, tmp_mask = in_model_xuanwu.get_train_img(self.img_path, case_name)

        img = tmp_img[np.newaxis]
        mask = tmp_mask[np.newaxis]

        return_list = [case_name, img, mask]
        return return_list

    def __len__(self):
        return len(self.img_list)


class val_Dataset:
    """宣武数据集验证数据集类"""
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y - 使用宣武数据加载模块
        crop_img, pos_list, tmp_mask = in_model_xuanwu.get_val_img(self.img_path, case_name)

        return_list = [case_name, crop_img, tmp_mask, pos_list]

        return return_list

    def __len__(self):
        return len(self.img_list)


class test_Dataset:
    """宣武数据集测试数据集类"""
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y - 使用宣武数据加载模块
        crop_img, pos_list, tmp_mask = in_model_xuanwu.get_test_img(self.img_path, case_name)

        return_list = [case_name, crop_img, tmp_mask, pos_list]

        return return_list

    def __len__(self):
        return len(self.img_list)
