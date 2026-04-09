import numpy as np
from utils import in_model_xuanwu
from config import opt
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates


class DataAugmentation:
    """数据增强类 - 用于训练时的数据增强"""
    
    def __init__(self, 
                 noise_prob=0.5,           # 噪声增强概率
                 poisson_prob=0.5,         # 泊松噪声 vs 高斯噪声的选择概率
                 poisson_scale=0.1,        # 泊松噪声强度
                 gaussian_sigma=0.05,      # 高斯噪声标准差
                 elastic_prob=0.3,         # 弹性形变概率
                 elastic_sigma=8,          # 弹性形变强度 (ROADMAP A2)
                 elastic_alpha=100):       # 弹性形变缩放因子
        self.noise_prob = noise_prob
        self.poisson_prob = poisson_prob
        self.poisson_scale = poisson_scale
        self.gaussian_sigma = gaussian_sigma
        self.elastic_prob = elastic_prob
        self.elastic_sigma = elastic_sigma
        self.elastic_alpha = elastic_alpha
    
    def add_poisson_noise(self, image):
        """
        添加泊松噪声 - 模拟CT物理特性 (ROADMAP A1)
        泊松噪声与信号强度相关，更符合CT成像原理
        """
        # 归一化到合适的范围进行泊松采样
        scaled = image * (255.0 / self.poisson_scale)
        scaled = np.clip(scaled, 0, None)
        
        # 生成泊松噪声
        noisy = np.random.poisson(scaled).astype(np.float32)
        
        # 还原到原始尺度
        result = noisy * (self.poisson_scale / 255.0)
        
        # 保持数据范围
        result = np.clip(result, image.min(), image.max())
        return result
    
    def add_gaussian_noise(self, image):
        """
        添加高斯噪声 (ROADMAP A1)
        """
        noise = np.random.normal(0, self.gaussian_sigma, image.shape).astype(np.float32)
        result = image + noise
        return result
    
    def apply_noise(self, image):
        """
        应用随机噪声增强
        """
        if np.random.rand() > self.noise_prob:
            return image
        
        if np.random.rand() < self.poisson_prob:
            return self.add_poisson_noise(image)
        else:
            return self.add_gaussian_noise(image)
    
    def generate_elastic_deformation(self, shape, sigma, alpha):
        """
        生成3D弹性形变场 (Thin Plate Spline近似)
        """
        # 生成随机位移场
        dz = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1
        dx = np.random.rand(*shape) * 2 - 1
        
        # 使用高斯滤波平滑位移场 (模拟TPS效果)
        dz = gaussian_filter(dz, sigma=sigma, mode='reflect') * alpha
        dy = gaussian_filter(dy, sigma=sigma, mode='reflect') * alpha
        dx = gaussian_filter(dx, sigma=sigma, mode='reflect') * alpha
        
        # 生成网格坐标
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # 应用位移
        indices = (z + dz).ravel(), (y + dy).ravel(), (x + dx).ravel()
        
        return indices
    
    def apply_elastic_transform(self, image, mask):
        """
        应用3D弹性形变 - 对image和mask应用相同的形变 (ROADMAP A2)
        """
        if np.random.rand() > self.elastic_prob:
            return image, mask
        
        shape = image.shape
        indices = self.generate_elastic_deformation(shape, self.elastic_sigma, self.elastic_alpha)
        
        # 对image应用形变
        image_deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
        # 对mask应用形变 (使用最近邻插值保持标签值)
        mask_deformed = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
        
        # 确保数据范围正确
        image_deformed = np.clip(image_deformed, image.min(), image.max())
        mask_deformed = np.clip(mask_deformed, 0, 1)
        
        return image_deformed, mask_deformed
    
    def __call__(self, image, mask):
        """
        应用所有增强
        注意: 先应用弹性形变，再添加噪声 (避免噪声被形变平滑)
        """
        # 1. 3D弹性形变 (同时对image和mask应用)
        image, mask = self.apply_elastic_transform(image, mask)
        
        # 2. 随机噪声 (只对image应用)
        image = self.apply_noise(image)
        
        return image, mask


class train_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        # 初始化数据增强器 (仅训练时使用)
        self.augmentation = DataAugmentation(
            noise_prob=0.5,           # 50%概率应用噪声
            poisson_prob=0.6,         # 60%概率用泊松噪声
            poisson_scale=0.1,        # 泊松噪声强度
            gaussian_sigma=0.03,      # 高斯噪声标准差 (3%)
            elastic_prob=0.3,         # 30%概率应用弹性形变
            elastic_sigma=8,          # 形变强度 (ROADMAP A2要求)
            elastic_alpha=100         # 形变缩放
        )
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y
        tmp_img, tmp_mask = in_model_xuanwu.get_train_img(self.img_path, case_name)

        # 应用数据增强 (仅训练时)
        tmp_img, tmp_mask = self.augmentation(tmp_img, tmp_mask)

        img = tmp_img[np.newaxis]
        mask = tmp_mask[np.newaxis]

        return_list = [case_name, img, mask]
        return return_list

    def __len__(self):
        return len(self.img_list)

class val_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y
        crop_img, pos_list, tmp_mask = in_model_xuanwu.get_val_img(self.img_path, case_name)

        return_list = [case_name, crop_img, tmp_mask, pos_list]

        return return_list

    def __len__(self):
        return len(self.img_list)

class test_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        case_name = self.img_list[idx]

        # for x and y
        crop_img, pos_list, tmp_mask = in_model_xuanwu.get_test_img(self.img_path, case_name)

        return_list = [case_name, crop_img, tmp_mask, pos_list]

        return return_list

    def __len__(self):
        return len(self.img_list)
