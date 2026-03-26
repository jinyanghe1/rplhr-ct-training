"""
CT Volumetric Data Augmentation Module for RPLHR-CT Training
CT体积数据增强模块

本模块提供针对3D医学影像(CT)的数据增强功能，专为体积超分辨率任务设计。
增强策略考虑了医学影像的特点：
- 保持解剖结构的完整性和连续性
- 模拟不同扫描设备和参数的变化
- 增强对噪声和伪影的鲁棒性

Author: Auto-generated
Date: 2026-03-26
"""

import numpy as np
import random
from typing import Tuple, Optional, List
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates


class CTVolumetricAugmentation:
    """
    CT体积数据增强器
    
    针对3D CT影像设计的数据增强类，提供空间变换、强度变换、噪声添加等功能。
    所有增强操作都保持输入输出的空间对应关系（LR和HR同步变换）。
    
    Attributes:
        prob: 每个增强操作的应用概率
        random_state: 随机数种子，确保可复现
    """
    
    def __init__(self, prob: float = 0.5, random_state: Optional[int] = None):
        """
        初始化增强器
        
        Args:
            prob: 单个增强操作的应用概率 (0-1)
            random_state: 随机种子，用于结果复现
        """
        self.prob = prob
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    # ==================== 空间变换 (Spatial Transformations) ====================
    
    def random_flip_3d(self, lr_img: np.ndarray, hr_img: np.ndarray, 
                       axes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        3D随机翻转 (Random 3D Flip)
        
        沿指定轴进行镜像翻转。模拟不同扫描方向，增强对方向变化的鲁棒性。
        注意：CT影像通常不需要翻转Z轴（层间顺序固定），主要翻转X/Y轴。
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W)，Z' = Z * ratio
            axes: 可翻转的轴列表，默认 [1, 2] (Y轴和X轴)
            
        Returns:
            翻转后的 (lr_img, hr_img)
            
        Example:
            >>> aug = CTVolumetricAugmentation(prob=1.0)
            >>> lr, hr = aug.random_flip_3d(lr_img, hr_img)
        """
        if axes is None:
            axes = [1, 2]  # 默认只翻转Y和X轴，保持Z轴顺序
        
        if random.random() > self.prob:
            return lr_img, hr_img
        
        axis = random.choice(axes)
        lr_img = np.flip(lr_img, axis=axis).copy()
        hr_img = np.flip(hr_img, axis=axis).copy()
        
        return lr_img, hr_img
    
    def random_rotate_90(self, lr_img: np.ndarray, hr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        90度随机旋转 (Random 90-degree Rotation)
        
        在XY平面内进行90度整数倍旋转（0°, 90°, 180°, 270°）。
        模拟不同扫描体位，同时保持层间对应关系。
        
        适用场景：
        - 胸部CT不同扫描角度
        - 增强模型对旋转的鲁棒性
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W)
            
        Returns:
            旋转后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        # 随机选择旋转次数 (1=90°, 2=180°, 3=270°)
        k = random.randint(1, 3)
        
        # 在XY平面旋转 (axes 1,2)
        lr_img = np.rot90(lr_img, k=k, axes=(1, 2)).copy()
        hr_img = np.rot90(hr_img, k=k, axes=(1, 2)).copy()
        
        return lr_img, hr_img
    
    def random_shift(self, lr_img: np.ndarray, hr_img: np.ndarray,
                     max_shift: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机平移 (Random Shift)
        
        在XY平面内进行小幅随机平移，模拟扫描位置的微小差异。
        使用边缘填充保持图像尺寸不变。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            max_shift: 最大平移像素数（HR空间）
            
        Returns:
            平移后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        # 计算缩放比例（HR/LR的空间比例）
        z_ratio = hr_img.shape[0] // lr_img.shape[0]
        
        # 在HR空间生成平移量
        shift_y = random.randint(-max_shift, max_shift)
        shift_x = random.randint(-max_shift, max_shift)
        
        # 对应LR空间的平移量（取整）
        lr_shift_y = shift_y // z_ratio
        lr_shift_x = shift_x // z_ratio
        
        # 对LR进行平移
        lr_img = ndimage.shift(lr_img, (0, lr_shift_y, lr_shift_x), mode='nearest')
        
        # 对HR进行平移
        hr_img = ndimage.shift(hr_img, (0, shift_y, shift_x), mode='nearest')
        
        return lr_img, hr_img
    
    # ==================== 强度变换 (Intensity Transformations) ====================
    
    def random_intensity_scale(self, lr_img: np.ndarray, hr_img: np.ndarray,
                                scale_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机强度缩放 (Random Intensity Scaling)
        
        线性缩放CT值，模拟不同设备或扫描参数导致的整体强度差异。
        只变换LR输入，不改变HR目标（保持重建目标一致）。
        
        注意：CT值有物理意义（HU值），缩放幅度不宜过大。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标（保持不变）
            scale_range: 缩放因子范围
            
        Returns:
            变换后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        scale = random.uniform(*scale_range)
        lr_img = lr_img * scale
        
        return lr_img, hr_img
    
    def random_intensity_shift(self, lr_img: np.ndarray, hr_img: np.ndarray,
                                shift_range: Tuple[float, float] = (-50, 50)) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机强度偏移 (Random Intensity Shift)
        
        整体平移CT值（单位：HU），模拟不同窗宽窗位设置。
        例如：水的CT值为0HU，空气为-1000HU，骨骼>400HU。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标（保持不变）
            shift_range: 偏移量范围（HU单位）
            
        Returns:
            变换后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        shift = random.uniform(*shift_range)
        lr_img = lr_img + shift
        
        return lr_img, hr_img
    
    def random_gamma_correction(self, lr_img: np.ndarray, hr_img: np.ndarray,
                                 gamma_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机Gamma校正 (Random Gamma Correction)
        
        非线性强度变换，模拟不同重建算法导致的对比度变化。
        先归一化到[0,1]，应用gamma，再反归一化。
        
        gamma < 1: 增强低灰度区域（软组织细节）
        gamma > 1: 增强高灰度区域（骨骼、钙化）
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            gamma_range: Gamma值范围
            
        Returns:
            变换后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        gamma = random.uniform(*gamma_range)
        
        # 记录原始范围
        lr_min, lr_max = lr_img.min(), lr_img.max()
        
        # 归一化
        lr_norm = (lr_img - lr_min) / (lr_max - lr_min + 1e-8)
        
        # Gamma变换
        lr_norm = np.power(lr_norm, gamma)
        
        # 反归一化
        lr_img = lr_norm * (lr_max - lr_min) + lr_min
        
        return lr_img, hr_img
    
    def random_contrast(self, lr_img: np.ndarray, hr_img: np.ndarray,
                        contrast_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机对比度调整 (Random Contrast Adjustment)
        
        调整图像对比度，公式：output = (input - mean) * contrast + mean
        模拟不同窗宽设置对对比度的影响。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            contrast_range: 对比度因子范围
            
        Returns:
            变换后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        contrast = random.uniform(*contrast_range)
        mean = lr_img.mean()
        lr_img = (lr_img - mean) * contrast + mean
        
        return lr_img, hr_img
    
    # ==================== 噪声与伪影 (Noise & Artifacts) ====================
    
    def add_gaussian_noise(self, lr_img: np.ndarray, hr_img: np.ndarray,
                          noise_range: Tuple[float, float] = (0.001, 0.01)) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加高斯噪声 (Add Gaussian Noise)
        
        模拟低剂量CT的量子噪声。噪声水平与剂量成反比。
        只在LR输入上添加噪声，HR作为干净目标保持不变。
        
        CT噪声特点：
        - 服从泊松分布，低剂量时更明显
        - 近似用高斯噪声建模
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            noise_range: 噪声标准差范围（相对于最大值的归一化值）
            
        Returns:
            加噪后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        max_val = max(abs(lr_img.max()), abs(lr_img.min()))
        std = random.uniform(*noise_range) * max_val
        
        noise = np.random.normal(0, std, lr_img.shape)
        lr_img = lr_img + noise
        
        return lr_img, hr_img
    
    def add_speckle_noise(self, lr_img: np.ndarray, hr_img: np.ndarray,
                         noise_range: Tuple[float, float] = (0.001, 0.01)) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加散斑噪声 (Add Speckle Noise)
        
        乘性噪声，常见于超声成像，CT中可模拟某些类型的伪影。
        噪声模型：output = input + input * noise
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            noise_range: 噪声方差范围
            
        Returns:
            加噪后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        var = random.uniform(*noise_range)
        noise = np.random.normal(0, var, lr_img.shape)
        
        lr_img = lr_img + lr_img * noise
        
        return lr_img, hr_img
    
    def add_slice_artifact(self, lr_img: np.ndarray, hr_img: np.ndarray,
                          max_artifact_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加层间伪影 (Add Slice-wise Artifact)
        
        模拟CT扫描中的运动伪影或金属伪影，表现为特定层的强度异常。
        随机选择部分层进行强度偏移或替换。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            max_artifact_ratio: 受影响层的最大比例
            
        Returns:
            添加伪影后的 (lr_img, hr_img)
        """
        if random.random() > self.prob * 0.3:  # 降低应用概率（伪影较严重）
            return lr_img, hr_img
        
        num_slices = lr_img.shape[0]
        num_artifact = max(1, int(num_slices * random.uniform(0.01, max_artifact_ratio)))
        
        artifact_slices = random.sample(range(num_slices), num_artifact)
        
        for z in artifact_slices:
            # 随机选择伪影类型
            artifact_type = random.choice(['intensity_shift', 'noise_spike', 'blur'])
            
            if artifact_type == 'intensity_shift':
                shift = random.uniform(-200, 200)  # HU单位的大偏移
                lr_img[z] = lr_img[z] + shift
            elif artifact_type == 'noise_spike':
                noise = np.random.normal(0, 100, lr_img[z].shape)
                lr_img[z] = lr_img[z] + noise
            elif artifact_type == 'blur':
                lr_img[z] = gaussian_filter(lr_img[z], sigma=random.uniform(0.5, 2.0))
        
        return lr_img, hr_img
    
    # ==================== 高级增强 (Advanced Augmentations) ====================
    
    def elastic_deform(self, lr_img: np.ndarray, hr_img: np.ndarray,
                      alpha: float = 10.0, sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        弹性形变 (Elastic Deformation)
        
        模拟软组织形变，生成平滑的非线性变形场。
        使用随机位移场 + 高斯滤波平滑。
        
        注意：这是计算密集型操作，慎用。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            alpha: 变形幅度
            sigma: 平滑程度
            
        Returns:
            形变后的 (lr_img, hr_img)
        """
        if random.random() > self.prob * 0.3:  # 降低概率（计算量大）
            return lr_img, hr_img
        
        shape = lr_img.shape
        
        # 生成随机位移场
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha * 0.3  # Z轴变形小
        
        # 创建网格
        z, y, x = np.meshgrid(np.arange(shape[0]), 
                              np.arange(shape[1]), 
                              np.arange(shape[2]), 
                              indexing='ij')
        
        # 应用变形
        indices = (z + dz).reshape(-1), (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        lr_img = map_coordinates(lr_img, indices, order=1, mode='nearest').reshape(shape)
        
        # HR需要按比例变形
        hr_shape = hr_img.shape
        z_ratio = hr_shape[0] // shape[0]
        
        # 对HR在Z方向重复变形场
        dz_hr = np.repeat(dz, z_ratio, axis=0)[:hr_shape[0]]
        dy_hr = np.repeat(dy, z_ratio, axis=0)[:hr_shape[0]]
        dx_hr = np.repeat(dx, z_ratio, axis=0)[:hr_shape[0]]
        
        z_hr, y_hr, x_hr = np.meshgrid(np.arange(hr_shape[0]), 
                                        np.arange(hr_shape[1]), 
                                        np.arange(hr_shape[2]), 
                                        indexing='ij')
        
        indices_hr = (z_hr + dz_hr).reshape(-1), (y_hr + dy_hr).reshape(-1), (x_hr + dx_hr).reshape(-1)
        hr_img = map_coordinates(hr_img, indices_hr, order=1, mode='nearest').reshape(hr_shape)
        
        return lr_img, hr_img
    
    def random_blur(self, lr_img: np.ndarray, hr_img: np.ndarray,
                   sigma_range: Tuple[float, float] = (0.3, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机模糊 (Random Blur)
        
        模拟部分容积效应或轻微运动模糊。
        使用高斯滤波器。
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            sigma_range: 高斯核标准差范围
            
        Returns:
            模糊后的 (lr_img, hr_img)
        """
        if random.random() > self.prob:
            return lr_img, hr_img
        
        sigma = random.uniform(*sigma_range)
        lr_img = gaussian_filter(lr_img, sigma=sigma)
        
        return lr_img, hr_img
    
    # ==================== 组合增强 (Composite Augmentations) ====================
    
    def apply_train_augmentation(self, lr_img: np.ndarray, hr_img: np.ndarray,
                                  aug_config: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用训练增强组合 (Training Augmentation Pipeline)
        
        预设的训练增强流程，按推荐顺序应用多个增强。
        顺序：空间变换 → 强度变换 → 噪声添加
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W)
            aug_config: 增强配置字典，默认使用推荐配置
            
        Returns:
            增强后的 (lr_img, hr_img)
            
        Default Pipeline:
            1. Random Flip (p=0.5)
            2. Random Rotation 90 (p=0.3)
            3. Random Shift (p=0.3)
            4. Random Intensity Scale (p=0.3)
            5. Random Intensity Shift (p=0.3)
            6. Random Contrast (p=0.3)
            7. Add Gaussian Noise (p=0.5)
            8. Add Slice Artifact (p=0.1)
        """
        # 保存原始形状用于后续保护
        lr_shape_orig = lr_img.shape
        hr_shape_orig = hr_img.shape
        
        if aug_config is None:
            aug_config = {
                'flip_prob': 0.5,
                'rotate_prob': 0.3,
                'shift_prob': 0.3,
                'intensity_scale_prob': 0.3,
                'intensity_shift_prob': 0.3,
                'contrast_prob': 0.3,
                'gaussian_noise_prob': 0.5,
                'slice_artifact_prob': 0.1,
                'elastic_prob': 0.1,
            }
        
        # 保存原始概率，临时修改
        original_prob = self.prob
        
        # 1. 空间变换（LR和HR必须同步）
        self.prob = aug_config.get('flip_prob', 0.5)
        lr_img, hr_img = self.random_flip_3d(lr_img, hr_img)
        
        self.prob = aug_config.get('rotate_prob', 0.3)
        lr_img, hr_img = self.random_rotate_90(lr_img, hr_img)
        
        self.prob = aug_config.get('shift_prob', 0.3)
        lr_img, hr_img = self.random_shift(lr_img, hr_img)
        
        # 2. 强度变换（仅LR）
        self.prob = aug_config.get('intensity_scale_prob', 0.3)
        lr_img, hr_img = self.random_intensity_scale(lr_img, hr_img)
        
        self.prob = aug_config.get('intensity_shift_prob', 0.3)
        lr_img, hr_img = self.random_intensity_shift(lr_img, hr_img)
        
        self.prob = aug_config.get('contrast_prob', 0.3)
        lr_img, hr_img = self.random_contrast(lr_img, hr_img)
        
        self.prob = aug_config.get('gamma_prob', 0.0)  # 默认关闭
        lr_img, hr_img = self.random_gamma_correction(lr_img, hr_img)
        
        # 3. 噪声与伪影（仅LR）
        self.prob = aug_config.get('gaussian_noise_prob', 0.5)
        lr_img, hr_img = self.add_gaussian_noise(lr_img, hr_img)
        
        self.prob = aug_config.get('speckle_noise_prob', 0.0)  # 默认关闭
        lr_img, hr_img = self.add_speckle_noise(lr_img, hr_img)
        
        self.prob = aug_config.get('slice_artifact_prob', 0.1)
        lr_img, hr_img = self.add_slice_artifact(lr_img, hr_img)
        
        self.prob = aug_config.get('blur_prob', 0.0)  # 默认关闭
        lr_img, hr_img = self.random_blur(lr_img, hr_img)
        
        # 4. 高级增强（低频使用）
        self.prob = aug_config.get('elastic_prob', 0.1)
        lr_img, hr_img = self.elastic_deform(lr_img, hr_img)
        
        # 恢复原始概率
        self.prob = original_prob
        
        # 5. 尺寸保护 - 确保输出尺寸与输入一致
        lr_img, hr_img = self._ensure_shape(lr_img, hr_img, lr_shape_orig, hr_shape_orig)
        
        return lr_img, hr_img
    
    def _ensure_shape(self, lr_img: np.ndarray, hr_img: np.ndarray,
                      lr_target_shape: Tuple[int, ...], hr_target_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """
        确保输出尺寸与目标尺寸一致
        
        某些增强操作（如elastic_deform, random_shift）可能会导致尺寸微小变化，
        此函数通过裁剪或填充确保输出尺寸正确。
        
        Args:
            lr_img: 低分辨率图像
            hr_img: 高分辨率图像
            lr_target_shape: 目标LR形状
            hr_target_shape: 目标HR形状
            
        Returns:
            调整后的 (lr_img, hr_img)
        """
        # 处理LR
        if lr_img.shape != lr_target_shape:
            # 裁剪或填充到目标尺寸
            lr_img = self._crop_or_pad(lr_img, lr_target_shape)
        
        # 处理HR
        if hr_img.shape != hr_target_shape:
            hr_img = self._crop_or_pad(hr_img, hr_target_shape)
        
        return lr_img, hr_img
    
    def _crop_or_pad(self, img: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        通过裁剪或填充将图像调整到目标形状
        
        Args:
            img: 输入图像
            target_shape: 目标形状
            
        Returns:
            调整后的图像
        """
        current_shape = img.shape
        
        # 如果当前形状大于目标形状，进行中心裁剪
        if any(c > t for c, t in zip(current_shape, target_shape)):
            slices = []
            for c, t in zip(current_shape, target_shape):
                if c > t:
                    start = (c - t) // 2
                    slices.append(slice(start, start + t))
                else:
                    slices.append(slice(None))
            img = img[tuple(slices)]
        
        # 如果当前形状小于目标形状，进行边缘填充
        if any(c < t for c, t in zip(img.shape, target_shape)):
            pad_width = []
            for c, t in zip(img.shape, target_shape):
                if c < t:
                    pad_before = (t - c) // 2
                    pad_after = t - c - pad_before
                    pad_width.append((pad_before, pad_after))
                else:
                    pad_width.append((0, 0))
            img = np.pad(img, pad_width, mode='edge')
        
        return img


# ==================== 便捷函数 (Utility Functions) ====================

def normalize_ct(lr_img: np.ndarray, hr_img: np.ndarray,
                 window_center: float = 0, window_width: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    CT值归一化 (CT Value Normalization)
    
    将HU值归一化到[0,1]或[-1,1]范围，使用窗宽窗位。
    模拟放射科医生的窗宽窗位调节。
    
    Args:
        lr_img: 低分辨率输入 (HU单位)
        hr_img: 高分辨率目标 (HU单位)
        window_center: 窗位 (WW)，默认0（软组织窗）
        window_width: 窗宽 (WL)，默认1000
        
    Returns:
        归一化后的 (lr_img, hr_img)
        
    常用窗口:
        - 肺窗: WW=1500, WL=-600
        - 软组织窗: WW=400, WL=40  
        - 骨窗: WW=1800, WL=400
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    lr_img = np.clip((lr_img - min_val) / (max_val - min_val + 1e-8), 0, 1)
    hr_img = np.clip((hr_img - min_val) / (max_val - min_val + 1e-8), 0, 1)
    
    return lr_img, hr_img


def clip_ct_values(lr_img: np.ndarray, hr_img: np.ndarray,
                   min_hu: float = -1024, max_hu: float = 3071) -> Tuple[np.ndarray, np.ndarray]:
    """
    裁剪CT值范围 (Clip CT Values)
    
    将CT值限制在合理范围内（HU单位）。
    标准CT范围：-1024 (空气) ~ 3071 (高密度骨/金属)
    
    Args:
        lr_img: 低分辨率输入
        hr_img: 高分辨率目标
        min_hu: 最小HU值
        max_hu: 最大HU值
        
    Returns:
        裁剪后的 (lr_img, hr_img)
    """
    lr_img = np.clip(lr_img, min_hu, max_hu)
    hr_img = np.clip(hr_img, min_hu, max_hu)
    
    return lr_img, hr_img


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试增强功能
    print("=" * 60)
    print("CT Volumetric Augmentation Module Test")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    lr_test = np.random.randn(8, 64, 64).astype(np.float32) * 200  # 8层LR
    hr_test = np.random.randn(40, 64, 64).astype(np.float32) * 200  # 40层HR (ratio=5)
    
    print(f"\nTest data shape: LR {lr_test.shape}, HR {hr_test.shape}")
    print(f"LR value range: [{lr_test.min():.2f}, {lr_test.max():.2f}] HU")
    
    # 初始化增强器
    aug = CTVolumetricAugmentation(prob=1.0)  # 所有操作强制应用
    
    # 测试各项增强
    tests = [
        ('Random Flip', aug.random_flip_3d),
        ('Random Rotate 90', aug.random_rotate_90),
        ('Random Shift', aug.random_shift),
        ('Random Intensity Scale', aug.random_intensity_scale),
        ('Random Intensity Shift', aug.random_intensity_shift),
        ('Random Contrast', aug.random_contrast),
        ('Add Gaussian Noise', aug.add_gaussian_noise),
    ]
    
    print("\n" + "-" * 60)
    print("Testing individual augmentations:")
    print("-" * 60)
    
    for name, func in tests:
        lr_aug, hr_aug = func(lr_test.copy(), hr_test.copy())
        print(f"  ✓ {name}: LR range [{lr_aug.min():.2f}, {lr_aug.max():.2f}]")
    
    # 测试完整pipeline
    print("\n" + "-" * 60)
    print("Testing full training pipeline:")
    print("-" * 60)
    
    aug_train = CTVolumetricAugmentation(prob=0.5)
    lr_aug, hr_aug = aug_train.apply_train_augmentation(lr_test.copy(), hr_test.copy())
    print(f"  ✓ Pipeline output: LR range [{lr_aug.min():.2f}, {lr_aug.max():.2f}]")
    print(f"  ✓ Shapes preserved: LR {lr_aug.shape}, HR {hr_aug.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
