"""
噪声增强模块
Noise Augmentation Module

支持以下噪声类型：
- Gaussian: 高斯噪声，模拟低剂量CT的量子噪声
- Poisson: 泊松噪声，更符合CT物理成像过程
- Both: 同时应用高斯和泊松噪声

噪声只添加到LR输入，HR目标保持干净。
"""

import numpy as np
import random
from typing import Tuple, Optional, Union
from .base_augment import BaseAugment, AugmentResult


class NoiseAugment(BaseAugment):
    """
    随机噪声增强器
    
    模拟CT扫描中的噪声，主要用于低剂量CT模拟。
    支持高斯噪声、泊松噪声或两者混合。
    
    Attributes:
        prob: 应用概率
        noise_type: 噪声类型 ('gaussian', 'poisson', 'both')
        sigma: 高斯噪声标准差（相对值）
        scale: 泊松噪声缩放因子
    
    Example:
        >>> # 仅高斯噪声
        >>> aug = NoiseAugment(prob=0.5, noise_type='gaussian', sigma=0.01)
        >>> 
        >>> # 泊松噪声
        >>> aug = NoiseAugment(prob=0.5, noise_type='poisson', scale=1.0)
        >>> 
        >>> # 混合噪声
        >>> aug = NoiseAugment(prob=0.5, noise_type='both', sigma=0.01)
    """
    
    def __init__(self, 
                 prob: float = 0.5,
                 noise_type: str = 'gaussian',
                 sigma: float = 0.01,
                 scale: float = 1.0,
                 random_state: Optional[int] = None):
        """
        初始化噪声增强器
        
        Args:
            prob: 应用概率 (0-1)
            noise_type: 噪声类型，可选 'gaussian', 'poisson', 'both'
            sigma: 高斯噪声标准差（相对于图像最大值的归一化值）
                   例如：0.01 表示噪声标准差为图像最大值的1%
            scale: 泊松噪声缩放因子
            random_state: 随机种子
            
        Raises:
            ValueError: 如果noise_type无效
        """
        super().__init__('NoiseAugment', prob, random_state)
        
        valid_types = ['gaussian', 'poisson', 'both']
        noise_type = noise_type.lower()
        if noise_type not in valid_types:
            raise ValueError(f"Invalid noise_type '{noise_type}'. "
                           f"Valid types: {valid_types}")
        
        self.noise_type = noise_type
        self.sigma = sigma
        self.scale = scale
    
    def _add_gaussian_noise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            img: 输入图像
            sigma: 噪声标准差（相对值）
            
        Returns:
            加噪后的图像
        """
        max_val = max(abs(img.max()), abs(img.min()))
        if max_val < 1e-8:
            return img
        
        std = sigma * max_val
        noise = np.random.normal(0, std, img.shape)
        return img + noise
    
    def _add_poisson_noise(self, img: np.ndarray, scale: float) -> np.ndarray:
        """
        添加泊松噪声
        
        泊松噪声更符合CT的量子噪声特性。
        先将图像缩放到合适的范围，应用泊松噪声，再缩放回原始范围。
        
        Args:
            img: 输入图像
            scale: 缩放因子
            
        Returns:
            加噪后的图像
        """
        # 归一化到正值（泊松分布需要正值）
        img_min = img.min()
        img_shifted = img - img_min
        max_val = img_shifted.max()
        
        if max_val < 1e-8:
            return img
        
        # 缩放到合适范围并添加泊松噪声
        scaled = img_shifted / max_val * scale * 1000
        noisy = np.random.poisson(scaled)
        
        # 缩放回原始范围
        noisy = noisy / (scale * 1000) * max_val + img_min
        
        return noisy.astype(img.dtype)
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """
        应用噪声增强
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W)
            is_training: 是否为训练模式
            
        Returns:
            AugmentResult: 增强结果
            
        Note:
            噪声只添加到LR，HR保持干净
        """
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        lr_noisy = lr_img.copy()
        applied_types = []
        
        # 根据噪声类型添加噪声
        if self.noise_type in ['gaussian', 'both']:
            lr_noisy = self._add_gaussian_noise(lr_noisy, self.sigma)
            applied_types.append('gaussian')
        
        if self.noise_type in ['poisson', 'both']:
            lr_noisy = self._add_poisson_noise(lr_noisy, self.scale)
            applied_types.append('poisson')
        
        return AugmentResult(
            lr_img=lr_noisy,
            hr_img=hr_img,  # HR保持不变
            applied=True,
            aug_name=self.name,
            params={
                'noise_type': self.noise_type,
                'applied_types': applied_types,
                'sigma': self.sigma if 'gaussian' in applied_types else None,
                'scale': self.scale if 'poisson' in applied_types else None,
            }
        )
    
    def get_config(self) -> dict:
        """获取配置"""
        config = super().get_config()
        config.update({
            'noise_type': self.noise_type,
            'sigma': self.sigma,
            'scale': self.scale,
        })
        return config


class SpeckleNoiseAugment(BaseAugment):
    """
    散斑噪声增强器（乘性噪声）
    
    模拟某些类型的伪影，噪声模型：output = input + input * noise
    
    Example:
        >>> aug = SpeckleNoiseAugment(prob=0.3, var=0.01)
    """
    
    def __init__(self, prob: float = 0.3, var: float = 0.01,
                 random_state: Optional[int] = None):
        """
        初始化
        
        Args:
            prob: 应用概率
            var: 噪声方差
            random_state: 随机种子
        """
        super().__init__('SpeckleNoiseAugment', prob, random_state)
        self.var = var
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """应用散斑噪声"""
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        noise = np.random.normal(0, self.var, lr_img.shape)
        lr_noisy = lr_img + lr_img * noise
        
        return AugmentResult(
            lr_img=lr_noisy,
            hr_img=hr_img,
            applied=True,
            aug_name=self.name,
            params={'var': self.var}
        )
