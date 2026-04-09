"""
强度变换增强模块
Intensity Transformation Augmentation Module

模拟不同的窗宽窗位设置、扫描参数变化等导致的强度变化。
这些变换只应用于LR输入，HR目标保持不变。

注意：CT值有物理意义（HU值），变换幅度不宜过大。
"""

import numpy as np
import random
from typing import Tuple, Optional
from .base_augment import BaseAugment, AugmentResult


class IntensityAugment(BaseAugment):
    """
    强度变换增强器
    
    包含多种强度变换操作，模拟CT扫描中的各种强度变化：
    - scale: 线性缩放
    - shift: 整体偏移
    - gamma: Gamma校正
    - contrast: 对比度调整
    
    这些变换只应用于LR输入，HR目标保持不变。
    
    Attributes:
        prob: 应用概率
        operations: 要应用的变换操作列表
        params: 各变换的参数范围
    
    Example:
        >>> # 启用所有强度变换
        >>> aug = IntensityAugment(prob=0.5, operations=['scale', 'shift', 'gamma'])
        >>> 
        >>> # 仅对比度调整
        >>> aug = IntensityAugment(prob=0.3, operations=['contrast'])
    """
    
    def __init__(self,
                 prob: float = 0.5,
                 operations: Optional[list] = None,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 shift_range: Tuple[float, float] = (-50, 50),
                 gamma_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 random_state: Optional[int] = None):
        """
        初始化强度变换增强器
        
        Args:
            prob: 应用概率
            operations: 变换操作列表，可选：
                       'scale', 'shift', 'gamma', 'contrast'
                       默认全部启用
            scale_range: 强度缩放范围
            shift_range: 强度偏移范围（HU单位）
            gamma_range: Gamma校正范围
            contrast_range: 对比度调整范围
            random_state: 随机种子
            
        Note:
            - scale_range: 0.9-1.1 表示±10%的缩放
            - shift_range: -50~50 表示±50 HU的偏移
            - gamma_range: <1增强暗部，>1增强亮部
            - contrast_range: 调整对比度
        """
        super().__init__('IntensityAugment', prob, random_state)
        
        if operations is None:
            operations = ['scale', 'shift', 'gamma', 'contrast']
        
        self.operations = operations
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.gamma_range = gamma_range
        self.contrast_range = contrast_range
    
    def _apply_scale(self, img: np.ndarray, scale: float) -> np.ndarray:
        """线性强度缩放"""
        return img * scale
    
    def _apply_shift(self, img: np.ndarray, shift: float) -> np.ndarray:
        """整体强度偏移"""
        return img + shift
    
    def _apply_gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """
        Gamma校正
        
        先归一化到[0,1]，应用gamma，再反归一化
        """
        # 记录原始范围
        img_min, img_max = img.min(), img.max()
        
        if img_max - img_min < 1e-8:
            return img
        
        # 归一化
        img_norm = (img - img_min) / (img_max - img_min)
        
        # Gamma变换
        img_norm = np.power(img_norm, gamma)
        
        # 反归一化
        img = img_norm * (img_max - img_min) + img_min
        
        return img
    
    def _apply_contrast(self, img: np.ndarray, contrast: float) -> np.ndarray:
        """
        对比度调整
        
        公式：output = (input - mean) * contrast + mean
        """
        mean = img.mean()
        return (img - mean) * contrast + mean
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """
        应用强度变换
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            is_training: 是否为训练模式
            
        Returns:
            AugmentResult: 增强结果
            
        Note:
            只变换LR输入，HR目标保持不变
        """
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        lr_transformed = lr_img.copy()
        applied_ops = []
        params = {}
        
        # 按顺序应用启用的变换
        if 'scale' in self.operations:
            scale = random.uniform(*self.scale_range)
            lr_transformed = self._apply_scale(lr_transformed, scale)
            applied_ops.append('scale')
            params['scale'] = scale
        
        if 'shift' in self.operations:
            shift = random.uniform(*self.shift_range)
            lr_transformed = self._apply_shift(lr_transformed, shift)
            applied_ops.append('shift')
            params['shift'] = shift
        
        if 'gamma' in self.operations:
            gamma = random.uniform(*self.gamma_range)
            lr_transformed = self._apply_gamma(lr_transformed, gamma)
            applied_ops.append('gamma')
            params['gamma'] = gamma
        
        if 'contrast' in self.operations:
            contrast = random.uniform(*self.contrast_range)
            lr_transformed = self._apply_contrast(lr_transformed, contrast)
            applied_ops.append('contrast')
            params['contrast'] = contrast
        
        return AugmentResult(
            lr_img=lr_transformed,
            hr_img=hr_img,  # HR保持不变
            applied=len(applied_ops) > 0,
            aug_name=self.name,
            params={
                'operations': applied_ops,
                **params
            }
        )
    
    def get_config(self) -> dict:
        """获取配置"""
        config = super().get_config()
        config.update({
            'operations': self.operations,
            'scale_range': self.scale_range,
            'shift_range': self.shift_range,
            'gamma_range': self.gamma_range,
            'contrast_range': self.contrast_range,
        })
        return config


class RandomBlurAugment(BaseAugment):
    """
    随机模糊增强器
    
    模拟部分容积效应或轻微运动模糊。
    使用高斯滤波器。
    
    Example:
        >>> aug = RandomBlurAugment(prob=0.3, sigma_range=(0.3, 1.0))
    """
    
    def __init__(self, prob: float = 0.3, 
                 sigma_range: Tuple[float, float] = (0.3, 1.0),
                 random_state: Optional[int] = None):
        """
        初始化
        
        Args:
            prob: 应用概率
            sigma_range: 高斯核标准差范围
            random_state: 随机种子
        """
        super().__init__('RandomBlurAugment', prob, random_state)
        self.sigma_range = sigma_range
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """应用随机模糊"""
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        from scipy.ndimage import gaussian_filter
        
        sigma = random.uniform(*self.sigma_range)
        lr_blurred = gaussian_filter(lr_img, sigma=sigma)
        
        return AugmentResult(
            lr_img=lr_blurred,
            hr_img=hr_img,
            applied=True,
            aug_name=self.name,
            params={'sigma': sigma}
        )


class WindowingAugment(BaseAugment):
    """
    窗宽窗位增强器
    
    模拟放射科医生调整窗宽窗位观察不同组织。
    将CT值限制在特定窗宽窗位范围内。
    
    Example:
        >>> # 软组织窗
        >>> aug = WindowingAugment(prob=0.3, window_center=40, window_width=400)
        >>> 
        >>> # 肺窗
        >>> aug = WindowingAugment(prob=0.3, window_center=-600, window_width=1500)
    """
    
    # 常用窗宽窗位设置
    WINDOWS = {
        'lung': {'center': -600, 'width': 1500},
        'soft_tissue': {'center': 40, 'width': 400},
        'bone': {'center': 400, 'width': 1800},
        'brain': {'center': 35, 'width': 80},
        'mediastinum': {'center': 50, 'width': 350},
    }
    
    def __init__(self, prob: float = 0.3,
                 window_center: float = 40,
                 window_width: float = 400,
                 random_state: Optional[int] = None):
        """
        初始化
        
        Args:
            prob: 应用概率
            window_center: 窗位（WW）
            window_width: 窗宽（WL）
            random_state: 随机种子
        """
        super().__init__('WindowingAugment', prob, random_state)
        self.window_center = window_center
        self.window_width = window_width
    
    @classmethod
    def from_preset(cls, preset: str, prob: float = 0.3):
        """
        从预设创建窗宽窗位增强器
        
        Args:
            preset: 预设名称 ('lung', 'soft_tissue', 'bone', 'brain', 'mediastinum')
            prob: 应用概率
            
        Returns:
            WindowingAugment实例
        """
        if preset not in cls.WINDOWS:
            raise ValueError(f"Unknown preset '{preset}'. "
                           f"Available: {list(cls.WINDOWS.keys())}")
        
        window = cls.WINDOWS[preset]
        return cls(prob=prob, 
                  window_center=window['center'],
                  window_width=window['width'])
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """应用窗宽窗位变换"""
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        
        # 裁剪并归一化
        lr_windowed = np.clip((lr_img - min_val) / (max_val - min_val + 1e-8), 0, 1)
        
        return AugmentResult(
            lr_img=lr_windowed,
            hr_img=hr_img,
            applied=True,
            aug_name=self.name,
            params={
                'window_center': self.window_center,
                'window_width': self.window_width,
            }
        )
