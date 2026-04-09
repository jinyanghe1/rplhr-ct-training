"""
翻转增强模块
Flip Augmentation Module

支持3D影像在三个轴向上的翻转：
- horizontal (H): 水平翻转，对应axis=2
- vertical (V): 垂直翻转，对应axis=1  
- depth (D): 深度翻转，对应axis=0

注意：CT影像的Z轴（层间顺序）通常不应翻转，
但本模块提供该选项以支持特殊需求。
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from .base_augment import BaseAugment, AugmentResult


class FlipAugment(BaseAugment):
    """
    3D翻转增强器
    
    支持水平、垂直、深度三个轴的随机翻转。
    翻转时LR和HR同步变换，保持空间对应关系。
    
    Attributes:
        prob: 应用概率
        axes: 可翻转的轴列表
        axis_map: 轴名称到数组维度的映射
    
    Example:
        >>> aug = FlipAugment(prob=0.5, axes=['horizontal', 'vertical'])
        >>> lr_aug, hr_aug = aug(lr_img, hr_img, is_training=True)
        
        >>> # 只水平翻转
        >>> aug = FlipAugment(prob=0.5, axes=['horizontal'])
    """
    
    # 轴名称到numpy轴编号的映射（对于3D数据 Z,H,W）
    AXIS_MAP = {
        'depth': 0,      # Z轴
        'vertical': 1,   # H轴（Y轴）
        'horizontal': 2, # W轴（X轴）
        'd': 0,
        'v': 1,
        'h': 2,
    }
    
    def __init__(self, prob: float = 0.5, 
                 axes: Optional[List[str]] = None,
                 random_state: Optional[int] = None):
        """
        初始化翻转增强器
        
        Args:
            prob: 应用概率 (0-1)
            axes: 可翻转的轴列表，可选值：
                  'horizontal'/'h', 'vertical'/'v', 'depth'/'d'
                  默认 ['horizontal', 'vertical']（不推荐翻转depth）
            random_state: 随机种子
            
        Raises:
            ValueError: 如果axes包含无效值
        """
        super().__init__('FlipAugment', prob, random_state)
        
        if axes is None:
            axes = ['horizontal', 'vertical']
        
        # 验证并转换轴名称
        self.axes = []
        for axis in axes:
            axis_lower = axis.lower()
            if axis_lower not in self.AXIS_MAP:
                raise ValueError(f"Invalid axis '{axis}'. "
                               f"Valid axes: {list(self.AXIS_MAP.keys())}")
            self.axes.append(self.AXIS_MAP[axis_lower])
        
        # 去重
        self.axes = list(set(self.axes))
        
        if not self.axes:
            raise ValueError("At least one axis must be specified")
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """
        应用翻转增强
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W)
            is_training: 是否为训练模式
            
        Returns:
            AugmentResult: 增强结果
        """
        # 检查是否应该应用增强
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img,
                hr_img=hr_img,
                applied=False,
                aug_name=self.name,
                params={'prob': self.prob, 'skipped': True}
            )
        
        # 随机选择一个轴进行翻转
        axis = random.choice(self.axes)
        
        # 获取轴名称用于记录
        axis_name = {v: k for k, v in self.AXIS_MAP.items()}[axis]
        
        # 对LR和HR同步翻转
        lr_flipped = np.flip(lr_img, axis=axis).copy()
        hr_flipped = np.flip(hr_img, axis=axis).copy()
        
        return AugmentResult(
            lr_img=lr_flipped,
            hr_img=hr_flipped,
            applied=True,
            aug_name=self.name,
            params={
                'axis': axis,
                'axis_name': axis_name,
                'prob': self.prob,
            }
        )
    
    def get_config(self) -> dict:
        """获取配置"""
        config = super().get_config()
        # 将数字轴转换回名称
        reverse_map = {0: 'depth', 1: 'vertical', 2: 'horizontal'}
        config['axes'] = [reverse_map[ax] for ax in self.axes]
        return config


class RandomFlip90Augment(BaseAugment):
    """
    90度随机旋转增强器（仅在XY平面）
    
    在XY平面内进行90度整数倍旋转（0°, 90°, 180°, 270°）。
    模拟不同扫描体位。
    
    Example:
        >>> aug = RandomFlip90Augment(prob=0.3)
        >>> lr_aug, hr_aug = aug(lr_img, hr_img, is_training=True)
    """
    
    def __init__(self, prob: float = 0.3, random_state: Optional[int] = None):
        """
        初始化
        
        Args:
            prob: 应用概率
            random_state: 随机种子
        """
        super().__init__('RandomFlip90Augment', prob, random_state)
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """应用90度旋转"""
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        # 随机选择旋转次数 (1=90°, 2=180°, 3=270°)
        k = random.randint(1, 3)
        
        # 在XY平面旋转 (axes 1,2)
        lr_rot = np.rot90(lr_img, k=k, axes=(1, 2)).copy()
        hr_rot = np.rot90(hr_img, k=k, axes=(1, 2)).copy()
        
        return AugmentResult(
            lr_img=lr_rot,
            hr_img=hr_rot,
            applied=True,
            aug_name=self.name,
            params={'rotation': k * 90, 'k': k}
        )
