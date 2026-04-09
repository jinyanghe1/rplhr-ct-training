"""
增强管道模块
Augmentation Pipeline Module

支持多种增强的组合应用，提供顺序和随机两种应用模式。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_augment import BaseAugment, AugmentResult


class AugmentPipeline(BaseAugment):
    """
    多增强组合管道
    
    将多个增强器组合成一个管道，支持顺序或随机应用。
    
    Attributes:
        augmenters: 增强器列表
        mode: 应用模式 ('sequential' 或 'random')
        max_augs: 随机模式下最多应用的增强数
    
    Example:
        >>> # 顺序应用所有增强
        >>> pipeline = AugmentPipeline([
        ...     FlipAugment(prob=0.5),
        ...     NoiseAugment(prob=0.5),
        ...     ElasticAugment(prob=0.1),
        ... ], mode='sequential')
        >>> 
        >>> # 随机选择1-2个增强应用
        >>> pipeline = AugmentPipeline([
        ...     FlipAugment(prob=1.0),
        ...     NoiseAugment(prob=1.0),
        ...     IntensityAugment(prob=1.0),
        ... ], mode='random', max_augs=2)
        >>> 
        >>> lr_aug, hr_aug = pipeline(lr_img, hr_img, is_training=True)
    """
    
    def __init__(self, 
                 augmenters: List[BaseAugment],
                 mode: str = 'sequential',
                 max_augs: Optional[int] = None,
                 random_state: Optional[int] = None):
        """
        初始化增强管道
        
        Args:
            augmenters: 增强器列表
            mode: 应用模式
                   'sequential': 按顺序应用所有增强
                   'random': 随机选择部分增强应用
            max_augs: 随机模式下最多应用的增强数，默认全部
            random_state: 随机种子
            
        Raises:
            ValueError: 如果mode无效
        """
        super().__init__('AugmentPipeline', prob=1.0, random_state=random_state)
        
        if mode not in ['sequential', 'random']:
            raise ValueError(f"Invalid mode '{mode}'. "
                           f"Valid modes: 'sequential', 'random'")
        
        self.augmenters = augmenters
        self.mode = mode
        self.max_augs = max_augs or len(augmenters)
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """
        应用增强管道
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            is_training: 是否为训练模式
            
        Returns:
            AugmentResult: 增强结果
        """
        if not is_training:
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': 'not_training'}
            )
        
        lr_current = lr_img.copy()
        hr_current = hr_img.copy()
        
        if self.mode == 'sequential':
            # 顺序应用所有增强
            applied_augs = []
            for aug in self.augmenters:
                result = aug.apply(lr_current, hr_current, is_training=True)
                lr_current = result.lr_img
                hr_current = result.hr_img
                if result.applied:
                    applied_augs.append(aug.name)
            
            return AugmentResult(
                lr_img=lr_current,
                hr_img=hr_current,
                applied=len(applied_augs) > 0,
                aug_name=self.name,
                params={
                    'mode': 'sequential',
                    'applied_augs': applied_augs,
                    'total_augs': len(self.augmenters),
                }
            )
        
        else:  # random mode
            # 随机选择要应用的增强
            import random
            num_to_apply = random.randint(1, min(self.max_augs, len(self.augmenters)))
            selected_augs = random.sample(self.augmenters, num_to_apply)
            
            applied_augs = []
            for aug in selected_augs:
                result = aug.apply(lr_current, hr_current, is_training=True)
                lr_current = result.lr_img
                hr_current = result.hr_img
                if result.applied:
                    applied_augs.append(aug.name)
            
            return AugmentResult(
                lr_img=lr_current,
                hr_img=hr_current,
                applied=len(applied_augs) > 0,
                aug_name=self.name,
                params={
                    'mode': 'random',
                    'selected_augs': [aug.name for aug in selected_augs],
                    'applied_augs': applied_augs,
                }
            )
    
    def __len__(self) -> int:
        """返回增强器数量"""
        return len(self.augmenters)
    
    def __iter__(self):
        """迭代增强器"""
        return iter(self.augmenters)
    
    def add(self, augmenter: BaseAugment):
        """
        添加增强器到管道
        
        Args:
            augmenter: 要添加的增强器
        """
        self.augmenters.append(augmenter)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'max_augs': self.max_augs,
            'augmenters': [aug.get_config() for aug in self.augmenters],
        })
        return config


class ProbabilisticPipeline(BaseAugment):
    """
    概率性增强管道
    
    为每个增强独立决定是否应用，每个增强有自己的概率。
    与AugmentPipeline的区别：每个增强独立判断，而不是整体判断。
    
    Example:
        >>> pipeline = ProbabilisticPipeline([
        ...     FlipAugment(prob=0.5),      # 50%概率
        ...     NoiseAugment(prob=0.3),     # 30%概率
        ...     ElasticAugment(prob=0.1),   # 10%概率
        ... ])
    """
    
    def __init__(self, augmenters: List[BaseAugment],
                 random_state: Optional[int] = None):
        """
        初始化
        
        Args:
            augmenters: 增强器列表（每个有自己的prob）
            random_state: 随机种子
        """
        super().__init__('ProbabilisticPipeline', prob=1.0, 
                        random_state=random_state)
        self.augmenters = augmenters
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """应用概率性增强管道"""
        if not is_training:
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': 'not_training'}
            )
        
        lr_current = lr_img.copy()
        hr_current = hr_img.copy()
        applied_augs = []
        
        for aug in self.augmenters:
            result = aug.apply(lr_current, hr_current, is_training=True)
            lr_current = result.lr_img
            hr_current = result.hr_img
            if result.applied:
                applied_augs.append(aug.name)
        
        return AugmentResult(
            lr_img=lr_current,
            hr_img=hr_current,
            applied=len(applied_augs) > 0,
            aug_name=self.name,
            params={'applied_augs': applied_augs}
        )
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        config = super().get_config()
        config['augmenters'] = [aug.get_config() for aug in self.augmenters]
        return config


class Compose:
    """
    简单的增强组合类（函数式接口）
    
    与AugmentPipeline类似，但更简洁，直接返回变换后的图像。
    
    Example:
        >>> transform = Compose([
        ...     FlipAugment(prob=0.5),
        ...     NoiseAugment(prob=0.5),
        ... ])
        >>> lr_aug, hr_aug = transform(lr_img, hr_img, is_training=True)
    """
    
    def __init__(self, transforms: List[BaseAugment]):
        """
        初始化
        
        Args:
            transforms: 增强器列表
        """
        self.transforms = transforms
    
    def __call__(self, lr_img: np.ndarray, hr_img: np.ndarray,
                 is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用所有变换
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            is_training: 是否为训练模式
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 变换后的图像
        """
        if not is_training:
            return lr_img, hr_img
        
        for transform in self.transforms:
            lr_img, hr_img = transform(lr_img, hr_img, is_training=True)
        
        return lr_img, hr_img
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '(['
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n])'
        return format_string
