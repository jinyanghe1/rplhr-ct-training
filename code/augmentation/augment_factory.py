"""
增强器工厂模块
Augmenter Factory Module

根据配置动态创建增强器实例。
支持从配置字典、配置文件创建增强器。
"""

import os
from typing import Dict, Any, Optional, List
from .base_augment import BaseAugment, IdentityAugment
from .flip_augment import FlipAugment, RandomFlip90Augment
from .noise_augment import NoiseAugment, SpeckleNoiseAugment
from .elastic_augment import ElasticAugment, ElasticAugment2D
from .intensity_augment import IntensityAugment, RandomBlurAugment, WindowingAugment
from .augment_pipeline import AugmentPipeline


class AugmentFactory:
    """
    增强器工厂类
    
    根据配置创建对应的增强器实例。
    支持单增强器和多增强器组合管道。
    
    Example:
        >>> # 从配置字典创建
        >>> config = {
        ...     'use_augmentation': True,
        ...     'augment_types': ['flip', 'noise'],
        ...     'augment_probability': 0.5,
        ... }
        >>> augmenter = AugmentFactory.create(config)
        >>> 
        >>> # 创建单个增强器
        >>> flip_aug = AugmentFactory.create_flip(prob=0.5, axes=['horizontal'])
        >>> 
        >>> # 从配置文件创建
        >>> augmenter = AugmentFactory.from_config_file('config.txt')
    """
    
    # 增强器类型到创建函数的映射
    _AUGMENT_REGISTRY = {
        'flip': FlipAugment,
        'noise': NoiseAugment,
        'elastic': ElasticAugment,
        'elastic2d': ElasticAugment2D,
        'intensity': IntensityAugment,
        'speckle': SpeckleNoiseAugment,
        'blur': RandomBlurAugment,
        'windowing': WindowingAugment,
        'rotate90': RandomFlip90Augment,
    }
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseAugment:
        """
        根据配置创建增强器
        
        Args:
            config: 配置字典，包含：
                - use_augmentation: 是否启用增强
                - augment_types: 增强类型列表
                - augment_probability: 基础概率
                - 各增强的详细参数
                
        Returns:
            BaseAugment: 增强器实例
            
        Example:
            >>> config = {
            ...     'use_augmentation': True,
            ...     'augment_types': ['flip', 'noise'],
            ...     'augment_probability': 0.5,
            ...     'flip_axis': ['horizontal', 'vertical'],
            ...     'noise_type': 'gaussian',
            ...     'noise_sigma': 0.01,
            ... }
            >>> augmenter = AugmentFactory.create(config)
        """
        # 检查是否启用增强
        if not config.get('use_augmentation', False):
            return IdentityAugment()
        
        # 获取增强类型列表
        augment_types = config.get('augment_types', [])
        
        if not augment_types:
            return IdentityAugment()
        
        # 获取基础概率
        base_prob = config.get('augment_probability', 0.5)
        
        # 如果只有一个增强类型，直接创建
        if len(augment_types) == 1:
            return cls._create_single_augment(
                augment_types[0], config, base_prob
            )
        
        # 多个增强类型，创建管道
        augmenters = []
        for aug_type in augment_types:
            aug = cls._create_single_augment(aug_type, config, base_prob)
            if not isinstance(aug, IdentityAugment):
                augmenters.append(aug)
        
        if not augmenters:
            return IdentityAugment()
        
        return AugmentPipeline(augmenters)
    
    @classmethod
    def _create_single_augment(cls, aug_type: str, 
                               config: Dict[str, Any],
                               base_prob: float) -> BaseAugment:
        """
        创建单个增强器
        
        Args:
            aug_type: 增强类型
            config: 配置字典
            base_prob: 基础概率
            
        Returns:
            BaseAugment: 增强器实例
        """
        aug_type = aug_type.lower()
        
        if aug_type not in cls._AUGMENT_REGISTRY:
            raise ValueError(f"Unknown augment type '{aug_type}'. "
                           f"Available types: {list(cls._AUGMENT_REGISTRY.keys())}")
        
        aug_class = cls._AUGMENT_REGISTRY[aug_type]
        
        # 根据类型提取参数
        kwargs = {'prob': base_prob}
        
        if aug_type == 'flip':
            kwargs['axes'] = config.get('flip_axis', ['horizontal', 'vertical'])
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'noise':
            kwargs['noise_type'] = config.get('noise_type', 'gaussian')
            kwargs['sigma'] = config.get('noise_sigma', 0.01)
            kwargs['scale'] = config.get('noise_scale', 1.0)
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'elastic':
            kwargs['alpha'] = config.get('elastic_alpha', 10.0)
            kwargs['sigma'] = config.get('elastic_sigma', 3.0)
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'elastic2d':
            kwargs['alpha'] = config.get('elastic_alpha', 10.0)
            kwargs['sigma'] = config.get('elastic_sigma', 3.0)
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'intensity':
            kwargs['operations'] = config.get('intensity_operations', 
                                            ['scale', 'shift', 'gamma', 'contrast'])
            kwargs['scale_range'] = config.get('intensity_scale_range', (0.9, 1.1))
            kwargs['shift_range'] = config.get('intensity_shift_range', (-50, 50))
            kwargs['gamma_range'] = config.get('intensity_gamma_range', (0.9, 1.1))
            kwargs['contrast_range'] = config.get('intensity_contrast_range', (0.9, 1.1))
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'speckle':
            kwargs['var'] = config.get('speckle_var', 0.01)
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'blur':
            kwargs['sigma_range'] = config.get('blur_sigma_range', (0.3, 1.0))
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'windowing':
            kwargs['window_center'] = config.get('window_center', 40)
            kwargs['window_width'] = config.get('window_width', 400)
            kwargs['random_state'] = config.get('random_state')
            
        elif aug_type == 'rotate90':
            kwargs['random_state'] = config.get('random_state')
        
        return aug_class(**kwargs)
    
    @classmethod
    def from_config_file(cls, config_path: str) -> BaseAugment:
        """
        从配置文件创建增强器
        
        配置文件格式与主项目的config.txt相同：
        key = value
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            BaseAugment: 增强器实例
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('*'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        # 尝试解析为Python对象
                        value = eval(value)
                    except:
                        # 保持为字符串
                        pass
                    config[key] = value
        
        return cls.create(config)
    
    @classmethod
    def create_flip(cls, prob: float = 0.5, 
                    axes: Optional[List[str]] = None,
                    random_state: Optional[int] = None) -> FlipAugment:
        """创建翻转增强器"""
        return FlipAugment(prob=prob, axes=axes, random_state=random_state)
    
    @classmethod
    def create_noise(cls, prob: float = 0.5,
                     noise_type: str = 'gaussian',
                     sigma: float = 0.01,
                     random_state: Optional[int] = None) -> NoiseAugment:
        """创建噪声增强器"""
        return NoiseAugment(prob=prob, noise_type=noise_type, 
                           sigma=sigma, random_state=random_state)
    
    @classmethod
    def create_elastic(cls, prob: float = 0.1,
                       alpha: float = 10.0,
                       sigma: float = 3.0,
                       random_state: Optional[int] = None) -> ElasticAugment:
        """创建弹性形变增强器"""
        return ElasticAugment(prob=prob, alpha=alpha, sigma=sigma,
                             random_state=random_state)
    
    @classmethod
    def create_intensity(cls, prob: float = 0.5,
                         operations: Optional[List[str]] = None,
                         random_state: Optional[int] = None) -> IntensityAugment:
        """创建强度变换增强器"""
        return IntensityAugment(prob=prob, operations=operations,
                               random_state=random_state)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取所有可用的增强器类型"""
        return list(cls._AUGMENT_REGISTRY.keys())


def create_augmenter_from_config(config: Dict[str, Any]) -> BaseAugment:
    """
    便捷函数：从配置创建增强器
    
    Args:
        config: 配置字典
        
    Returns:
        BaseAugment: 增强器实例
    """
    return AugmentFactory.create(config)
