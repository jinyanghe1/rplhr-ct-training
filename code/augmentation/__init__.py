"""
模块化数据增强系统 - Modular Data Augmentation System

本模块提供基于配置的数据增强功能，支持多种增强策略的灵活组合。

使用示例:
    >>> from augmentation import AugmentFactory
    >>> # 通过配置创建增强器
    >>> config = {
    ...     'use_augmentation': True,
    ...     'augment_types': ['flip', 'noise'],
    ...     'augment_probability': 0.5,
    ...     'flip_axis': ['horizontal', 'vertical'],
    ...     'noise_type': 'gaussian',
    ...     'noise_sigma': 0.01,
    ... }
    >>> augmenter = AugmentFactory.create(config)
    >>> 
    >>> # 在Dataset中使用
    >>> lr_img, hr_img = augmenter(lr_img, hr_img, is_training=True)

模块结构:
    - base_augment.py: 基础增强抽象类
    - flip_augment.py: 翻转增强（H/V/D 三轴）
    - noise_augment.py: 随机噪声（泊松+高斯）
    - elastic_augment.py: 3D弹性形变
    - intensity_augment.py: 强度变换
    - augment_factory.py: 增强器工厂类
    - augment_pipeline.py: 多增强组合管道

Author: Auto-generated
Date: 2026-04-03
"""

from .base_augment import BaseAugment, AugmentResult
from .flip_augment import FlipAugment
from .noise_augment import NoiseAugment
from .elastic_augment import ElasticAugment
from .intensity_augment import IntensityAugment
from .augment_factory import AugmentFactory
from .augment_pipeline import AugmentPipeline

# 兼容模块化配置系统的增强器
try:
    from .augmentor import Augmentor, get_augmentor_from_config, get_preset_augmentor
    MODULAR_AUGMENT_AVAILABLE = True
except ImportError:
    MODULAR_AUGMENT_AVAILABLE = False

__version__ = '1.0.0'
__all__ = [
    # 基础类
    'BaseAugment',
    'AugmentResult',
    
    # 具体增强器
    'FlipAugment',
    'NoiseAugment', 
    'ElasticAugment',
    'IntensityAugment',
    
    # 工厂和管道
    'AugmentFactory',
    'AugmentPipeline',
]

# 添加模块化增强器到导出列表
if MODULAR_AUGMENT_AVAILABLE:
    __all__.extend([
        'Augmentor',
        'get_augmentor_from_config',
        'get_preset_augmentor',
    ])


def create_augmenter(config: dict):
    """
    便捷函数：根据配置创建增强器
    
    Args:
        config: 配置字典，包含use_augmentation, augment_types等参数
        
    Returns:
        增强器实例（AugmentPipeline或BaseAugment子类）
        
    Example:
        >>> config = {
        ...     'use_augmentation': True,
        ...     'augment_types': ['flip', 'noise'],
        ...     'augment_probability': 0.5,
        ... }
        >>> augmenter = create_augmenter(config)
    """
    return AugmentFactory.create(config)


def get_available_augmenters():
    """
    获取所有可用的增强器类型列表
    
    Returns:
        list: 增强器名称列表
    """
    return ['flip', 'noise', 'elastic', 'intensity']


def get_default_config(aug_type: str = 'conservative'):
    """
    获取默认配置
    
    Args:
        aug_type: 配置类型 ('conservative', 'aggressive', 'noise_only', 'flip_only')
        
    Returns:
        dict: 配置字典
    """
    configs = {
        'conservative': {
            'use_augmentation': True,
            'augment_types': ['flip', 'noise'],
            'augment_probability': 0.5,
            'flip_axis': ['horizontal', 'vertical'],
            'noise_type': 'gaussian',
            'noise_sigma': 0.01,
        },
        'aggressive': {
            'use_augmentation': True,
            'augment_types': ['flip', 'noise', 'elastic', 'intensity'],
            'augment_probability': 0.7,
            'flip_axis': ['horizontal', 'vertical', 'depth'],
            'noise_type': 'both',
            'noise_sigma': 0.02,
            'elastic_alpha': 10.0,
            'elastic_sigma': 3.0,
        },
        'noise_only': {
            'use_augmentation': True,
            'augment_types': ['noise'],
            'augment_probability': 0.8,
            'noise_type': 'both',
            'noise_sigma': 0.015,
        },
        'flip_only': {
            'use_augmentation': True,
            'augment_types': ['flip'],
            'augment_probability': 0.5,
            'flip_axis': ['horizontal', 'vertical', 'depth'],
        },
    }
    return configs.get(aug_type, configs['conservative'])
