"""
基础数据增强抽象类
Base Data Augmentation Abstract Class

定义所有增强器的统一接口和公共功能。
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AugmentResult:
    """增强结果数据类"""
    lr_img: np.ndarray
    hr_img: np.ndarray
    applied: bool  # 是否应用了增强
    aug_name: str  # 增强器名称
    params: Dict[str, Any]  # 应用的参数


class BaseAugment(ABC):
    """
    数据增强基类
    
    所有具体增强器的抽象基类，定义统一接口。
    
    Attributes:
        prob: 增强应用概率 (0-1)
        random_state: 随机种子
        name: 增强器名称
    
    Example:
        >>> class MyAugment(BaseAugment):
        ...     def __init__(self, prob=0.5, **kwargs):
        ...         super().__init__('MyAugment', prob, **kwargs)
        ...     
        ...     def apply(self, lr_img, hr_img, is_training=True):
        ...         # 实现增强逻辑
        ...         return AugmentResult(lr_img, hr_img, True, self.name, {})
    """
    
    def __init__(self, name: str, prob: float = 0.5, random_state: Optional[int] = None):
        """
        初始化增强器
        
        Args:
            name: 增强器名称
            prob: 应用概率 (0-1)
            random_state: 随机种子，用于结果复现
        """
        self.name = name
        self.prob = prob
        self.random_state = random_state
        
        if random_state is not None:
            self.set_random_state(random_state)
    
    def set_random_state(self, seed: int):
        """
        设置随机种子
        
        Args:
            seed: 随机种子
        """
        self.random_state = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def should_apply(self, is_training: bool = True) -> bool:
        """
        判断是否应用增强
        
        增强只在训练时应用，验证/测试时不应用。
        
        Args:
            is_training: 是否为训练模式
            
        Returns:
            bool: 是否应用增强
        """
        if not is_training:
            return False
        return random.random() < self.prob
    
    @abstractmethod
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray, 
              is_training: bool = True) -> AugmentResult:
        """
        应用增强（子类必须实现）
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W) 或 (C, Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W) 或 (C, Z', H, W)
            is_training: 是否为训练模式
            
        Returns:
            AugmentResult: 增强结果
        """
        pass
    
    def __call__(self, lr_img: np.ndarray, hr_img: np.ndarray,
                 is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        调用增强器（便捷接口）
        
        Args:
            lr_img: 低分辨率输入
            hr_img: 高分辨率目标
            is_training: 是否为训练模式
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 增强后的 (lr_img, hr_img)
        """
        result = self.apply(lr_img, hr_img, is_training)
        return result.lr_img, result.hr_img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', prob={self.prob})"
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置字典
        
        Returns:
            dict: 配置信息
        """
        return {
            'name': self.name,
            'prob': self.prob,
            'random_state': self.random_state,
        }


class IdentityAugment(BaseAugment):
    """
    恒等增强器（不做任何操作）
    
    用于禁用增强或作为占位符。
    """
    
    def __init__(self):
        super().__init__('Identity', prob=0.0)
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """直接返回原图"""
        return AugmentResult(
            lr_img=lr_img,
            hr_img=hr_img,
            applied=False,
            aug_name=self.name,
            params={}
        )
