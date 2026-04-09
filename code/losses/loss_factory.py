"""
Loss 工厂模块

提供统一的 Loss 创建接口，支持通过配置字符串或配置文件创建 Loss
"""

import os
import re
from typing import Dict, Optional, Union
import torch.nn as nn

# 导入所有 Loss 类
from .l1_loss import L1Loss, SmoothL1Loss
from .eagle3d_loss import EAGLELoss3D
from .charbonnier_loss import CharbonnierLoss
from .ssim_loss import SSIMLoss
from .combined_loss import CombinedLoss, L1SSIMLoss3D, WeightedLoss


class LossFactory:
    """
    Loss 函数工厂类
    
    根据配置创建对应的 Loss 函数实例
    
    Supported loss types:
        - "l1": L1Loss
        - "eagle3d": EAGLELoss3D
        - "charbonnier": CharbonnierLoss
        - "ssim": SSIMLoss
        - "l1_ssim": CombinedLoss (L1 + SSIM)
        - "smooth_l1": SmoothL1Loss
    
    Example:
        >>> # 创建 L1 Loss
        >>> loss_fn = LossFactory.create_loss('l1')
        
        >>> # 创建带参数的 EAGLE3D Loss
        >>> loss_fn = LossFactory.create_loss('eagle3d', alpha=0.2)
        
        >>> # 从配置字典创建
        >>> config = {'loss_type': 'charbonnier', 'eps': 1e-3}
        >>> loss_fn = LossFactory.create_loss_from_config(config)
        
        >>> # 从配置文件创建
        >>> loss_fn = LossFactory.create_loss_from_file('config/loss.txt')
    """
    
    # Loss 类型到类的映射
    LOSS_REGISTRY: Dict[str, type] = {
        'l1': L1Loss,
        'eagle3d': EAGLELoss3D,
        'charbonnier': CharbonnierLoss,
        'ssim': SSIMLoss,
        'l1_ssim': CombinedLoss,
        'l1ssim': CombinedLoss,
        'smooth_l1': SmoothL1Loss,
    }
    
    @classmethod
    def register_loss(cls, name: str, loss_class: type):
        """
        注册新的 Loss 类型
        
        Args:
            name: Loss 类型名称
            loss_class: Loss 类
        """
        cls.LOSS_REGISTRY[name] = loss_class
    
    @classmethod
    def create_loss(cls, loss_type: str, **kwargs) -> nn.Module:
        """
        根据类型创建 Loss 实例
        
        Args:
            loss_type: Loss 类型名称
            **kwargs: 传递给 Loss 构造函数的参数
        
        Returns:
            Loss 实例
        
        Raises:
            ValueError: 如果 loss_type 不被支持
        """
        loss_type = loss_type.lower().strip()
        
        if loss_type not in cls.LOSS_REGISTRY:
            raise ValueError(
                f"不支持的 loss 类型: '{loss_type}'. "
                f"支持的类型: {list(cls.LOSS_REGISTRY.keys())}"
            )
        
        loss_class = cls.LOSS_REGISTRY[loss_type]
        return loss_class(**kwargs)
    
    @classmethod
    def create_loss_from_config(cls, config: Dict) -> nn.Module:
        """
        从配置字典创建 Loss 实例
        
        Args:
            config: 配置字典，必须包含 'loss_type' 键
        
        Returns:
            Loss 实例
        """
        if 'loss_type' not in config:
            raise ValueError("配置字典必须包含 'loss_type' 键")
        
        loss_type = config['loss_type']
        kwargs = {k: v for k, v in config.items() if k != 'loss_type'}
        
        return cls.create_loss(loss_type, **kwargs)
    
    @classmethod
    def create_loss_from_file(cls, config_path: str) -> nn.Module:
        """
        从配置文件创建 Loss 实例
        
        配置文件格式:
            loss_type=l1
            reduction=mean
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            Loss 实例
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        config = cls._parse_config_file(config_path)
        return cls.create_loss_from_config(config)
    
    @classmethod
    def _parse_config_file(cls, config_path: str) -> Dict:
        """
        解析配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置字典
        """
        config = {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                
                # 解析键值对
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试转换为合适的类型
                    value = cls._convert_value(value)
                    config[key] = value
        
        return config
    
    @classmethod
    def _convert_value(cls, value: str) -> Union[str, int, float, bool]:
        """
        将字符串值转换为适当的类型
        """
        # 布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 尝试转换为整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试转换为浮点数
        try:
            # 处理科学计数法 (如 1e-6)
            return float(value)
        except ValueError:
            pass
        
        # 保持为字符串
        return value
    
    @classmethod
    def list_supported_losses(cls) -> list:
        """
        获取所有支持的 Loss 类型列表
        """
        return list(cls.LOSS_REGISTRY.keys())


# 便捷函数
def get_loss(config_source: Union[str, Dict]) -> nn.Module:
    """
    创建 Loss 函数的便捷函数
    
    Args:
        config_source: 配置源，可以是:
            - Loss 类型名称 (如 'l1', 'eagle3d')
            - 配置文件路径 (如 'config/loss.txt')
            - 配置字典 (如 {'loss_type': 'l1', 'reduction': 'mean'})
    
    Returns:
        Loss 实例
    
    Example:
        >>> # 通过类型名称
        >>> loss_fn = get_loss('l1')
        
        >>> # 通过配置文件
        >>> loss_fn = get_loss('config/loss_configs/loss_l1.txt')
        
        >>> # 通过配置字典
        >>> loss_fn = get_loss({'loss_type': 'eagle3d', 'alpha': 0.2})
    """
    if isinstance(config_source, str):
        # 判断是配置文件路径还是类型名称
        if os.path.exists(config_source):
            return LossFactory.create_loss_from_file(config_source)
        else:
            return LossFactory.create_loss(config_source)
    elif isinstance(config_source, dict):
        return LossFactory.create_loss_from_config(config_source)
    else:
        raise ValueError(f"不支持配置源类型: {type(config_source)}")
