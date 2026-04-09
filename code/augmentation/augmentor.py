"""
3D医学图像数据增强器
支持: 翻转、旋转、噪声、缩放等多种增强策略
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple


class Augmentor:
    """
    3D医学图像数据增强器
    
    支持以下增强策略:
    - 随机水平/垂直翻转
    - 随机旋转 (90度倍数)
    - 高斯噪声
    - 随机缩放
    - 随机裁剪
    """
    
    def __init__(self, 
                 enabled: bool = False,
                 flip_prob: float = 0.5,
                 noise_prob: float = 0.0,
                 noise_std: float = 0.01,
                 rotation_prob: float = 0.0,
                 max_rotation_angle: float = 15,
                 scale_prob: float = 0.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 elastic_prob: float = 0.0,
                 elastic_alpha: float = 10.0,
                 elastic_sigma: float = 3.0):
        """
        初始化增强器
        
        Args:
            enabled: 是否启用增强
            flip_prob: 翻转概率
            noise_prob: 噪声概率
            noise_std: 噪声标准差
            rotation_prob: 旋转概率
            max_rotation_angle: 最大旋转角度
            scale_prob: 缩放概率
            scale_range: 缩放范围
            elastic_prob: 弹性形变概率
            elastic_alpha: 弹性形变强度
            elastic_sigma: 弹性形变平滑度
        """
        self.enabled = enabled
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.rotation_prob = rotation_prob
        self.max_rotation_angle = max_rotation_angle
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Augmentor':
        """从配置字典创建增强器"""
        return cls(
            enabled=config.get('enabled', False),
            flip_prob=config.get('flip_prob', 0.5),
            noise_prob=config.get('noise_prob', 0.0),
            noise_std=config.get('noise_std', 0.01),
            rotation_prob=config.get('rotation_prob', 0.0),
            max_rotation_angle=config.get('max_rotation_angle', 15),
            scale_prob=config.get('scale_prob', 0.0),
            scale_range=config.get('scale_range', (0.9, 1.1)),
            elastic_prob=config.get('elastic_prob', 0.0),
            elastic_alpha=config.get('elastic_alpha', 10.0),
            elastic_sigma=config.get('elastic_sigma', 3.0),
        )
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入数据应用增强
        
        Args:
            x: 低分辨率输入 [B, C, D, H, W]
            y: 高分辨率目标 [B, C, D, H, W]
            
        Returns:
            增强后的 (x, y)
        """
        if not self.enabled:
            return x, y
        
        # 确保同步: 使用相同的随机种子
        seed = random.randint(0, 2**32 - 1)
        
        # 对 x 应用增强
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        x = self._augment_single(x)
        
        # 对 y 应用相同的增强
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        y = self._augment_single(y)
        
        return x, y
    
    def _augment_single(self, img: torch.Tensor) -> torch.Tensor:
        """对单张图像应用增强"""
        # 随机水平翻转 (W维度)
        if random.random() < self.flip_prob:
            img = torch.flip(img, [4])
        
        # 随机垂直翻转 (H维度)
        if random.random() < self.flip_prob:
            img = torch.flip(img, [3])
        
        # 随机深度翻转 (D维度)
        if random.random() < self.flip_prob:
            img = torch.flip(img, [2])
        
        # 随机旋转 (90度倍数，保持体素形状)
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])  # 90, 180, 270度
            # 在 H-W 平面旋转
            img = torch.rot90(img, k=k, dims=[3, 4])
        
        # 高斯噪声
        if random.random() < self.noise_prob:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
            img = torch.clamp(img, 0, 1)
        
        # 随机缩放
        if random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_range)
            img = self._random_scale(img, scale)
        
        return img
    
    def _random_scale(self, img: torch.Tensor, scale: float) -> torch.Tensor:
        """随机缩放图像"""
        B, C, D, H, W = img.shape
        new_H = int(H * scale)
        new_W = int(W * scale)
        new_D = int(D * scale)
        
        # 插值缩放
        img_scaled = F.interpolate(
            img, size=(new_D, new_H, new_W),
            mode='trilinear', align_corners=False
        )
        
        # 裁剪或填充回原尺寸
        if scale > 1.0:
            # 裁剪中心区域
            start_d = (new_D - D) // 2
            start_h = (new_H - H) // 2
            start_w = (new_W - W) // 2
            img = img_scaled[:, :, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]
        else:
            # 填充
            pad_d = (D - new_D) // 2
            pad_h = (H - new_H) // 2
            pad_w = (W - new_W) // 2
            img = F.pad(img_scaled, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
        
        return img
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            'enabled': self.enabled,
            'flip_prob': self.flip_prob,
            'noise_prob': self.noise_prob,
            'noise_std': self.noise_std,
            'rotation_prob': self.rotation_prob,
            'max_rotation_angle': self.max_rotation_angle,
            'scale_prob': self.scale_prob,
            'scale_range': self.scale_range,
        }
    
    def __repr__(self):
        return f"Augmentor(enabled={self.enabled}, flip={self.flip_prob}, noise={self.noise_prob})"


def get_augmentor_from_config(config_path: str) -> Augmentor:
    """
    从配置文件加载增强器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Augmentor 实例
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '' or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            try:
                v = eval(v)
            except:
                pass
            config[k] = v
    
    return Augmentor.from_config(config)


# 预定义的增强策略
AUGMENTATION_PRESETS = {
    'none': {
        'enabled': False,
    },
    'flip': {
        'enabled': True,
        'flip_prob': 0.5,
    },
    'light': {
        'enabled': True,
        'flip_prob': 0.3,
        'noise_prob': 0.1,
        'noise_std': 0.005,
    },
    'medium': {
        'enabled': True,
        'flip_prob': 0.5,
        'noise_prob': 0.2,
        'noise_std': 0.01,
        'rotation_prob': 0.1,
    },
    'heavy': {
        'enabled': True,
        'flip_prob': 0.5,
        'noise_prob': 0.3,
        'noise_std': 0.02,
        'rotation_prob': 0.2,
        'scale_prob': 0.1,
    },
}


def get_preset_augmentor(preset_name: str) -> Augmentor:
    """获取预定义增强策略"""
    if preset_name not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(AUGMENTATION_PRESETS.keys())}")
    
    return Augmentor.from_config(AUGMENTATION_PRESETS[preset_name])
