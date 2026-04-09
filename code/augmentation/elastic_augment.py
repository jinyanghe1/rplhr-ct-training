"""
3D弹性形变增强模块
3D Elastic Deformation Augmentation Module

模拟软组织形变，生成平滑的非线性变形场。
使用随机位移场 + 高斯滤波平滑实现。

注意：这是计算密集型操作，建议适当降低应用概率。
"""

import numpy as np
import random
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter, map_coordinates
from .base_augment import BaseAugment, AugmentResult


class ElasticAugment(BaseAugment):
    """
    3D弹性形变增强器
    
    模拟软组织的非线性形变，使用随机位移场 + 高斯平滑。
    对LR和HR应用同步的空间变换。
    
    Attributes:
        prob: 应用概率（建议较低，如0.1-0.3）
        alpha: 变形幅度（控制变形强度）
        sigma: 平滑程度（控制变形平滑度）
    
    Example:
        >>> # 轻度形变
        >>> aug = ElasticAugment(prob=0.1, alpha=10, sigma=3)
        >>> 
        >>> # 较强形变
        >>> aug = ElasticAugment(prob=0.1, alpha=20, sigma=4)
        >>> 
        >>> lr_aug, hr_aug = aug(lr_img, hr_img, is_training=True)
    
    Reference:
        Simard et al., "Best Practices for Convolutional Neural Networks 
        applied to Visual Document Analysis", ICDAR 2003
    """
    
    def __init__(self, 
                 prob: float = 0.1,
                 alpha: float = 10.0,
                 sigma: float = 3.0,
                 random_state: Optional[int] = None):
        """
        初始化弹性形变增强器
        
        Args:
            prob: 应用概率（建议0.1-0.3，计算量大）
            alpha: 变形幅度，值越大变形越大
            sigma: 平滑程度，值越大变形越平滑
            random_state: 随机种子
            
        Note:
            alpha和sigma的比值建议约为2-5：
            - alpha=10, sigma=3: 轻度平滑形变
            - alpha=20, sigma=4: 中度形变
            - alpha=30, sigma=5: 较强形变
        """
        super().__init__('ElasticAugment', prob, random_state)
        self.alpha = alpha
        self.sigma = sigma
    
    def _generate_displacement_field(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成随机位移场
        
        Args:
            shape: 图像形状 (Z, H, W)
            
        Returns:
            Tuple of (dx, dy, dz): 三个方向的位移场
        """
        # 生成随机位移场
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        
        # Z轴变形通常较小（层间连续性强）
        dz = dz * 0.3
        
        return dx, dy, dz
    
    def _apply_deformation(self, img: np.ndarray, 
                          dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        """
        应用形变到图像
        
        Args:
            img: 输入图像
            dx, dy, dz: 位移场
            
        Returns:
            形变后的图像
        """
        shape = img.shape
        
        # 创建网格
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # 应用形变
        indices = (
            np.clip(z + dz, 0, shape[0] - 1).reshape(-1),
            np.clip(y + dy, 0, shape[1] - 1).reshape(-1),
            np.clip(x + dx, 0, shape[2] - 1).reshape(-1)
        )
        
        deformed = map_coordinates(img, indices, order=1, mode='nearest').reshape(shape)
        
        return deformed
    
    def _repeat_displacement_for_hr(self, disp: np.ndarray, 
                                    lr_shape: Tuple, hr_shape: Tuple) -> np.ndarray:
        """
        将LR空间的位移场扩展到HR空间
        
        由于HR在Z方向通常是LR的ratio倍，需要将位移场在Z方向重复。
        
        Args:
            disp: LR空间的位移场
            lr_shape: LR图像形状
            hr_shape: HR图像形状
            
        Returns:
            HR空间的位移场
        """
        z_ratio = hr_shape[0] // lr_shape[0]
        
        # 在Z方向重复
        disp_hr = np.repeat(disp, z_ratio, axis=0)
        
        # 如果尺寸不完全匹配，裁剪或填充
        if disp_hr.shape[0] > hr_shape[0]:
            disp_hr = disp_hr[:hr_shape[0]]
        elif disp_hr.shape[0] < hr_shape[0]:
            pad_size = hr_shape[0] - disp_hr.shape[0]
            disp_hr = np.pad(disp_hr, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
        
        return disp_hr
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """
        应用弹性形变
        
        Args:
            lr_img: 低分辨率输入 (Z, H, W)
            hr_img: 高分辨率目标 (Z', H, W)
            is_training: 是否为训练模式
            
        Returns:
            AugmentResult: 增强结果
        """
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        lr_shape = lr_img.shape
        hr_shape = hr_img.shape
        
        # 生成LR空间的位移场
        dx, dy, dz = self._generate_displacement_field(lr_shape)
        
        # 对LR应用形变
        lr_deformed = self._apply_deformation(lr_img, dx, dy, dz)
        
        # 对HR应用形变（需要按比例扩展位移场）
        dx_hr = self._repeat_displacement_for_hr(dx, lr_shape, hr_shape)
        dy_hr = self._repeat_displacement_for_hr(dy, lr_shape, hr_shape)
        dz_hr = self._repeat_displacement_for_hr(dz, lr_shape, hr_shape)
        
        hr_deformed = self._apply_deformation(hr_img, dx_hr, dy_hr, dz_hr)
        
        return AugmentResult(
            lr_img=lr_deformed,
            hr_img=hr_deformed,
            applied=True,
            aug_name=self.name,
            params={
                'alpha': self.alpha,
                'sigma': self.sigma,
                'lr_shape': lr_shape,
                'hr_shape': hr_shape,
            }
        )
    
    def get_config(self) -> dict:
        """获取配置"""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'sigma': self.sigma,
        })
        return config


class ElasticAugment2D(BaseAugment):
    """
    2D弹性形变增强器（逐层应用）
    
    在XY平面对每一层独立应用弹性形变，Z方向不变形。
    计算效率更高，适合大多数CT增强场景。
    
    Example:
        >>> aug = ElasticAugment2D(prob=0.1, alpha=10, sigma=3)
    """
    
    def __init__(self,
                 prob: float = 0.1,
                 alpha: float = 10.0,
                 sigma: float = 3.0,
                 random_state: Optional[int] = None):
        """
        初始化
        
        Args:
            prob: 应用概率
            alpha: 变形幅度
            sigma: 平滑程度
            random_state: 随机种子
        """
        super().__init__('ElasticAugment2D', prob, random_state)
        self.alpha = alpha
        self.sigma = sigma
    
    def _apply_2d_elastic(self, img_slice: np.ndarray) -> np.ndarray:
        """对单层2D图像应用弹性形变"""
        shape = img_slice.shape
        
        # 生成2D位移场
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        
        # 创建网格
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        
        # 应用形变
        indices = (
            np.clip(y + dy, 0, shape[0] - 1).reshape(-1),
            np.clip(x + dx, 0, shape[1] - 1).reshape(-1)
        )
        
        deformed = map_coordinates(img_slice, indices, order=1, mode='nearest').reshape(shape)
        return deformed
    
    def apply(self, lr_img: np.ndarray, hr_img: np.ndarray,
              is_training: bool = True) -> AugmentResult:
        """应用2D弹性形变"""
        if not self.should_apply(is_training):
            return AugmentResult(
                lr_img=lr_img, hr_img=hr_img,
                applied=False, aug_name=self.name,
                params={'skipped': True}
            )
        
        # 对LR的每一层应用
        lr_deformed = np.zeros_like(lr_img)
        for z in range(lr_img.shape[0]):
            lr_deformed[z] = self._apply_2d_elastic(lr_img[z])
        
        # 对HR的每一层应用（注意：HR的层数是LR的ratio倍）
        hr_deformed = np.zeros_like(hr_img)
        z_ratio = hr_img.shape[0] // lr_img.shape[0]
        
        for z in range(hr_img.shape[0]):
            # 找到对应的LR层
            lr_z = z // z_ratio
            # 使用相同的随机种子确保对应层的形变一致
            np.random.seed(self.random_state or 0 + z)
            hr_deformed[z] = self._apply_2d_elastic(hr_img[z])
        
        return AugmentResult(
            lr_img=lr_deformed,
            hr_img=hr_deformed,
            applied=True,
            aug_name=self.name,
            params={
                'alpha': self.alpha,
                'sigma': self.sigma,
                'mode': '2d_per_slice',
            }
        )
