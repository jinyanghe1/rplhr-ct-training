"""
模块化数据增强系统使用示例
Usage Example for Modular Data Augmentation System

本文件展示如何在现有项目中集成新的模块化数据增强系统。
"""

import numpy as np
import sys

# 导入增强模块
from augmentation import (
    AugmentFactory, 
    AugmentPipeline,
    FlipAugment, 
    NoiseAugment,
    ElasticAugment,
    IntensityAugment,
    get_default_config,
    get_available_augmenters
)


def example_1_basic_usage():
    """示例1: 基本用法 - 从配置创建增强器"""
    print("=" * 60)
    print("示例1: 基本用法")
    print("=" * 60)
    
    # 定义配置
    config = {
        'use_augmentation': True,
        'augment_types': ['flip', 'noise'],
        'augment_probability': 0.5,
        'flip_axis': ['horizontal', 'vertical'],
        'noise_type': 'gaussian',
        'noise_sigma': 0.01,
    }
    
    # 创建增强器
    augmenter = AugmentFactory.create(config)
    print(f"创建增强器: {augmenter}")
    
    # 创建测试数据
    np.random.seed(42)
    lr_img = np.random.randn(8, 64, 64).astype(np.float32) * 200
    hr_img = np.random.randn(40, 64, 64).astype(np.float32) * 200
    
    print(f"原始数据形状: LR {lr_img.shape}, HR {hr_img.shape}")
    print(f"原始数据范围: LR [{lr_img.min():.2f}, {lr_img.max():.2f}]")
    
    # 应用增强（训练模式）
    lr_aug, hr_aug = augmenter(lr_img, hr_img, is_training=True)
    print(f"增强后数据范围: LR [{lr_aug.min():.2f}, {lr_aug.max():.2f}]")
    print(f"增强后数据形状: LR {lr_aug.shape}, HR {hr_aug.shape}")
    
    # 验证模式（不应用增强）
    lr_val, hr_val = augmenter(lr_img, hr_img, is_training=False)
    print(f"验证模式 - 数据是否变化: {not np.allclose(lr_img, lr_val)}")


def example_2_from_config_file():
    """示例2: 从配置文件加载"""
    print("\n" + "=" * 60)
    print("示例2: 从配置文件加载")
    print("=" * 60)
    
    # 从配置文件创建增强器
    config_path = 'config/augment_configs/aug_combined.txt'
    try:
        augmenter = AugmentFactory.from_config_file(config_path)
        print(f"从配置文件加载成功: {config_path}")
        print(f"增强器类型: {type(augmenter).__name__}")
    except FileNotFoundError:
        print(f"配置文件不存在，请检查路径: {config_path}")


def example_3_create_individual():
    """示例3: 创建单个增强器"""
    print("\n" + "=" * 60)
    print("示例3: 创建单个增强器")
    print("=" * 60)
    
    # 翻转增强器
    flip_aug = AugmentFactory.create_flip(
        prob=0.5, 
        axes=['horizontal', 'vertical']
    )
    print(f"翻转增强器: {flip_aug}")
    
    # 噪声增强器
    noise_aug = AugmentFactory.create_noise(
        prob=0.5,
        noise_type='both',
        sigma=0.01
    )
    print(f"噪声增强器: {noise_aug}")
    
    # 弹性形变增强器
    elastic_aug = AugmentFactory.create_elastic(
        prob=0.1,
        alpha=10.0,
        sigma=3.0
    )
    print(f"弹性增强器: {elastic_aug}")


def example_4_pipeline():
    """示例4: 使用增强管道"""
    print("\n" + "=" * 60)
    print("示例4: 使用增强管道")
    print("=" * 60)
    
    # 创建多个增强器
    augmenters = [
        FlipAugment(prob=0.5, axes=['horizontal', 'vertical']),
        NoiseAugment(prob=0.5, noise_type='gaussian', sigma=0.01),
    ]
    
    # 创建顺序管道
    pipeline = AugmentPipeline(augmenters, mode='sequential')
    print(f"顺序管道: {len(pipeline)} 个增强器")
    
    # 测试
    np.random.seed(42)
    lr_img = np.random.randn(8, 64, 64).astype(np.float32) * 200
    hr_img = np.random.randn(40, 64, 64).astype(np.float32) * 200
    
    lr_aug, hr_aug = pipeline(lr_img, hr_img, is_training=True)
    print(f"顺序管道应用完成")


def example_5_integration_with_dataset():
    """示例5: 集成到Dataset类（不修改原有代码）"""
    print("\n" + "=" * 60)
    print("示例5: 集成到Dataset类")
    print("=" * 60)
    
    print("""
# 在Dataset类中使用（创建新的Dataset类继承原有类）

from augmentation import AugmentFactory
from make_dataset import train_Dataset as BaseTrainDataset

class TrainDatasetWithAugment(BaseTrainDataset):
    def __init__(self, img_list, augment_config=None):
        super().__init__(img_list)
        
        # 创建增强器
        if augment_config:
            self.augmenter = AugmentFactory.create(augment_config)
        else:
            # 默认配置
            config = {
                'use_augmentation': True,
                'augment_types': ['flip', 'noise'],
                'augment_probability': 0.5,
                'flip_axis': ['horizontal', 'vertical'],
                'noise_type': 'gaussian',
                'noise_sigma': 0.01,
            }
            self.augmenter = AugmentFactory.create(config)
    
    def __getitem__(self, idx):
        # 获取原始数据
        result = super().__getitem__(idx)
        case_name, img, mask = result
        
        # 应用数据增强
        # 注意: img shape 可能是 (1, Z, H, W)
        if img.ndim == 4:
            img_squeeze = img[0]  # (Z, H, W)
            mask_squeeze = mask[0] if mask.ndim == 4 else mask
        else:
            img_squeeze = img
            mask_squeeze = mask
        
        # 应用增强（训练模式）
        img_aug, mask_aug = self.augmenter(
            img_squeeze, mask_squeeze, is_training=True
        )
        
        # 恢复原始形状
        if img.ndim == 4:
            img_aug = img_aug[np.newaxis]
            if mask.ndim == 4:
                mask_aug = mask_aug[np.newaxis]
        
        return [case_name, img_aug, mask_aug]


# 在训练脚本中使用
# train_dataset = TrainDatasetWithAugment(train_list, augment_config)
# val_dataset = val_Dataset(val_list)  # 验证集不使用增强
""")


def example_6_different_configs():
    """示例6: 不同配置的使用场景"""
    print("\n" + "=" * 60)
    print("示例6: 不同配置的使用场景")
    print("=" * 60)
    
    # 获取默认配置
    configs = {
        'conservative': get_default_config('conservative'),
        'aggressive': get_default_config('aggressive'),
        'noise_only': get_default_config('noise_only'),
        'flip_only': get_default_config('flip_only'),
    }
    
    for name, config in configs.items():
        print(f"\n{name} 配置:")
        augmenter = AugmentFactory.create(config)
        print(f"  增强器: {augmenter}")


def example_7_available_augmenters():
    """示例7: 查看所有可用增强器"""
    print("\n" + "=" * 60)
    print("示例7: 查看所有可用增强器")
    print("=" * 60)
    
    available = get_available_augmenters()
    print(f"可用增强器类型: {available}")
    
    all_types = AugmentFactory.get_available_types()
    print(f"工厂支持的所有类型: {all_types}")


def example_8_custom_config():
    """示例8: 自定义配置"""
    print("\n" + "=" * 60)
    print("示例8: 自定义配置")
    print("=" * 60)
    
    # 自定义配置 - 用于低剂量CT模拟
    low_dose_config = {
        'use_augmentation': True,
        'augment_types': ['noise', 'intensity'],
        'augment_probability': 0.8,
        'noise_type': 'both',
        'noise_sigma': 0.02,
        'noise_scale': 1.5,
        'intensity_operations': ['scale', 'shift'],
        'intensity_scale_range': (0.95, 1.05),
        'intensity_shift_range': (-20, 20),
    }
    
    augmenter = AugmentFactory.create(low_dose_config)
    print(f"低剂量CT模拟配置创建成功")
    print(f"配置详情: {low_dose_config}")


if __name__ == '__main__':
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("模块化数据增强系统使用示例")
    print("Modular Data Augmentation System Usage Examples")
    print("=" * 70)
    
    example_1_basic_usage()
    example_2_from_config_file()
    example_3_create_individual()
    example_4_pipeline()
    example_5_integration_with_dataset()
    example_6_different_configs()
    example_7_available_augmenters()
    example_8_custom_config()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成！")
    print("=" * 70)
