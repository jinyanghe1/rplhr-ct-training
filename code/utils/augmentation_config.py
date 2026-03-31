"""
数据增强配置示例
Configuration Example for Data Augmentation

在 config.py 中添加以下配置项即可启用数据增强。
Add these configurations to config.py to enable data augmentation.
"""

# ==================== 基础增强配置 ====================
# 增强操作的基础概率 (0-1)
AUG_PROB = 0.5

# CT值裁剪范围 (HU单位)
MIN_HU = -1024   # 空气
MAX_HU = 3071    # 高密度骨/金属

# 窗宽窗位 (归一化用)
WINDOW_CENTER = 0
WINDOW_WIDTH = 1000

# ==================== 增强配置字典 ====================
# 可单独控制每种增强的概率
CONSERVATIVE_AUG = {
    # 空间变换
    'flip_prob': 0.5,          # 随机翻转
    'rotate_prob': 0.3,        # 90度旋转
    'shift_prob': 0.2,         # 随机平移

    # 强度变换
    'intensity_scale_prob': 0.2,   # 强度缩放
    'intensity_shift_prob': 0.2,   # 强度偏移
    'contrast_prob': 0.2,          # 对比度调整
    'gaussian_noise_prob': 0.3,    # 高斯噪声
    'slice_artifact_prob': 0.05,   # 层间伪影
    'elastic_prob': 0.0,           # 弹性形变 (计算量大)
}


# ==================== 推荐配置组合 ====================

# 配置A：保守增强（推荐用于小数据集或baseline）
# 注意：已在上面定义

# 配置B：激进增强（适用于大数据集，追求泛化性）
AGGRESSIVE_AUG = {
    'flip_prob': 0.5,
    'rotate_prob': 0.5,
    'shift_prob': 0.4,
    'intensity_scale_prob': 0.4,
    'intensity_shift_prob': 0.4,
    'contrast_prob': 0.4,
    'gamma_prob': 0.2,
    'gaussian_noise_prob': 0.6,
    'speckle_noise_prob': 0.2,
    'slice_artifact_prob': 0.15,
    'blur_prob': 0.1,
    'elastic_prob': 0.1,
}

# 配置C：仅噪声增强（用于测试鲁棒性）
NOISE_ONLY_AUG = {
    'flip_prob': 0.0,
    'rotate_prob': 0.0,
    'shift_prob': 0.0,
    'intensity_scale_prob': 0.0,
    'intensity_shift_prob': 0.0,
    'contrast_prob': 0.0,
    'gaussian_noise_prob': 0.8,
    'speckle_noise_prob': 0.3,
    'slice_artifact_prob': 0.2,
    'elastic_prob': 0.0,
}


"""
使用示例 / Usage Example:

# 在 config.py 中:
from augmentation_config import CONSERVATIVE_AUG

opt.use_augmentation = True
opt.aug_prob = 0.5
opt.aug_config = CONSERVATIVE_AUG
opt.clip_ct = True
opt.normalize_ct = False  # 保持原始HU值

# 或者在训练脚本中动态修改:
if opt.mode == 'train':
    opt.use_augmentation = True
    if opt.augment_level == 'conservative':
        opt.aug_config = CONSERVATIVE_AUG
    elif opt.augment_level == 'aggressive':
        opt.aug_config = AGGRESSIVE_AUG
"""
