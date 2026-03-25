"""
数据增强配置示例
Configuration Example for Data Augmentation

在 config.py 中添加以下配置项即可启用数据增强。
Add these configurations to config.py to enable data augmentation.
"""

# ==================== 基础增强开关 ====================
# 是否启用数据增强
opt.use_augmentation = True

# 增强操作的基础概率 (0-1)
opt.aug_prob = 0.5

# 是否裁剪CT值到合理范围 (HU单位)
opt.clip_ct = True
opt.min_hu = -1024   # 空气
opt.max_hu = 3071    # 高密度骨/金属

# 是否归一化到[0,1] (使用窗宽窗位)
opt.normalize_ct = False
opt.window_center = 0      # 窗位
opt.window_width = 1000    # 窗宽

# ==================== 增强配置字典 ====================
# 可单独控制每种增强的概率
opt.aug_config = {
    # 空间变换
    'flip_prob': 0.5,          # 随机翻转
    'rotate_prob': 0.3,        # 90度旋转
    'shift_prob': 0.3,         # 随机平移
    
    # 强度变换
    'intensity_scale_prob': 0.3,   # 强度缩放
    'intensity_shift_prob': 0.3,   # 强度偏移
    'contrast_prob': 0.3,          # 对比度调整
    'gamma_prob': 0.0,             # Gamma校正 (默认关闭)
    
    # 噪声与伪影
    'gaussian_noise_prob': 0.5,    # 高斯噪声
    'speckle_noise_prob': 0.0,     # 散斑噪声 (默认关闭)
    'slice_artifact_prob': 0.1,    # 层间伪影
    'blur_prob': 0.0,              # 随机模糊 (默认关闭)
    
    # 高级增强
    'elastic_prob': 0.1,           # 弹性形变 (计算量大)
}


# ==================== 推荐配置组合 ====================

# 配置A：保守增强（推荐用于小数据集或baseline）
CONSERVATIVE_AUG = {
    'flip_prob': 0.5,
    'rotate_prob': 0.3,
    'shift_prob': 0.2,
    'intensity_scale_prob': 0.2,
    'intensity_shift_prob': 0.2,
    'contrast_prob': 0.2,
    'gaussian_noise_prob': 0.3,
    'slice_artifact_prob': 0.05,
    'elastic_prob': 0.0,
}

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
