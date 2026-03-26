"""
数据增强配置
Configuration for Data Augmentation
"""

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

# 配置D：仅几何增强（修复 #1 - 禁用所有强度变换）
GEOMETRY_ONLY_AUG = {
    'flip_prob': 0.5,
    'rotate_prob': 0.3,
    'shift_prob': 0.2,
    'intensity_scale_prob': 0.0,   # 禁用：破坏数值物理意义
    'intensity_shift_prob': 0.0,   # 禁用：破坏数值物理意义
    'contrast_prob': 0.0,          # 禁用：破坏数值物理意义
    'gamma_prob': 0.0,             # 禁用
    'gaussian_noise_prob': 0.2,    # 降低概率
    'speckle_noise_prob': 0.0,     # 禁用
    'slice_artifact_prob': 0.0,    # 禁用
    'blur_prob': 0.0,              # 禁用
    'elastic_prob': 0.0,           # 禁用（计算量大）
}


# ==================== 配置说明文档 ====================
"""
使用示例 / Usage Example:

# 方法1: 在 config.py 中直接设置
opt.use_augmentation = True
opt.aug_prob = 0.5
opt.aug_config = CONSERVATIVE_AUG
opt.clip_ct = True
opt.min_hu = -1024
opt.max_hu = 3071
opt.normalize_ct = False

# 方法2: 在训练脚本中动态修改
if opt.mode == 'train':
    opt.use_augmentation = True
    if opt.augment_level == 'conservative':
        opt.aug_config = CONSERVATIVE_AUG
    elif opt.augment_level == 'aggressive':
        opt.aug_config = AGGRESSIVE_AUG
    elif opt.augment_level == 'geometry_only':
        opt.aug_config = GEOMETRY_ONLY_AUG  # 修复推荐

# 各增强操作说明:
# - flip_prob: 随机翻转概率 (X/Y轴)
# - rotate_prob: 90度旋转概率
# - shift_prob: 随机平移概率
# - intensity_scale_prob: 强度缩放概率
# - intensity_shift_prob: 强度偏移概率 (模拟窗宽窗位)
# - contrast_prob: 对比度调整概率
# - gamma_prob: Gamma校正概率
# - gaussian_noise_prob: 高斯噪声概率 (模拟低剂量CT)
# - speckle_noise_prob: 散斑噪声概率
# - slice_artifact_prob: 层间伪影概率
# - blur_prob: 随机模糊概率
# - elastic_prob: 弹性形变概率 (计算量大，慎用)
"""
