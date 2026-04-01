#!/usr/bin/env python3
"""
CT 厚薄层配准脚本
对 cleaned_final 目录中的厚扫-薄扫配对进行配准处理

主要功能：
1. 刚体配准（Rigid Registration）：解决thick和thin之间的空间偏移
2. XY重采样：统一XY平面的spacing
3. 质量验证：计算NMI（归一化互信息）评估配准质量

作者: Auto-generated
日期: 2026-03-20
"""

import os
import sys
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 配置参数
DATA_ROOT = Path("RPLHR-CT-main/data/thick-thin-layer-paired/cleaned_final")
OUTPUT_ROOT = Path("RPLHR-CT-main/data/thick-thin-layer-paired/registered")
BACKUP_ROOT = Path("RPLHR-CT-main/data/thick-thin-layer-paired/cleaned_final_backup")

# 刚体配准参数
REGISTRATION_PARAMS = {
    'learning_rate': 1.0,
    'min_step': 0.0001,
    'max_iterations': 200,
    'relaxation_factor': 0.5,
    'gradient_magnitude_tolerance': 0.0001,
}


def get_patient_files(patient_id, split='train'):
    """获取患者的thick和thin文件路径"""
    thick_path = DATA_ROOT / split / 'thick' / f'{patient_id}.nii.gz'
    thin_path = DATA_ROOT / split / 'thin' / f'{patient_id}.nii.gz'
    return thick_path, thin_path


def read_nifti(filepath):
    """读取NIfTI文件"""
    image = sitk.ReadImage(str(filepath))
    return image


def save_nifti(image, filepath):
    """保存NIfTI文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteImage(image, str(filepath))
    print(f"  已保存: {filepath}")


def compute_nmi(fixed_img, moving_img):
    """计算归一化互信息 (Normalized Mutual Information)"""
    # 转换为numpy数组
    fixed_arr = sitk.GetArrayFromImage(fixed_img).flatten()
    moving_arr = sitk.GetArrayFromImage(moving_img).flatten()
    
    # 使用直方图计算互信息
    hist_2d, _, _ = np.histogram2d(fixed_arr, moving_arr, bins=50)
    
    # 计算联合熵和边缘熵
    p_xy = hist_2d / float(np.sum(hist_2d))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # 避免log(0)
    p_xy = p_xy[p_xy > 0]
    p_x = p_x[p_x > 0]
    p_y = p_y[p_y > 0]
    
    h_xy = -np.sum(p_xy * np.log2(p_xy))
    h_x = -np.sum(p_x * np.log2(p_x))
    h_y = -np.sum(p_y * np.log2(p_y))
    
    # 归一化互信息 = (H(x) + H(y)) / H(x,y)
    nmi = (h_x + h_y) / h_xy if h_xy > 0 else 0
    return nmi


def resample_to_reference(moving_img, reference_img, interpolator=sitk.sitkBSpline, default_value=-1024):
    """将moving图像重采样到reference图像的物理空间"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampled = resampler.Execute(moving_img)
    return resampled


def rigid_registration(fixed_img, moving_img, verbose=False):
    """
    执行3D刚体配准
    
    参数:
        fixed_img: 固定图像 (通常是厚扫)
        moving_img: 移动图像 (通常是薄扫)
        verbose: 是否输出详细日志
    
    返回:
        registered_img: 配准后的图像
        transform: 变换参数
        metric_value: 最终度量值
    """
    # 初始化变换
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # 配准方法
    registration_method = sitk.ImageRegistrationMethod()
    
    # 相似度度量：互信息 (适合多模态或不同参数CT)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.3)
    
    # 优化器
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=REGISTRATION_PARAMS['learning_rate'],
        minStep=REGISTRATION_PARAMS['min_step'],
        numberOfIterations=REGISTRATION_PARAMS['max_iterations'],
        relaxationFactor=REGISTRATION_PARAMS['relaxation_factor'],
        gradientMagnitudeTolerance=REGISTRATION_PARAMS['gradient_magnitude_tolerance']
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # 插值器
    registration_method.SetInterpolator(sitk.sitkBSpline)
    
    # 初始变换
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # 执行配准
    if verbose:
        print(f"  开始配准... (最大迭代次数: {REGISTRATION_PARAMS['max_iterations']})")
        registration_method.AddCommand(sitk.sitkIterationEvent, 
            lambda: print(f"    迭代 {registration_method.GetOptimizerIteration()}: "
                         f"度量值 = {registration_method.GetMetricValue():.6f}"))
    
    final_transform = registration_method.Execute(fixed_img, moving_img)
    
    if verbose:
        print(f"  配准完成! 最终度量值: {registration_method.GetMetricValue():.6f}")
        print(f"  变换参数: 平移 = [{final_transform.GetTranslation()[0]:.2f}, "
              f"{final_transform.GetTranslation()[1]:.2f}, {final_transform.GetTranslation()[2]:.2f}]")
    
    # 应用变换
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetTransform(final_transform)
    registered_img = resampler.Execute(moving_img)
    
    return registered_img, final_transform, registration_method.GetMetricValue()


def align_xy_spacing(thin_img, thick_img):
    """
    将薄扫的XY spacing对齐到厚扫
    
    用于处理XY spacing不匹配的情况
    """
    thick_spacing = thick_img.GetSpacing()
    thin_spacing = thin_img.GetSpacing()
    
    # 检查是否需要重采样
    xy_diff = abs(thin_spacing[0] - thick_spacing[0]) + abs(thin_spacing[1] - thick_spacing[1])
    
    if xy_diff < 0.01:
        print(f"  XY spacing已对齐 (thin: {thin_spacing[:2]}, thick: {thick_spacing[:2]})")
        return thin_img
    
    print(f"  XY spacing不匹配! thin: {thin_spacing[:2]} vs thick: {thick_spacing[:2]}")
    print(f"  对薄扫进行XY重采样...")
    
    # 创建新的spacing：保持Z不变，XY使用厚扫的spacing
    new_spacing = (thick_spacing[0], thick_spacing[1], thin_spacing[2])
    
    # 计算新尺寸
    thin_size = thin_img.GetSize()
    new_size = [
        int(round(thin_size[0] * thin_spacing[0] / new_spacing[0])),
        int(round(thin_size[1] * thin_spacing[1] / new_spacing[1])),
        thin_size[2]  # Z维度不变
    ]
    
    # 重采样
    resampled = sitk.Resample(
        thin_img,
        new_size,
        sitk.Transform(),
        sitk.sitkBSpline,
        thin_img.GetOrigin(),
        new_spacing,
        thin_img.GetDirection(),
        -1024,
        thin_img.GetPixelID()
    )
    
    print(f"  重采样后尺寸: {resampled.GetSize()}, spacing: {resampled.GetSpacing()}")
    return resampled


def process_patient(patient_id, split, use_rigid=True, verbose=False):
    """
    处理单个患者的配准
    
    参数:
        patient_id: 患者ID (如 HCTSR-0001)
        split: 数据集划分 (train/val/test)
        use_rigid: 是否使用刚体配准
        verbose: 是否输出详细日志
    
    返回:
        result_dict: 包含处理结果的字典
    """
    print(f"\n处理患者: {patient_id} ({split})")
    
    thick_path, thin_path = get_patient_files(patient_id, split)
    
    if not thick_path.exists() or not thin_path.exists():
        print(f"  错误: 文件不存在!")
        return None
    
    # 读取图像
    thick_img = read_nifti(thick_path)
    thin_img = read_nifti(thin_path)
    
    print(f"  厚扫: shape={thick_img.GetSize()}, spacing={thick_img.GetSpacing()}")
    print(f"  薄扫: shape={thin_img.GetSize()}, spacing={thin_img.GetSpacing()}")
    
    result = {
        'patient_id': patient_id,
        'split': split,
        'thick_path': str(thick_path),
        'thin_path': str(thin_path),
        'original_thick_shape': list(thick_img.GetSize()),
        'original_thin_shape': list(thin_img.GetSize()),
        'original_thick_spacing': list(thick_img.GetSpacing()),
        'original_thin_spacing': list(thin_img.GetSpacing()),
    }
    
    # 步骤1: XY spacing对齐 (如果需要)
    thin_img_aligned = align_xy_spacing(thin_img, thick_img)
    
    # 计算配准前的NMI
    # 由于thick和thin的Z维度不同，需要先将thick上采样到thin的Z维度进行比较
    thick_upsampled = resample_to_reference(thick_img, thin_img_aligned, 
                                            interpolator=sitk.sitkBSpline, default_value=-1024)
    nmi_before = compute_nmi(thick_upsampled, thin_img_aligned)
    print(f"  配准前 NMI: {nmi_before:.4f}")
    result['nmi_before'] = float(nmi_before)
    
    # 步骤2: 刚体配准 (如果启用)
    if use_rigid and nmi_before < 0.5:  # 只在NMI较低时进行配准
        print(f"  NMI较低，执行刚体配准...")
        try:
            # 使用下采样的图像进行配准以加快速度
            thick_down = sitk.Shrink(thick_img, [2, 2, 1])
            thin_down = sitk.Shrink(thin_img_aligned, [2, 2, 1])
            
            # 配准
            _, transform, metric_val = rigid_registration(thick_down, thin_down, verbose=verbose)
            
            # 应用变换到原始分辨率图像
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(thick_img)
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetDefaultPixelValue(-1024)
            resampler.SetTransform(transform)
            thin_img_registered = resampler.Execute(thin_img_aligned)
            
            result['registration_applied'] = True
            result['transform_translation'] = list(transform.GetTranslation())
            result['transform_rotation'] = list(transform.GetVersor())
            result['metric_value'] = float(metric_val)
            
        except Exception as e:
            print(f"  配准失败: {e}")
            thin_img_registered = thin_img_aligned
            result['registration_applied'] = False
            result['registration_error'] = str(e)
    else:
        if not use_rigid:
            print(f"  跳过刚体配准 (use_rigid=False)")
        else:
            print(f"  NMI正常，跳过刚体配准")
        thin_img_registered = thin_img_aligned
        result['registration_applied'] = False
    
    # 计算配准后的NMI
    thick_upsampled_final = resample_to_reference(thick_img, thin_img_registered, 
                                                   interpolator=sitk.sitkBSpline, default_value=-1024)
    nmi_after = compute_nmi(thick_upsampled_final, thin_img_registered)
    print(f"  配准后 NMI: {nmi_after:.4f}")
    result['nmi_after'] = float(nmi_after)
    result['nmi_improvement'] = float(nmi_after - nmi_before)
    
    # 保存配准后的薄扫
    output_thin_dir = OUTPUT_ROOT / split / 'thin'
    output_thin_path = output_thin_dir / f'{patient_id}.nii.gz'
    save_nifti(thin_img_registered, output_thin_path)
    
    # 复制厚扫到输出目录 (保持不变)
    output_thick_dir = OUTPUT_ROOT / split / 'thick'
    output_thick_path = output_thick_dir / f'{patient_id}.nii.gz'
    
    # 使用符号链接或复制
    os.makedirs(output_thick_dir, exist_ok=True)
    if not output_thick_path.exists():
        import shutil
        shutil.copy(thick_path, output_thick_path)
        print(f"  已复制厚扫: {output_thick_path}")
    
    result['output_thin_path'] = str(output_thin_path)
    result['output_thick_path'] = str(output_thick_path)
    
    return result


def load_split_manifest():
    """加载数据集划分清单"""
    manifest_path = DATA_ROOT / 'split_manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return None


def main():
    global OUTPUT_ROOT
    
    parser = argparse.ArgumentParser(description='CT 厚薄层配准处理')
    parser.add_argument('--patients', nargs='+', help='指定处理的患者ID (如 HCTSR-0001)')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all',
                       help='处理哪个数据集划分')
    parser.add_argument('--no-rigid', action='store_true', help='禁用刚体配准')
    parser.add_argument('--verbose', action='store_true', help='输出详细日志')
    parser.add_argument('--output-dir', default=str(OUTPUT_ROOT), help='输出目录')
    
    args = parser.parse_args()
    
    OUTPUT_ROOT = Path(args.output_dir)
    
    print("=" * 60)
    print("CT 厚薄层配准处理")
    print("=" * 60)
    print(f"输入目录: {DATA_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"刚体配准: {'禁用' if args.no_rigid else '启用'}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载划分清单
    manifest = load_split_manifest()
    if manifest is None:
        print("错误: 无法加载 split_manifest.json")
        return
    
    # 确定要处理的患者列表
    patients_to_process = []
    
    # 辅助函数：从manifest条目获取patient_id
    def get_pid(entry):
        return entry if isinstance(entry, str) else entry.get('patient_id')
    
    if args.patients:
        # 指定了具体患者
        for pid in args.patients:
            for split in ['train', 'val', 'test']:
                for entry in manifest.get(split, []):
                    if get_pid(entry) == pid:
                        patients_to_process.append((pid, split))
                        break
    else:
        # 按split处理
        splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
        for split in splits:
            for entry in manifest.get(split, []):
                pid = get_pid(entry)
                if pid:
                    patients_to_process.append((pid, split))
    
    print(f"\n共需处理 {len(patients_to_process)} 个患者")
    
    # 处理每个患者
    results = []
    for i, (patient_id, split) in enumerate(patients_to_process, 1):
        print(f"\n[{i}/{len(patients_to_process)}] ", end="")
        result = process_patient(
            patient_id, 
            split, 
            use_rigid=not args.no_rigid,
            verbose=args.verbose
        )
        if result:
            results.append(result)
    
    # 保存结果报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_root': str(DATA_ROOT),
        'output_root': str(OUTPUT_ROOT),
        'use_rigid': not args.no_rigid,
        'total_processed': len(results),
        'results': results,
        'summary': {
            'avg_nmi_before': np.mean([r['nmi_before'] for r in results]),
            'avg_nmi_after': np.mean([r['nmi_after'] for r in results]),
            'avg_improvement': np.mean([r['nmi_improvement'] for r in results]),
            'patients_with_registration': sum([r.get('registration_applied', False) for r in results]),
        }
    }
    
    report_path = OUTPUT_ROOT / 'registration_report.json'
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"报告已保存: {report_path}")
    print(f"平均NMI: 配准前={report['summary']['avg_nmi_before']:.4f}, "
          f"配准后={report['summary']['avg_nmi_after']:.4f}")
    print(f"进行刚体配准的患者数: {report['summary']['patients_with_registration']}")


if __name__ == '__main__':
    main()
