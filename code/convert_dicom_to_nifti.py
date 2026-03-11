#!/usr/bin/env python3
"""
DICOM 转换为 NIfTI 格式

支持将 DICOM 文件夹转换为 .nii.gz 格式，保持与 RPLHR-CT 代码兼容

用法:
    # 转换单个 DICOM 文件夹
    python convert_dicom_to_nifti.py \
        --input /path/to/dicom_folder \
        --output /path/to/output.nii.gz
    
    # 批量转换整个数据集
    python convert_dicom_to_nifti.py \
        --input_root /path/to/dicom_data \
        --output_root /path/to/nifti_data \
        --pattern "*/5mm" "*/1mm"

目录结构示例:
    dicom_data/                 nifti_data/ (输出)
    ├── train/                  ├── train/
    │   ├── 5mm/                │   ├── 5mm/
    │   │   ├── patient_001/    │   │   ├── patient_001.nii.gz
    │   │   ├── patient_002/    │   │   ├── patient_002.nii.gz
    │   │   └── ...             │   │   └── ...
    │   └── 1mm/                │   └── 1mm/
    │       ├── patient_001/    │       ├── patient_001.nii.gz
    │       └── ...             │       └── ...
    └── val/                    └── val/
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

try:
    import SimpleITK as sitk
except ImportError:
    print("错误: 请安装 SimpleITK: pip install SimpleITK")
    sys.exit(1)


def convert_single_dicom(dicom_folder, output_file, resample=None):
    """
    将单个 DICOM 文件夹转换为 NIfTI
    
    Args:
        dicom_folder: DICOM 文件夹路径
        output_file: 输出 .nii.gz 文件路径
        resample: 可选，重采样到指定体素大小 (如 [1,1,1])
    """
    # 读取 DICOM 系列
    reader = sitk.ImageSeriesReader()
    
    # 获取 DICOM 文件列表
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    
    if len(dicom_files) == 0:
        print(f"错误: 在 {dicom_folder} 中未找到 DICOM 文件")
        return False
    
    # 读取图像
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    
    # 可选：重采样
    if resample is not None:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # 计算新尺寸
        new_spacing = resample
        new_size = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(3)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        image = resampler.Execute(image)
        print(f"  重采样: {original_spacing} -> {new_spacing}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为 NIfTI
    sitk.WriteImage(image, output_file)
    
    return True


def batch_convert(input_root, output_root, subsets=['train', 'val', 'test'], 
                  resolutions=['5mm', '1mm'], copy_non_dicom=False):
    """
    批量转换 DICOM 数据集
    
    Args:
        input_root: DICOM 数据根目录
        output_root: NIfTI 输出根目录
        subsets: 子集列表 (train/val/test)
        resolutions: 分辨率文件夹列表 (5mm/1mm)
        copy_non_dicom: 是否复制非 DICOM 文件
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print()
    
    total_converted = 0
    total_failed = 0
    
    for subset in subsets:
        for res in resolutions:
            input_dir = input_root / subset / res
            output_dir = output_root / subset / res
            
            if not input_dir.exists():
                print(f"跳过: {input_dir} 不存在")
                continue
            
            print(f"处理: {input_dir}")
            
            # 获取所有子文件夹 (每个文件夹是一个 DICOM 序列)
            subfolders = [f for f in input_dir.iterdir() if f.is_dir()]
            
            if len(subfolders) == 0:
                # 可能是直接包含 DICOM 文件的文件夹
                print(f"  警告: {input_dir} 中没有子文件夹")
                continue
            
            print(f"  发现 {len(subfolders)} 个病例")
            
            # 转换每个病例
            for dicom_folder in tqdm(subfolders, desc=f"  转换 {subset}/{res}"):
                case_name = dicom_folder.name
                output_file = output_dir / f"{case_name}.nii.gz"
                
                # 跳过已存在的文件
                if output_file.exists():
                    continue
                
                try:
                    if convert_single_dicom(str(dicom_folder), str(output_file)):
                        total_converted += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    print(f"  错误 {case_name}: {e}")
                    total_failed += 1
    
    print()
    print(f"转换完成: {total_converted} 成功, {total_failed} 失败")


def verify_conversion(input_root, output_root):
    """验证转换结果"""
    print("\n验证转换结果...")
    
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    for subset in ['train', 'val', 'test']:
        for res in ['5mm', '1mm']:
            input_dir = input_root / subset / res
            output_dir = output_root / subset / res
            
            if not input_dir.exists():
                continue
            
            input_cases = len([f for f in input_dir.iterdir() if f.is_dir()])
            output_cases = len([f for f in output_dir.iterdir() if f.suffix == '.gz'])
            
            status = "✓" if input_cases == output_cases else "✗"
            print(f"  {status} {subset}/{res}: {input_cases} 输入, {output_cases} 输出")


def main():
    parser = argparse.ArgumentParser(
        description='将 DICOM 格式转换为 NIfTI (.nii.gz) 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1. 批量转换整个数据集
  python convert_dicom_to_nifti.py \\
      --input_root ../dicom_data \\
      --output_root ../data

  # 2. 转换单个文件夹
  python convert_dicom_to_nifti.py \\
      --input ../dicom_data/train/5mm/patient_001 \\
      --output ../data/train/5mm/patient_001.nii.gz

  # 3. 验证转换结果
  python convert_dicom_to_nifti.py \\
      --input_root ../dicom_data \\
      --output_root ../data \\
      --verify
""")
    
    parser.add_argument('--input', type=str, help='输入 DICOM 文件夹路径')
    parser.add_argument('--output', type=str, help='输出 .nii.gz 文件路径')
    parser.add_argument('--input_root', type=str, help='输入 DICOM 数据根目录')
    parser.add_argument('--output_root', type=str, help='输出 NIfTI 数据根目录')
    parser.add_argument('--subsets', nargs='+', default=['train', 'val', 'test'],
                        help='要处理的子集 (默认: train val test)')
    parser.add_argument('--resolutions', nargs='+', default=['5mm', '1mm'],
                        help='要处理的分辨率文件夹 (默认: 5mm 1mm)')
    parser.add_argument('--verify', action='store_true',
                        help='验证转换结果')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.verify:
        if not args.input_root or not args.output_root:
            print("错误: --verify 需要 --input_root 和 --output_root")
            sys.exit(1)
        verify_conversion(args.input_root, args.output_root)
        return
    
    if args.input and args.output:
        # 单文件转换
        print(f"转换: {args.input} -> {args.output}")
        if convert_single_dicom(args.input, args.output):
            print("成功!")
        else:
            print("失败!")
            sys.exit(1)
    
    elif args.input_root and args.output_root:
        # 批量转换
        batch_convert(
            args.input_root,
            args.output_root,
            subsets=args.subsets,
            resolutions=args.resolutions
        )
    
    else:
        parser.print_help()
        print("\n错误: 请指定 --input 和 --output，或 --input_root 和 --output_root")
        sys.exit(1)


if __name__ == '__main__':
    main()
