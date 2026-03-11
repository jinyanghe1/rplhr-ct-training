# DICOM 格式支持指南

## ❓ 问题：DICOM 格式能直接使用吗？

**答案：不能直接使用**，需要**转换格式**或**修改代码**。

原因：
1. 当前代码硬编码了 `.nii.gz` 文件扩展名
2. DICOM 通常是一个文件夹包含多个 `.dcm` 切片，而非单个文件
3. 病例列表生成逻辑基于文件扩展名过滤

---

## 🛠️ 解决方案

### 方案 1：DICOM → NIfTI 转换（推荐 ⭐）

将 DICOM 文件夹批量转换为 `.nii.gz` 格式，然后直接使用原代码训练。

#### 优点
- ✅ 无需修改训练代码
- ✅ 转换一次，永久使用
- ✅ I/O 效率更高（单文件 vs 多文件）
- ✅ 与原始代码 100% 兼容

#### 使用方法

**1. 准备 DICOM 数据**

```
dicom_data/                     # 你的 DICOM 数据目录
├── train/
│   ├── 5mm/
│   │   ├── CT00000000/         # DICOM 文件夹（包含多个 .dcm 文件）
│   │   │   ├── 000001.dcm
│   │   │   ├── 000002.dcm
│   │   │   └── ...
│   │   ├── CT00000001/
│   │   └── ...
│   └── 1mm/
│       ├── CT00000000/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

**2. 运行转换脚本**

```bash
cd RPLHR-CT-main/code

# 批量转换整个数据集
python convert_dicom_to_nifti.py \
    --input_root ../dicom_data \
    --output_root ../data

# 验证转换结果
python convert_dicom_to_nifti.py \
    --input_root ../dicom_data \
    --output_root ../data \
    --verify
```

**3. 开始训练**

```bash
# 使用转换后的数据
python train.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN
```

#### 输出结构

```
data/                           # 转换后的 NIfTI 数据
├── train/
│   ├── 5mm/
│   │   ├── CT00000000.nii.gz   # 转换后的文件
│   │   ├── CT00000001.nii.gz
│   │   └── ...
│   └── 1mm/
│       ├── CT00000000.nii.gz
│       └── ...
└── ...
```

---

### 方案 2：修改代码支持 DICOM

直接修改代码，使其支持读取 DICOM 文件夹。

#### 优点
- ✅ 无需数据转换
- ✅ 支持 DICOM 和 NIfTI 混合使用

#### 缺点
- ❌ 需要修改多处代码
- ❌ 每次读取需要加载多个 DICOM 文件（较慢）
- ❌ 需要确保 `train/5mm` 和 `train/1mm` 中的 DICOM 文件夹命名一致

#### 使用方法

**1. 替换数据加载模块**

将 `code/utils/in_model.py` 替换为 `in_model_dicom.py`：

```bash
cd RPLHR-CT-main/code

# 备份原文件
mv utils/in_model.py utils/in_model_original.py

# 使用 DICOM 版本
mv utils/in_model_dicom.py utils/in_model.py
```

**2. 修改病例列表生成**

原 `train.py` 中的：
```python
train_list = [each.split('.')[0] for each in sorted(os.listdir(opt.path_img + 'train/1mm/'))]
```

需要修改为支持文件夹检测的版本。

已为你准备好完整的 `train_dicom.py`，直接使用即可：

```bash
# 使用 DICOM 支持的训练脚本
python train_dicom.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN
```

**3. 确保目录结构正确**

```
data/                           # DICOM 数据目录
├── train/
│   ├── 5mm/
│   │   ├── CT00000000/         # DICOM 文件夹
│   │   ├── CT00000001/
│   │   └── ...
│   └── 1mm/
│       ├── CT00000000/         # 同名文件夹（与 5mm 配对）
│       └── ...
└── ...
```

---

## 📊 方案对比

| 特性 | 方案 1: 转换格式 | 方案 2: 修改代码 |
|-----|-----------------|-----------------|
| **代码修改** | ❌ 不需要 | ✅ 需要 |
| **转换时间** | ✅ 一次性 | ❌ 无 |
| **磁盘空间** | 需要额外 ~30GB | ❌ 不需要 |
| **I/O 速度** | ✅ 快（单文件） | ❌ 慢（多文件）|
| **数据完整性** | ✅ 高 | ✅ 高 |
| **适用场景** | 长期训练 | 快速测试 |

---

## 🔍 常见问题

### Q1: DICOM 转 NIfTI 会丢失信息吗？

**A:** 不会。SimpleITK 的 DICOM 读取会保留所有体素数据和元数据（如 spacing、origin、direction）。但某些 DICOM 特有的标签（如患者信息）可能会被简化。

### Q2: 我的 DICOM 文件命名不规范，能转换吗？

**A:** 可以。`convert_dicom_to_nifti.py` 使用 SimpleITK 的 `ImageSeriesReader`，它会自动识别同一序列的所有 DICOM 文件，不依赖文件名。只要一个文件夹内是一个完整的 CT 序列即可。

### Q3: DICOM 和 NIfTI 可以混用吗？

**A:** 可以。`in_model_dicom.py` 支持自动检测：
- 如果是文件夹 → 按 DICOM 读取
- 如果是 `.nii.gz` 文件 → 按 NIfTI 读取

### Q4: 如何检查 DICOM 数据是否正确？

**A:** 使用以下代码检查：

```python
import SimpleITK as sitk

# 读取 DICOM 文件夹
reader = sitk.ImageSeriesReader()
dicom_files = reader.GetGDCMSeriesFileNames('path/to/dicom_folder')
print(f'找到 {len(dicom_files)} 个 DICOM 文件')

reader.SetFileNames(dicom_files)
image = reader.Execute()
print(f'图像尺寸: {image.GetSize()}')
print(f'体素间距: {image.GetSpacing()}')
```

### Q5: 转换后如何验证配对是否正确？

**A:** 使用以下脚本验证 5mm 和 1mm 的配对关系：

```bash
cd RPLHR-CT-main/code
python data_flow_demo.py --data_path ../data
```

确保 5mm 和 1mm 文件夹中的病例名称一一对应。

---

## 🚀 推荐流程

对于大多数人，推荐以下流程：

```bash
# Step 1: 准备 DICOM 数据（确保 5mm 和 1mm 配对）
# 你的 DICOM 数据目录结构：
# dicom_data/train/5mm/case_001/, dicom_data/train/1mm/case_001/

# Step 2: 转换格式
cd RPLHR-CT-main/code
python convert_dicom_to_nifti.py \
    --input_root ../../dicom_data \
    --output_root ../data

# Step 3: 验证转换结果
python convert_dicom_to_nifti.py \
    --input_root ../../dicom_data \
    --output_root ../data \
    --verify

# Step 4: 运行训练
python train.py train --path_key SRM --gpu_idx 0 --net_idx TVSRN
```

---

## 📁 提供的文件

| 文件 | 用途 |
|-----|------|
| `code/convert_dicom_to_nifti.py` | DICOM 转 NIfTI 转换脚本 |
| `code/utils/in_model_dicom.py` | 支持 DICOM 的数据加载模块 |
| `code/train_dicom.py` | 支持 DICOM 的训练脚本 |
| `DICOM_SUPPORT_GUIDE.md` | 本说明文档 |

---

## 📝 注意事项

1. **DICOM 序列完整性**：确保每个文件夹包含完整的 CT 序列，不要混放不同序列的文件
2. **配对一致性**：`train/5mm/case_001` 必须与 `train/1mm/case_001` 对应同一患者
3. **层厚比例**：5mm 和 1mm 的 Z 方向比例应为 5:1，这是网络设计的上采样倍数
4. **内存使用**：转换大容量数据时，确保有足够磁盘空间（约 30GB）
