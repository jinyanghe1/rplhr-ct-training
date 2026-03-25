# RPLHR-CT 数据流程详解

## 一、能直接用 nii.gz 训练吗？

**答案：可以直接使用，无需转换格式**

代码使用 `SimpleITK` (sitk) 直接读取 `.nii.gz` 文件：

```python
import SimpleITK as sitk

# 直接读取 nii.gz
tmp_img = sitk.GetArrayFromImage(sitk.ReadImage('train/5mm/CT00000000.nii.gz'))
tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage('train/1mm/CT00000000.nii.gz'))
```

## 二、数据配对方式（核心逻辑）

### 配对规则
| 输入 (X) | 标签 (Y) | 上采样倍数 |
|---------|---------|-----------|
| `train/5mm/*.nii.gz` | `train/1mm/*.nii.gz` | 5x（层厚方向） |

**命名约定**：相同文件名的 5mm 和 1mm 构成一对

```
数据目录结构：
../data/
├── train/
│   ├── 1mm/           # HR (高分辨率)
│   │   ├── CT00000000.nii.gz
│   │   └── CT00000001.nii.gz
│   └── 5mm/           # LR (低分辨率)
│       ├── CT00000000.nii.gz  ← 配对
│       └── CT00000001.nii.gz  ← 配对
├── val/
│   ├── 1mm/
│   └── 5mm/
└── test/
    ├── 1mm/
    └── 5mm/
```

## 三、训练 vs 验证的数据处理差异

### 📊 对比表

| 阶段 | 裁剪策略 | 输入尺寸 | 标签尺寸 | 处理方式 |
|-----|---------|---------|---------|---------|
| **训练** | 随机裁剪 | 4×256×256 | 16×256×256 | 每次随机取一个块 |
| **验证** | 滑动窗口切块 | 整卷分割 | 整卷分割 | 先切块→推理→拼接 |

### 🔍 详细解释

#### 1. 训练阶段 (`get_train_img`)

```python
# 简化的伪代码
def get_train_img(img_path, case_name):
    # 读取配对数据
    hr = read(f'{img_path}/train/1mm/{case_name}.nii.gz')  # 形状: (Z*5, 512, 512)
    lr = read(f'{img_path}/train/5mm/{case_name}.nii.gz')  # 形状: (Z, 512, 512)
    
    # 随机裁剪 LR 块 (z, y, x)
    lr_crop = lr[z_s:z_s+4, y_s:y_s+256, x_s:x_s+256]
    
    # 对应 HR 块（5倍关系）
    # 层厚方向：5mm → 1mm 是 5:1 关系
    # 4层 5mm 对应 16层 1mm
    hr_z_start = z_s * 5 + 3
    hr_z_end = (z_s + 4 - 1) * 5 - 2
    hr_crop = hr[hr_z_start:hr_z_end, y_s:y_s+256, x_s:x_s+256]
    
    return lr_crop, hr_crop  # (4,256,256) 和 (16,256,256)
```

**直观例子**：
```
一个 CT 卷：
- 5mm 版本: 60 层 × 512 × 512
- 1mm 版本: 300 层 × 512 × 512

训练时随机取：
- 从 5mm 随机取 4 层 + 256×256 区域
- 对应 1mm 的 16 层 + 同样 256×256 区域
```

#### 2. 验证/测试阶段 (`get_val_img`)

由于验证时需要计算整个卷的性能指标，采用**切块→推理→拼接**策略：

```python
def get_val_img(img_path, case_name):
    # 读取整卷
    hr = read(f'{img_path}/val/1mm/{case_name}.nii.gz')
    lr = read(f'{img_path}/val/5mm/{case_name}.nii.gz')
    
    # 滑动窗口切块 (避免显存溢出)
    # vc_z=4, vc_y=256, vc_x=256
    crops = []
    positions = []
    
    for z in range(0, lr.shape[0], vc_z-2):  # 重叠 2 层
        for y in range(0, 512, vc_y):
            for x in range(0, 512, vc_x):
                crop = lr[z:z+vc_z, y:y+vc_y, x:x+vc_x]
                crops.append(crop)
                positions.append((z, y, x))
    
    return crops, positions, hr  # 多个小块 + 位置 + 完整HR
```

**验证时的推理流程**：

```
┌─────────────────────────────────────┐
│  1. 读取 5mm 完整 CT 卷              │
├─────────────────────────────────────┤
│  2. 切成多个 4×256×256 小块           │
│     (重叠 2 层保证连续性)              │
├─────────────────────────────────────┤
│  3. 每个小块 → 网络 → 16×256×256     │
├─────────────────────────────────────┤
│  4. 按位置拼接回完整卷                │
│     (重叠区域取平均)                  │
├─────────────────────────────────────┤
│  5. 与 1mm GT 计算 PSNR/SSIM         │
└─────────────────────────────────────┘
```

## 四、完整的训练-验证流程示例

### 文件准备
```
# 确保数据路径正确
RPLHR-CT-main/
├── code/               # 代码目录
├── config/
│   ├── SRM_dict.json   # 配置: {"path_img": "../data/"}
│   └── default.txt     # 超参数配置
└── data/               # 数据目录 (需要解压 ZIP)
    ├── train/
    ├── val/
    └── test/
```

### 运行训练
```bash
cd RPLHR-CT-main/code

# 训练命令
python train.py train \
    --path_key SRM \      # 使用 config/SRM_dict.json
    --gpu_idx 0 \         # GPU 编号
    --net_idx TVSRN      # 网络名称
```

### 代码执行流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 配置加载                                            │
│  ├── 读取 config/SRM_dict.json → 得到 path_img="../data/"    │
│  └── 读取 config/default.txt → 得到训练参数                   │
├─────────────────────────────────────────────────────────────┤
│  Step 2: 数据集构建                                           │
│  ├── 扫描 train/1mm/ 目录 → 获取病例列表                       │
│  │   train_list = ['CT00000000', 'CT00000001', ...]         │
│  ├── 扫描 val/1mm/ 目录 → 获取验证列表                         │
│  │   val_list = ['CT0000100', 'CT0000101', ...]             │
│  └── 创建 Dataset 和 DataLoader                              │
├─────────────────────────────────────────────────────────────┤
│  Step 3: 训练循环                                             │
│  for epoch in range(epochs):                                 │
│      for batch in train_loader:                              │
│          case_name, x, y = batch    # x=5mm块, y=1mm块        │
│          pred = model(x)                                     │
│          loss = L1Loss(pred, y)                              │
│          loss.backward()                                     │
│          optimizer.step()                                    │
├─────────────────────────────────────────────────────────────┤
│  Step 4: 验证 (每 N 个 epoch)                                │
│  for batch in val_loader:                                    │
│      case_name, crops, hr_gt, positions = batch              │
│      for crop, pos in zip(crops, positions):                │
│          pred_crop = model(crop)                            │
│          按位置拼接 pred_crop → 完整预测卷                     │
│      psnr = cal_psnr(pred_volume, hr_gt)                    │
├─────────────────────────────────────────────────────────────┤
│  Step 5: 保存最佳模型                                         │
│  if psnr > best_psnr:                                        │
│      save_checkpoint()                                       │
└─────────────────────────────────────────────────────────────┘
```

## 五、关键参数说明

### 数据相关参数 (config/default.txt)

```python
# 数据裁剪尺寸 (训练时)
c_z = 4       # 5mm 方向裁剪层数
c_y = 256     # Y 方向裁剪大小
c_x = 256     # X 方向裁剪大小

# 验证裁剪尺寸 (验证/测试时)
vc_z = 4      # 5mm 方向裁剪层数
vc_y = 256    # Y 方向裁剪大小
vc_x = 256    # X 方向裁剪大小

# 上采样倍数
ratio = 5     # 5mm → 1mm 是 5 倍

# 数据路径 (从 SRM_dict.json 读取)
path_img = "../data/"
```

### 尺寸对应关系

| 描述 | 5mm (输入) | 1mm (标签/输出) |
|-----|-----------|----------------|
| 训练裁剪 | 4×256×256 | 16×256×256 |
| 验证裁剪 | 4×256×256 | 16×256×256 |
| 原始卷大小 | Z×512×512 | (Z×5)×512×512 |

**注意**：5mm → 1mm 的上采样只在 **Z 方向**（层厚方向）进行，X/Y 方向保持 512 不变。

## 六、常见问题

### Q1: 为什么训练用随机裁剪，验证用滑动窗口？
**A**: 
- 训练随机裁剪可以增加数据多样性，防止过拟合
- 验证需要完整推理结果来计算 PSNR/SSIM，但整卷太大无法一次性送入网络，所以切块处理后拼接

### Q2: 验证时的重叠（overlap）是什么？
**A**: 
- 代码中 `vc_z - 2 = 2` 表示 Z 方向重叠 2 层
- 这是为了保证拼接处的连续性，避免边界效应
- 拼接时会自动处理重叠区域

### Q3: 可以自己修改裁剪尺寸吗？
**A**: 
- 可以，但需要同时修改 `c_z/c_y/c_x` 和 `vc_z/vc_y/vc_x`
- 注意保持 5:1 的 Z 方向比例（c_z × 5 = 输出 Z 维度）
- 需要确保网络支持对应的输入尺寸

### Q4: 如何添加数据增强？
**A**: 
- 代码中已有简单的镜像增强 (`opt.mirror`)
- 可以在 `get_train_img` 函数中添加其他增强，如旋转、噪声等
