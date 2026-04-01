# RPLHR-CT Ratio=4 架构适配方案

> **目标**：不使用插值，通过修改模型架构适配1:4数据集（宣武数据集）  
> **要求**：保留原有1:5模型，用于其他数据集  
> **预期**：Val PSNR > 25dB (理想 > 30dB)

---

## 一、问题诊断

### 1.1 当前问题核心

| 参数 | 当前值 | 真实值 | 问题描述 |
|------|--------|--------|----------|
| `c_z` | 4 | 4 | LR输入层数 |
| `ratio` | 4 | 4 | 上采样比例 |
| `out_z` | 13 | - | 模型计算 `(c_z-1)*ratio+1` |
| 模型输出 | **7层** | **16层** | 裁剪 `[:,:,3:-3]` 后 |
| 真实HR | 16层 | 16层 | 4层输入 × 4比例 |

### 1.2 当前代码的问题

**`in_model_xuanwu.py` 训练时：**
```python
# 错误做法：将16层HR插值到7层
hr_target_z = _get_hr_target_shape(opt.c_z, ratio)  # = 7
if crop_mask.shape[0] != hr_target_z:
    crop_mask = _interpolate_to_shape(crop_mask, target_shape, order=3)
```

**`trainxuanwu.py` 验证时：**
```python
# 错误做法：将y从16层插值到模型输出尺寸
if y_pre.shape[0] != y.shape[0]:
    y = zoom(y, zoom_factors, order=3)  # 插值破坏真实HR
```

### 1.3 结果
- Val PSNR 仅 ~10 dB（正常应 > 25 dB）
- 模型学习的是 thick↔插值后的thin 映射，而非真实 thick↔thin

---

## 二、解决方案：架构适配（无插值）

### 2.1 核心思路

**目标：让模型直接输出16层，匹配真实HR尺寸**

计算公式：
```
out_z = (c_z - 1) * ratio + 1
target_output = out_z - 裁剪层数 = 16
```

**最优参数组合：**
- `c_z = 6` （LR输入从4层改为6层）
- `ratio = 4` （保持1:4比例）
- `out_z = (6-1)*4+1 = 21`
- **裁剪策略**：`[:, :, 2:-3]` （裁剪5层，输出16层）✅

### 2.2 Window Size 兼容性验证

| 配置 | c_z | ratio | out_z | TD_Tw | window_size | 整除验证 |
|------|-----|-------|-------|-------|-------------|----------|
| 1:5模型 | 4 | 5 | 16 | 4 | 4 | 16%4=0 ✅ |
| 1:4模型 | 6 | 4 | 21 | 1 | 1 | 21%1=0 ✅ |

---

## 三、具体实施步骤

### 步骤1：创建1:4专用配置文件

**文件：`config/xuanwu_ratio4.txt`**
```ini
########## spec_config ##########

### 宣武数据集1:4配置 (无插值架构适配)
net_idx = xuanwu_ratio4
path_key = dataset01_xuanwu

# loss
loss_f = 'L1'

### data
dim = 1
ratio = 4

### Task based model design
### TVSRN
# global config
T_mlp = 4
T_pos = True

# Encoder config
TE_c = 8
TE_l = 1
TE_d = 4
TE_n = 8
TE_w = 8
TE_p = 8

# Decoder config
TD_p = 8
TD_s = 1

TD_Tw = 1
TD_Tl = 1
TD_Td = 4

TD_Iw = 8
TD_Il = 2
TD_Id = 4
TD_n = 8

########## common_config ##########
**** hardware config ****
gpu_idx = 0
*************************

**** mode ****
mode = 'train'
***********************************

**** 网络部分 ****
pre_train = False

***********************************

**** train & val config ****
# train set
epoch = 2000
start_epoch = 1
gap_epoch = 200

# dataloader set
train_bs = 1
num_workers = 4
val_bs = 1
test_num_workers = 4
***********************************

**** data config ****
# 关键修改：c_z从4改为6，让模型输出16层
c_z = 6
c_y = 256
c_x = 256

v_crop = True
vc_z = 6
vc_y = 256
vc_x = 256

mirror = False
***********************************

**** optimizer config ****
optim = 'AdamW'
wd = 0.0001
lr = 0.0003
flood = False
gap_val = 5

# 学习率策略
patience = 15
cos_lr = False
Tmax = 20
lr_gap = 1000
cycle_r = False
Tmin = False
***********************************

**** model config ****
save_log = False
***********************************
```

### 步骤2：修改模型裁剪逻辑

**文件：`code/net/model_TransSR.py`**

修改 `forward` 函数末尾的裁剪逻辑：

```python
def forward(self, x):
    x = x.squeeze().unsqueeze(0)

    # Encoder (原有代码保持不变)
    x_patch = einops.rearrange(x, 'B C (nH hp) (nW wp) -> B (C hp wp) nH nW', wp=self.E_patch, hp=self.E_patch)
    x_LP = self.LP(x_patch)
    x_SF = einops.rearrange(x_LP, 'B (C hp wp) nH nW -> B C (nH hp) (nW wp)', wp=self.E_patch, hp=self.E_patch)
    x_Eout = self.Encoder.forward_features(x_SF) + x_SF

    # Token (原有代码保持不变)
    x_patch_vis = x_Eout.reshape(-1, self.c, opt.c_y, opt.c_x)
    x_patch_embed = torch.cat([x_patch_vis, self.x_patch_mask], dim=0)
    x_patch_embed = x_patch_embed[self.slice_sequence]

    if opt.T_pos:
        trans_input = x_patch_embed + self.positions_z
    else:
        trans_input = x_patch_embed

    # Decoder (原有代码保持不变)
    trans_feature = trans_input.reshape(1, -1, opt.c_y, opt.c_x)
    for i in range(1, opt.TD_s + 1):
        trans_feature = eval('self.cal_xy(trans_feature, self.Decoder_T%s)' % i) + trans_feature
        trans_feature = eval('self.cal_z(trans_feature, self.Decoder_I%s)' % i) + trans_feature

    trans_output = trans_feature + trans_input.reshape(1, -1, opt.c_y, opt.c_x)
    trans_output = trans_output.reshape(1, self.c, -1, opt.c_y, opt.c_x)
    x_out = self.conv_last(self.conv_before_upsample(trans_output))

    # ========== 修改裁剪逻辑 ==========
    # 根据配置动态决定裁剪层数
    output_depth = x_out.shape[2]
    target_ratio = getattr(opt, 'ratio', 5)
    target_cz = getattr(opt, 'c_z', 4)
    
    # 计算期望的HR深度
    expected_hr_depth = target_cz * target_ratio  # 如 c_z=6, ratio=4 → 24层原始HR
    
    # 对于宣武数据集，我们需要输出16层（从24层中选取有效区域）
    if target_ratio == 4 and target_cz == 6:
        # c_z=6, ratio=4: out_z=21, 裁剪5层 → 16层
        return x_out[:, :, 2:-3]  # 裁剪5层
    else:
        # 默认：裁剪6层
        return x_out[:, :, 3:-3]
```

### 步骤3：移除训练数据加载的插值

**文件：`code/utils/in_model_xuanwu.py`**

修改 `get_train_img` 函数：

```python
def get_train_img(img_path, case_name):
    """
    加载宣武数据集训练数据（无插值版本）
    """
    case_mask_path = os.path.join(img_path, 'train', 'thin', case_name + '.nii.gz')
    tmp_mask = sitk.GetArrayFromImage(sitk.ReadImage(case_mask_path))

    case_img_path = os.path.join(img_path, 'train', 'thick', case_name + '.nii.gz')
    tmp_img = sitk.GetArrayFromImage(sitk.ReadImage(case_img_path))

    # 使用配置中的ratio和c_z
    ratio = opt.ratio
    c_z = opt.c_z

    z = tmp_img.shape[0]
    z_s = random.randint(0, z - 1 - c_z)
    y_s = random.randint(0, 512 - opt.c_y)
    x_s = random.randint(0, 512 - opt.c_x)
    z_e = z_s + c_z
    y_e = y_s + opt.c_y
    x_e = x_s + opt.c_x

    # 裁剪LR (thick)
    crop_img = tmp_img[z_s:z_e, y_s:y_e, x_s:x_e]

    # 裁剪HR (thin) - 直接使用真实比例
    thin_z_s = z_s * ratio
    thin_z_e = thin_z_s + c_z * ratio  # 如 c_z=6 → 24层
    crop_mask = tmp_mask[thin_z_s:thin_z_e, y_s:y_e, x_s:x_e]

    # 移除：不再进行插值！
    # 模型应该直接学习 thick(6层) → thin(24层中有效区域)
    # 注意：模型输出将是16层（裁剪后），需要在损失计算时对齐

    # 保留镜像增强
    if opt.mirror and np.random.uniform() <= 0.3:
        crop_img = crop_img[:, :, ::-1].copy()
        crop_mask = crop_mask[:, :, ::-1].copy()
    
    # 数据归一化
    if hasattr(opt, 'normalize_ct') and opt.normalize_ct:
        crop_img, crop_mask = normalize_ct(
            crop_img, crop_mask,
            window_center=getattr(opt, 'window_center', 40),
            window_width=getattr(opt, 'window_width', 400)
        )
    
    # 数据增强
    if hasattr(opt, 'use_augmentation') and opt.use_augmentation:
        aug = CTVolumetricAugmentation(
            prob=opt.aug_prob if hasattr(opt, 'aug_prob') else 0.5
        )
        aug_config = getattr(opt, 'aug_config', GEOMETRY_ONLY_AUG)
        crop_img, crop_mask = aug.apply_train_augmentation(
            crop_img, crop_mask, aug_config=aug_config
        )

    return crop_img, crop_mask
```

**注意**：这里 `crop_mask` 是24层，但模型输出是16层。需要在训练脚本中对齐。

### 步骤4：修改训练脚本处理尺寸对齐

**文件：`code/trainxuanwu.py`**

修改训练时的损失计算：

```python
for i, return_list in tqdm(enumerate(train_batch)):
    case_name, x, y = return_list
    x = x.float().to(device, non_blocking=True)
    y = y.float().to(device, non_blocking=True)

    y_pre = net(x)  # 输出是16层
    
    # 对齐y的维度：如果y是24层，取中间16层
    if y.shape[2] != y_pre.shape[2]:
        # y: [B, 1, 24, 256, 256], y_pre: [B, 1, 16, 256, 256]
        offset = (y.shape[2] - y_pre.shape[2]) // 2
        y = y[:, :, offset:offset+y_pre.shape[2], :, :]
    
    loss = train_criterion(y_pre, y)
    # ... 后续代码不变
```

修改验证时的PSNR计算（移除插值）：

```python
for i, return_list in tqdm(enumerate(val_batch)):
    case_name, x, y, pos_list = return_list
    case_name = case_name[0]
    x = x.squeeze().data.numpy()
    y = y.squeeze().data.numpy()  # 原始HR，保持16层或24层

    if e == 0 and i == 0:
        print('thin size:', y.shape)

    # 初始化y_pre，大小应与模型输出匹配
    # 注意：这里需要根据pos_list计算实际覆盖范围
    y_pre = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype=np.float32)
    
    pos_list = pos_list.data.numpy()[0]

    for pos_idx, pos in enumerate(pos_list):
        tmp_x = x[pos_idx]
        tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

        tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
        tmp_y_pre = net(tmp_x)  # 输出16层
        tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
        y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

        D = y_for_psnr.shape[0]  # 16层
        ratio = opt.ratio  # 4
        pos_z_s = ratio * tmp_pos_z + 3  # 与模型内部计算一致
        pos_y_s = tmp_pos_y
        pos_x_s = tmp_pos_x

        y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

    del tmp_y_pre, tmp_x

    # 移除：不再对y进行插值！
    # 直接比较y_pre和y的重叠区域
    
    # 计算重叠区域
    min_z = min(y_pre.shape[0], y.shape[0])
    y_pre_eval = y_pre[5:min_z-5]  # 裁剪边界
    y_eval = y[5:min_z-5]
    
    # 确保尺寸匹配
    if y_pre_eval.shape[0] != y_eval.shape[0]:
        # 如果仍有差异，取较短的长度
        min_len = min(y_pre_eval.shape[0], y_eval.shape[0])
        y_pre_eval = y_pre_eval[:min_len]
        y_eval = y_eval[:min_len]
    
    psnr = non_model.cal_psnr(y_pre_eval, y_eval)
    mse = non_model.cal_mse(y_pre_eval, y_eval)
    
    # SSIM计算
    pid_ssim_list = []
    for z_idx, z_layer in enumerate(y_pre_eval):
        mask_layer = y_eval[z_idx]
        tmp_ssim = non_model.cal_ssim(mask_layer, z_layer, device=device)
        pid_ssim_list.append(tmp_ssim)
    ssim_val = np.mean(pid_ssim_list)
    
    psnr_list.append(psnr)
    mse_list.append(mse)
    ssim_list.append(ssim_val)
```

---

## 四、验证清单

### 4.1 维度验证

```python
# 测试代码：验证各层维度
import torch
from net import model_TransSR
from config import opt

# 加载1:4配置
opt.load_config('../config/xuanwu_ratio4.txt')
opt.c_z = 6
opt.ratio = 4

# 创建模型
net = model_TransSR.TVSRN()

# 测试输入：6层LR
x = torch.randn(1, 1, 6, 256, 256)
out = net(x)

print(f"输入LR: {x.shape}")      # [1, 1, 6, 256, 256]
print(f"输出HR: {out.shape}")    # [1, 1, 16, 256, 256] ✅

# 验证out_z计算
out_z = (opt.c_z - 1) * opt.ratio + 1  # 21
print(f"内部out_z: {out_z}")      # 21
print(f"裁剪后: {out_z - 5}")     # 16
```

### 4.2 训练前检查

```bash
# 1. 检查配置文件
python -c "from config import opt; opt.load_config('../config/xuanwu_ratio4.txt'); print('c_z:', opt.c_z, 'ratio:', opt.ratio)"

# 2. 检查数据加载
python -c "from make_dataset_xuanwu import train_Dataset; ds = train_Dataset(['HCTSR-0001']); x, y = ds[0]; print('LR:', x.shape, 'HR:', y.shape)"
# 预期输出：LR: (6, 256, 256) HR: (24, 256, 256)

# 3. 检查模型输出
python -c "import torch; from net import model_TransSR; from config import opt; opt.c_z=6; opt.ratio=4; net = model_TransSR.TVSRN(); out = net(torch.randn(1,1,6,256,256)); print('输出:', out.shape)"
# 预期输出：[1, 1, 16, 256, 256]
```

---

## 五、训练命令

```bash
# 使用新配置训练
cd /root/autodl-tmp/rplhr-ct-training-main/code

python trainxuanwu.py train \
    --net_idx=xuanwu_ratio4 \
    --path_key=dataset01_xuanwu \
    --config=../config/xuanwu_ratio4.txt \
    --epoch=50 \
    --use_augmentation=True \
    --aug_prob=0.5 \
    --normalize_ct=True \
    --num_workers=4 \
    --test_num_workers=2
```

---

## 六、预期结果

| 指标 | 当前（插值方案） | 目标（架构适配） |
|------|-----------------|-----------------|
| Val PSNR | ~10 dB | **> 25 dB** |
| Val SSIM | ~0.6 | **> 0.8** |
| 训练稳定性 | 差 | 好 |
| 输出尺寸 | 7层（插值） | 16层（直接） |

---

## 七、保留1:5模型

原有配置完全保留：
- `config/default.txt` - 1:5配置（c_z=4, ratio=5）
- 训练命令：`--net_idx=xuanwu_50epoch` 使用原配置

两个模型可以共存，通过 `net_idx` 区分。

---

## 八、故障排查

### 问题1：RuntimeError - window_size不整除
**解决**：确保 `TD_Tw=1` 当使用 c_z=6, ratio=4

### 问题2：尺寸不匹配错误
**检查**：
1. 确认 `in_model_xuanwu.py` 的 `vc_z = 6`
2. 确认 `trainxuanwu.py` 中的对齐逻辑正确

### 问题3：PSNR仍然很低
**检查**：
1. 确认移除了所有 `zoom`/`interpolate` 调用
2. 确认模型输出确实是16层
3. 检查HR数据是否被正确归一化

---

*文档版本：v1.0*  
*创建时间：2026-04-01*  
*适用数据集：宣武数据集 (1.25mm:5mm, ratio=4)*
