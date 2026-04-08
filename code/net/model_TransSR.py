# full assembly of the sub-parts to form the complete net
# Ratio-Aware version: out_z, slice_sequence, positional encoding 动态计算

import torch.nn.functional as F
import math
import torch
import einops
from timm.models.layers import trunc_normal_
from .swin_utils import *

def positionalencoding1d(d_model, length, ratio):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :param ratio: scaling ratio (used in position encoding)
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1) * ratio
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class TVSRN(nn.Module):
    def __init__(self):
        super(TVSRN, self).__init__()

        ####################### Global config #######################
        img_size = opt.c_y
        mlp_ratio = opt.T_mlp

        ###################### Linear Projection ####################
        self.E_patch = opt.TE_p
        init_C = self.E_patch * self.E_patch * opt.c_z

        E_num_in_ch = opt.c_z
        E_num_out_ch = opt.TE_c

        E_embed_dim = E_num_out_ch * E_num_in_ch

        self.LP = nn.Conv2d(init_C, E_embed_dim * self.E_patch * self.E_patch, 1, 1, 0)

        # region
        ######################## MAE Encoder ########################
        E_win = opt.TE_w
        E_depths = [opt.TE_d] * opt.TE_l
        E_num_heads = [opt.TE_n] * opt.TE_l

        self.Encoder = Swin_backbone_unConv(img_size=img_size, in_chans=E_num_in_ch, embed_dim=E_embed_dim,
                                     depths=E_depths, num_heads=E_num_heads, window_size=E_win,
                                     mlp_ratio=mlp_ratio)
        # endregion

        # region
        ######################## MAE MToken ########################
        # Ratio-Aware: 保留最大尺寸的可学习 mask token 参数
        # 动态 out_z 在 forward 中根据 ratio 计算
        self.c = E_num_out_ch
        self.c_z = opt.c_z
        self.c_y = opt.c_y
        self.c_x = opt.c_x

        # 计算最大可能的 out_z (ratio=5 时最大)
        max_ratio = getattr(opt, 'max_ratio', 5)
        self.max_out_z = (opt.c_z - 1) * max_ratio + 1
        self.x_patch_mask = torch.nn.Parameter(
            torch.zeros(self.max_out_z - opt.c_z, self.c, opt.c_y, opt.c_x)
        )

        # positional encoding 和 slice_sequence 移到 forward 动态计算
        # endregion

        # region
        ######################## MAE Decoder ########################
        self.D_patch = opt.TD_p
        D_T_depths = [opt.TD_Td] * opt.TD_Tl
        D_T_num_heads = [opt.TD_n] * opt.TD_Tl

        D_I_depths = [opt.TD_Id] * opt.TD_Il
        D_I_num_heads = [opt.TD_n] * opt.TD_Il

        T_win = opt.TD_Tw
        I_win = opt.TD_Iw

        # 使用默认 ratio 初始化 Decoder (权重可在 forward 中被不同 out_z 使用)
        default_out_z = (opt.c_z - 1) * opt.ratio + 1

        for i in range(1, opt.TD_s+1):
            T_embed = self.c * self.D_patch
            exec('''self.Decoder_T%s = Swin_backbone_unConv(img_size=(default_out_z, img_size), embed_dim=T_embed,
                                         depths=D_T_depths, num_heads=D_T_num_heads, window_size=T_win,
                                         mlp_ratio=mlp_ratio)''' % (i))

            I_embed = self.c * default_out_z
            exec('''self.Decoder_I%s = Swin_backbone_unConv(img_size=img_size, embed_dim=I_embed,
                                            depths=D_I_depths, num_heads=D_I_num_heads, window_size=I_win,
                                            mlp_ratio=mlp_ratio)''' % (i))
        # endregion

        ######################## MAE Rec ########################
        self.conv_before_upsample = nn.Sequential(nn.Conv3d(self.c, 16, 1, 1, 0),
                                                  nn.LeakyReLU(inplace=True))
        self.conv_last = nn.Conv3d(16, 1, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_slice_sequence(self, c_z, out_z, ratio, device):
        """
        动态构建 slice sequence: 已知切片和 mask 切片的交错排列
        与 __init__ 中原始逻辑完全一致，只是移到 forward 中动态计算
        """
        slice_list = list(range(out_z))
        vis_list = slice_list[:c_z]
        mask_list = slice_list[c_z:]
        re_list = []

        while len(vis_list) != 0:
            if len(re_list) % ratio == 0:
                re_list.append(vis_list.pop(0))
            else:
                re_list.append(mask_list.pop(0))

        return torch.tensor(re_list, dtype=torch.long, device=device)

    def cal_z(self, x, D):
        x_in = x.reshape(1, -1, opt.c_y, opt.c_x)
        x_out = D.forward_features(x_in)
        return x_out

    def cal_xy(self, x, D):
        x_in = x.reshape(-1, self.c, opt.c_y, opt.c_x)

        x_in_sag = einops.rearrange(x_in, 'dn c h (wn wp) -> wn (c wp) dn h', wp=self.D_patch)
        x_out_sag = D.forward_features(x_in_sag)
        x_out_sag = einops.rearrange(x_out_sag, 'wn (c wp) dn h -> dn c h (wn wp)', wp=self.D_patch)

        x_in_cor = einops.rearrange(x_in, 'dn c (hn hp) w -> hn (c hp) dn w', hp=self.D_patch)
        x_out_cor = D.forward_features(x_in_cor)
        x_out_cor = einops.rearrange(x_out_cor, 'hn (c hp) dn w -> dn c (hn hp) w', hp=self.D_patch)

        x_out = x_out_sag + x_out_cor
        x_out = x_out.reshape(1, -1, opt.c_y, opt.c_x)

        return x_out

    def forward(self, x, ratio=None):
        """
        Ratio-Aware forward
        Args:
            x: input tensor [B, 1, c_z, c_y, c_x]
            ratio: 超分倍率 (默认使用 opt.ratio)
        Returns:
            output tensor [1, 1, target_z, c_y, c_x]
        """
        if ratio is None:
            ratio = opt.ratio

        x = x.squeeze().unsqueeze(0)

        # ========== 动态计算 out_z ==========
        out_z = (self.c_z - 1) * ratio + 1

        # Encoder
        x_patch = einops.rearrange(x, 'B C (nH hp) (nW wp) -> B (C hp wp) nH nW', wp=self.E_patch, hp=self.E_patch)
        x_LP = self.LP(x_patch)
        x_SF = einops.rearrange(x_LP, 'B (C hp wp) nH nW -> B C (nH hp) (nW wp)', wp=self.E_patch, hp=self.E_patch)
        x_Eout = self.Encoder.forward_features(x_SF) + x_SF

        # ========== 动态 MToken ==========
        x_patch_vis = x_Eout.reshape(-1, self.c, opt.c_y, opt.c_x)

        # 动态截取 mask tokens (前 num_masks 个)
        num_masks = out_z - self.c_z
        x_patch_mask = self.x_patch_mask[:num_masks]

        x_patch_embed = torch.cat([x_patch_vis, x_patch_mask], dim=0)

        # 动态构建 slice sequence
        slice_sequence = self._build_slice_sequence(self.c_z, out_z, ratio, x.device)
        x_patch_embed = x_patch_embed[slice_sequence]

        # ========== 动态 Positional Encoding ==========
        if opt.T_pos:
            positions_z = positionalencoding1d(self.c, out_z, 1).unsqueeze(2).unsqueeze(2)
            positions_z = positions_z.to(x.device)
            trans_input = x_patch_embed + positions_z
        else:
            trans_input = x_patch_embed

        # Decoder
        trans_feature = trans_input.reshape(1, -1, opt.c_y, opt.c_x)
        for i in range(1, opt.TD_s + 1):
            trans_feature = eval('self.cal_xy(trans_feature, self.Decoder_T%s)' % i) + trans_feature
            trans_feature = eval('self.cal_z(trans_feature, self.Decoder_I%s)' % i) + trans_feature

        trans_output = trans_feature + trans_input.reshape(1, -1, opt.c_y, opt.c_x)
        trans_output = trans_output.reshape(1, self.c, -1, opt.c_y, opt.c_x)
        x_out = self.conv_last(self.conv_before_upsample(trans_output))

        # ========== 统一裁剪逻辑 ==========
        # 计算目标输出 z 维度
        # 对于 ratio=4, c_z=4: out_z=13, 中心裁剪后保留有效区域
        # 对于 ratio=5, c_z=4: out_z=16, 中心裁剪后保留有效区域
        # 裁剪策略: 去掉首尾的边界层 (边界效应)
        crop_start = out_z // 4   # 近似 1/4 作为起始裁剪
        crop_end = out_z - out_z // 4  # 近似 3/4 作为结束裁剪

        # 但保持与原始逻辑一致的裁剪方式
        # 原始: ratio=5,c_z=4 -> crop 3:-3 (crop 6 layers)
        # 原始: ratio=4,c_z=6 -> crop 2:-3 (crop 5 layers)
        # 通用化: crop_start 和 crop_end
        if ratio == 4 and self.c_z == 6:
            return x_out[:, :, 2:-3]
        elif ratio == 5 and self.c_z == 4:
            return x_out[:, :, 3:-3]
        else:
            # 通用裁剪: 对称裁剪边界
            margin = max(1, out_z // 8)
            return x_out[:, :, margin:-margin]
