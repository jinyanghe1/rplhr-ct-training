"""
Z-Axis Attention and Residual Dense Connection modules for TVSRN.
Designed for 3D CT super-resolution: enhances z-direction modeling.

Module 1: ZAxisAttention - Cross-attention along z-axis between visible and mask slices
Module 2: ResidualDenseBlock - Dense feature reuse with residual connections
Module 3: ZEnhancedDecoderBlock - Integrates both into the decoder pipeline

v2: Fixed ratio-aware support - RDC uses max_out_z channels, z_attn uses actual out_z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_


class ZAxisAttention(nn.Module):
    """
    Lightweight cross-attention along z-axis.

    Architecture:
    - All slices (out_z) attend to each other via compressed representations
    - Uses spatial pooling to get one global vector per slice
    - Adds relative z-position bias for distance-aware attention

    Key design: channel compression keeps params low (~5K params per head)
    """
    def __init__(self, c, num_heads=4, reduction=4):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        self.reduction = reduction

        # All projections use compressed dim: c // reduction
        self.comp_dim = max(c // reduction, num_heads)
        self.head_dim = self.comp_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Channel compression for efficiency
        self.q_proj = nn.Linear(c, self.comp_dim)
        self.k_proj = nn.Linear(c, self.comp_dim)
        self.v_proj = nn.Linear(c, self.comp_dim)
        self.out_proj = nn.Linear(self.comp_dim, c)

        # Learnable relative z-position bias
        self.max_z = 32  # max out_z we expect
        self.rel_z_bias = nn.Parameter(torch.zeros(1, num_heads, 1, self.max_z * 2 + 1))
        trunc_normal_(self.rel_z_bias, std=.02)

        self.norm = nn.LayerNorm(c)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable residual scale, init=0

    def forward(self, x, out_z):
        """
        Args:
            x: [out_z, c, c_y, c_x] - all slice features
            out_z: actual out_z (may vary with ratio)
        Returns:
            enhanced features [out_z, c, c_y, c_x]
        """
        Bz, C, H, W = x.shape
        N = H * W  # spatial tokens per slice

        # Normalize per-slice features
        x_flat = x.permute(0, 2, 3, 1).reshape(Bz, N, C)
        x_normed = self.norm(x_flat)

        # Compute attention using spatially-pooled features (global z attention)
        x_pooled = x_normed.mean(dim=1)  # [out_z, C]

        q = self.q_proj(x_pooled)
        k = self.k_proj(x_pooled)
        v = self.v_proj(x_pooled)

        # Reshape for multi-head attention
        q = q.view(Bz, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
        k = k.view(Bz, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
        v = v.view(Bz, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative z-position bias
        z_range = torch.arange(Bz, device=x.device)
        rel_z = z_range.unsqueeze(0) - z_range.unsqueeze(1)
        rel_z_idx = (rel_z + self.max_z).clamp(0, 2 * self.max_z).long()
        bias = self.rel_z_bias[:, :, 0, rel_z_idx]
        attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).squeeze(0).reshape(Bz, self.comp_dim)
        out = self.out_proj(out)

        # Residual: broadcast attention output to spatial dims
        out = out.unsqueeze(1).unsqueeze(1)  # [Bz, 1, 1, C]
        x_enhanced = x.permute(0, 2, 3, 1) + self.gamma * out
        x_enhanced = x_enhanced.permute(0, 3, 1, 2)

        return x_enhanced


class ResidualDenseBlock(nn.Module):
    """
    Lightweight residual dense block for z-direction feature refinement.
    Uses 1x1 convolutions + LeakyReLU to reuse features from each decoder layer.
    """
    def __init__(self, in_channels, growth_rate=16, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate

        layers = []
        for i in range(num_layers):
            in_ch = in_channels + i * growth_rate
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, growth_rate, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.dense_layers = nn.ModuleList(layers)

        # Final fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        features = [x]
        for layer in self.dense_layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = self.fuse(torch.cat(features, dim=1))
        return x + self.scale * out


class ZEnhancedDecoderBlock(nn.Module):
    """
    Combined z-axis attention + residual dense connection wrapper.
    v2: Supports ratio-aware operation with padding for weight compatibility.
    """
    def __init__(self, c, max_out_z, out_z_channels, num_heads=4, growth_rate=16, rdc_layers=3):
        super().__init__()
        self.c = c
        self.max_out_z = max_out_z
        self.z_attn = ZAxisAttention(c, num_heads=num_heads)
        # RDC always operates on max_out_z * c channels (weight compatibility)
        self.rdc = ResidualDenseBlock(out_z_channels, growth_rate=growth_rate, num_layers=rdc_layers)

    def forward(self, trans_feature, out_z):
        """
        Args:
            trans_feature: [1, out_z * c, c_y, c_x]
            out_z: actual output z dimension (may be less than max_out_z)
        Returns:
            enhanced features [1, out_z * c, c_y, c_x]
        """
        B, C_total, H, W = trans_feature.shape
        c = C_total // out_z
        max_out_z = self.max_out_z
        max_C_total = max_out_z * c
        
        # --- Z-axis attention (operates on per-slice features) ---
        x_slices = trans_feature.reshape(out_z, c, H, W)
        x_slices = self.z_attn(x_slices, out_z)
        x_z = x_slices.reshape(B, C_total, H, W)
        
        # --- Residual dense connection (needs max_out_z * c channels) ---
        if out_z < max_out_z:
            # Pad trans_feature to max_out_z channels for RDC compatibility
            pad_channels = max_C_total - C_total
            padding = torch.zeros(B, pad_channels, H, W, device=trans_feature.device)
            x_padded = torch.cat([trans_feature, padding], dim=1)  # [1, max_out_z*c, H, W]
            x_rdc_full = self.rdc(x_padded)  # [1, max_out_z*c, H, W]
            # Extract only the out_z*c channels
            x_rdc = x_rdc_full[:, :C_total, :, :]  # [1, out_z*c, H, W]
        else:
            x_rdc = self.rdc(trans_feature)

        # Combine z_attn + rdc, avoiding double-counting original
        return x_z + x_rdc - trans_feature
