import torch
import torch.nn as nn
from einops import rearrange

class DGGTToDiTAdapter(nn.Module):
    def __init__(self, g_channels=128, dit_hidden_size=1152): # 1152 是 DiT-XL 的典型维度
        super().__init__()
        # 处理 (B, NC, T, C, H, W) -> 映射到隐藏层维度
        self.conv_in = nn.Sequential(
            nn.Conv3d(g_channels + 1, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, dit_hidden_size, kernel_size=1)
        )

    def forward(self, depth, g_features):
        # x: [B, NC, T, C, H, W]
        x = torch.cat([depth, g_features], dim=2) # 假设深度和特征已对齐
        x = self.conv_in(x)
        return x # 返回与 DiT 空间维度对齐的几何特征