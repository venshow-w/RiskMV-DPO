
# filename: models/localdpo/vggt_scorer.py
# FIXED VERSION:
#   Bug #1 修复: forward() 末尾 drop_cond_mask 处理代码补全（CFG null latent 混合）
#   改进:        geometry_latents 支持按需 detach（用于对齐损失时不 detach）

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VGGTGeometryAdapter(nn.Module):
    """
    Gen3R 风格的几何 latent 生成器（FIXED）

    架构：
        VGGT(冻结) → image tokens → Linear Proj → Transformer 压缩
        → geometry_latents [B*NC, num_tokens, latent_dim]

    修复说明：
        - 补全了 drop_cond_mask 的 CFG null latent 混合逻辑
        - 添加 null_geo_token 参数（用于 classifier-free guidance）
    """

    def __init__(
        self,
        vggt_model,
        vggt_feat_dim: int = 3072,
        latent_dim: int = 1152,
        num_tokens: int = 16,
        num_heads: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()

        # 冻结 VGGT
        self.vggt = vggt_model
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False

        # 1. 特征投影：VGGT patch 特征 → latent 空间
        self.proj = nn.Sequential(
            nn.Linear(vggt_feat_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # 2. 可学习的几何 query tokens
        self.geo_queries = nn.Parameter(
            torch.randn(1, num_tokens, latent_dim) * 0.02
        )

        # 3. Transformer 压缩器（cross-attention 风格：query + patch_tokens → query）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.compressor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出 LayerNorm
        self.norm_out = nn.LayerNorm(latent_dim)

        # ===== FIXED: 添加 null_geo_token（用于 CFG drop）=====
        self.null_geo_token = nn.Parameter(
            torch.zeros(1, num_tokens, latent_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.geo_queries, mean=0.0, std=0.02)
        nn.init.zeros_(self.null_geo_token)

    def extract_vggt_features(self, first_frame_images: torch.Tensor) -> torch.Tensor:
        """
        提取 VGGT patch 特征。

        Args:
            first_frame_images: [B, T, NC, C, H, W]（T 通常为 1，即首帧）

        Returns:
            features: [(B*NC), P, vggt_feat_dim]  P 为 patch 数量
        """
        B, T, NC, C, H, W = first_frame_images.shape

        with torch.no_grad():
            # 重排为 VGGT 期望的格式：[B, (NC*T), C, H, W]
            images = rearrange(first_frame_images, "B T NC C H W -> B (NC T) C H W")

            # VGGT 前向
            (
                aggregated_tokens_list,
                image_tokens_list,
                dino_token_list,
                image_feature,
                patch_start_idx,
            ) = self.vggt.aggregator(images)

        # 取最后一层的 patch tokens，shape: [B, NC*T, P, D]
        features = image_tokens_list[-1][:, :, patch_start_idx:, :]

        # 展平 batch 和相机维度：[B*NC, P, D]
        features = rearrange(features, "B NC P C -> (B NC) P C")
        return features

    def forward(
        self,
        first_frame_images: torch.Tensor,
        drop_cond_mask: torch.Tensor = None,
        T: int = None,
        NC: int = None,
    ) -> torch.Tensor:
        """
        生成几何 latent。

        Args:
            first_frame_images: [B, T_in, NC_in, C, H, W]
            drop_cond_mask:     [B] float tensor，0 表示该样本丢弃几何条件（CFG）
            T:                  视频总帧数（保留接口，暂未使用）
            NC:                 相机数（可覆盖 NC_in）

        Returns:
            geometry_latents: [B*NC, num_tokens, latent_dim]
        """
        B, T_in, NC_in, C, H, W = first_frame_images.shape
        NC_actual = NC if NC is not None else NC_in

        # 1. 提取 VGGT 特征 [(B*NC), P, vggt_feat_dim]
        vggt_features = self.extract_vggt_features(first_frame_images)

        # 2. 投影到 latent 空间 [(B*NC), P, latent_dim]
        geo_tokens = self.proj(vggt_features)

        # 3. Transformer 压缩（query tokens + patch tokens → query tokens）
        queries = self.geo_queries.expand(B * NC_actual, -1, -1)    # [B*NC, num_tokens, D]
        combined = torch.cat([queries, geo_tokens], dim=1)           # [B*NC, num_tokens+P, D]
        encoded = self.compressor(combined)
        geometry_latents = encoded[:, : self.geo_queries.shape[1], :]  # [B*NC, num_tokens, D]
        geometry_latents = self.norm_out(geometry_latents)

        # ===== FIXED: 补全 CFG null latent 混合逻辑 =====
        if drop_cond_mask is not None:
            if drop_cond_mask.dim() != 1 or drop_cond_mask.shape[0] != B:
                raise ValueError(
                    f"drop_cond_mask should be [B]={B}, got {drop_cond_mask.shape}"
                )
            # [B] → [B*NC, 1, 1] 用于广播
            mask = (
                drop_cond_mask.view(B, 1)
                .expand(B, NC_actual)
                .reshape(B * NC_actual, 1, 1)
            )  # [B*NC, 1, 1]

            # null latent：扩展到当前 batch 大小
            null_latents = self.null_geo_token.expand(B * NC_actual, -1, -1)  # [B*NC, num_tokens, D]
            breakpoint()
            # 按 mask 混合：mask=1 保留几何条件，mask=0 替换为 null
            geometry_latents = mask * geometry_latents + (1.0 - mask) * null_latents

        return geometry_latents


class VGGTGeoScorer(nn.Module):
    """
    几何一致性评分器（复用已有 VGGTFusionBlockV2 参数）。
    将融合模块改造为输出 0-1 分数。
    """

    def __init__(self, original_fusion_block):
        super().__init__()

        # 复用原有模块参数（只冻结，不复制）
        self.cross_attn = original_fusion_block.cross_attn
        self.norm_q = original_fusion_block.norm_q
        self.norm_kv = original_fusion_block.norm_kv
        self.vggt_proj = original_fusion_block.vggt_proj
        hidden_size = original_fusion_block.hidden_size

        # 新增：分数投影层（可训练）
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

        # 初始化分数投影层（小增益，避免初始输出极端）
        for m in self.score_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 冻结复用的参数
        for module in [self.cross_attn, self.norm_q, self.norm_kv, self.vggt_proj]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        video_features: torch.Tensor,
        vggt_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_features: [B, N, C]  生成视频的特征
            vggt_features:  [B, M, C]  VGGT 几何特征

        Returns:
            score: [B, 1]  几何一致性分数，范围 [0, 1]
        """
        video_norm = self.norm_q(video_features)
        vggt_norm = self.norm_kv(self.vggt_proj(vggt_features))
        attn_out = self.cross_attn(video_norm, vggt_norm, mask=None)
        pooled = attn_out.mean(dim=1)      # [B, C]
        score = self.score_proj(pooled)    # [B, 1]
        return torch.sigmoid(score)