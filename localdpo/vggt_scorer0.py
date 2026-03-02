# filename: models/localdpo/vggt_scorer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VGGTGeometryAdapter(nn.Module):
    """
    Gen3R风格：在VGGT token上训练适配器，生成几何潜在表示
    - 完全复用您已有的VGGT模型
    - 可学习的几何token压缩
    - 输出与MagicDrive hidden_size对齐的几何latent
    """
    def __init__(self,
                 vggt_model,
                 vggt_feat_dim=3072,
                 latent_dim=1152,
                 num_tokens=16,
                 num_heads=16,
                 num_layers=2):
        super().__init__()
        
        # 冻结VGGT
        self.vggt = vggt_model
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False
        
        # 1. 特征投影：VGGT特征 → 几何latent空间
        self.proj = nn.Sequential(
            nn.Linear(vggt_feat_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # 2. 可学习的几何query tokens
        self.geo_queries = nn.Parameter(
            torch.randn(1, num_tokens, latent_dim) * 0.02
        )
        
        # 3. Transformer压缩器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.compressor = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 4. 输出LayerNorm
        self.norm_out = nn.LayerNorm(latent_dim)

        # 5. 添加null token用于CFG 
        # self.null_geo_token = nn.Parameter(
        #     torch.zeros(1, num_tokens, latent_dim)
        # )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.geo_queries, mean=0.0, std=0.02)
        # nn.init.zeros_(self.null_geo_token)  # null token初始化为0
    
    def extract_vggt_features(self, first_frame_images):
        """复用您已有的VGGT特征提取逻辑"""
        B, T, NC, C, H, W = first_frame_images.shape
        
        with torch.no_grad():
            # 格式转换
            images = rearrange(first_frame_images, "B T NC C ... -> B (NC T) C ...")
            
            # VGGT前向
            aggregated_tokens_list, image_tokens_list, dino_token_list, image_feature, patch_start_idx = self.vggt.aggregator(images)
        # 取最后一层image_tokens的patch特征 features.shape torch.Size([1, 6, 1920, 3072]) 
        features = image_tokens_list[-1][:, :, patch_start_idx:, :]    
        features = rearrange(features, "B NC P C -> (B NC) P C")
        # 跨视图平均池化（利用多视图优势）
        
        # features = features.mean(dim=1)  # [B, P, D]
    
        return features
    
    def forward(self, first_frame_images, drop_cond_mask=None, T=None, NC=None):
        """
        输入: first_frame_images [B,T, NC, C, H, W]
        T: 视频总帧数（用于扩展）
        NC: 相机数
        输出: geometry_latents [B, num_tokens, latent_dim]
        """
        B, T_in, NC_in, C, H, W = first_frame_images.shape
        B_actual = B
        NC_actual = NC if NC is not None else NC_in
        T_actual = T if T is not None else 1
        
        # 1. 提取VGGT特征
        vggt_features = self.extract_vggt_features(first_frame_images) # (6.1920, 3072)
        
        # 2. 投影到latent空间
        geo_tokens = self.proj(vggt_features) # (6, 1920, 1152)
        # geo_tokens = rearrange(geo_tokens,"(B T NC) ... -> B (T NC)...", B=B, T=T, NC=NC)
        # 3. 可学习压缩
        queries = self.geo_queries.expand(B_actual * NC_actual, -1, -1) # queries shape (6,16,1152)
        
        combined = torch.cat([queries, geo_tokens], dim=1) # combineds.shape (6, 1936, 1152)
        encoded = self.compressor(combined) # encoded shape (6,1936,1152)
        geometry_latents = encoded[:, :self.geo_queries.shape[1], :]
        geometry_latents = self.norm_out(geometry_latents)
        
        # # 4. CFG处理
        # if drop_cond_mask is not None:
        #     mask = drop_cond_mask.view(B, 1, 1)
        #     null_latents = torch.zeros_like(geometry_latents)
        #     geometry_latents = mask * geometry_latents + (1 - mask) * null_latents
       
        if drop_cond_mask is not None:
            # drop_cond_mask 应该是 [B]
            if drop_cond_mask.dim() == 1:
                # [B] -> [B, 1, 1] 用于广播
                mask = drop_cond_mask.view(B_actual, 1, 1)
                # 扩展到相机维度
                mask = mask.expand(-1, NC_actual, -1)  # [B, NC, 1]
                # 展平为 [B*NC, 1]
                mask = mask.reshape(B_actual * NC_actual, 1, 1)  # [B*NC, 1, 1]
            else:
                raise ValueError(f"drop_cond_mask should be [B], got {drop_cond_mask.shape}")
            
            # 创建null latents
            # null_latents = self.null_geo_token.expand(B * NC, -1, -1)  # [B*NC, num_tokens, D]
            # null_latents = torch.zeros_like(geometry_latents).to(geometry_latents.device)
            # # 应用mask
            # geometry_latents = mask * geometry_latents + (1 - mask) * null_latents
        return geometry_latents


class VGGTGeoScorer(nn.Module):
    """
    将您的VGGTFusionBlockV2改造为几何评分器
    完全复用已有参数，只改forward返回值
    """
    def __init__(self, original_fusion_block):
        super().__init__()
        
        # 完全复用原有模块的参数
        self.cross_attn = original_fusion_block.cross_attn
        self.norm_q = original_fusion_block.norm_q
        self.norm_kv = original_fusion_block.norm_kv
        self.vggt_proj = original_fusion_block.vggt_proj
        
        # 新增：分数投影层（需单独训练）
        self.score_proj = nn.Sequential(
            nn.Linear(original_fusion_block.hidden_size, 
                     original_fusion_block.hidden_size // 4),
            nn.GELU(),
            nn.Linear(original_fusion_block.hidden_size // 4, 1)
        )
        
        # 初始化分数投影层
        for m in self.score_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 冻结原始融合模块参数（保持预训练特征）
        for param in self.cross_attn.parameters():
            param.requires_grad = False
        for param in self.norm_q.parameters():
            param.requires_grad = False
        for param in self.norm_kv.parameters():
            param.requires_grad = False
        for param in self.vggt_proj.parameters():
            param.requires_grad = False
    
    def forward(self, video_features, vggt_features):
        """
        输入：
            video_features: 生成视频的特征 [B, N, C]
            vggt_features: VGGT几何特征 [B, M, C]
        输出：
            几何一致性分数 [B, 1] 范围0-1
        """
        # 1. 特征归一化
        video_norm = self.norm_q(video_features)
        vggt_norm = self.norm_kv(self.vggt_proj(vggt_features))
        
        # 2. Cross Attention（不修改输入）
        attn_out = self.cross_attn(video_norm, vggt_norm, mask=None)
        
        # 3. 池化 + 投影 -> 分数
        pooled = attn_out.mean(dim=1)  # [B, C]
        score = self.score_proj(pooled)  # [B, 1]
        
        return torch.sigmoid(score)  # 映射到0-1