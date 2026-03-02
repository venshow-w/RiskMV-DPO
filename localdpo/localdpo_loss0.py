# filename: losses/localdpo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiViewRegionAwareDPOLoss(nn.Module):
    """
    创新点3：多视图区域感知DPO损失
    - 原文LocalDPO + 多视图适配 + VGGT几何奖励
    """
    def __init__(self,
                 lambda_ra=1.0,      # 区域DPO损失权重
                 lambda_sft=0.1,     # 监督微调损失权重
                 lambda_align=0.005,    # VGGT几何奖励权重
                 alpha_l=0.1,        # 低噪声水平阈值
                 alpha_h=0.9,        # 高噪声水平阈值
                 align_type='kl',    # 'kl', 'contrastive', 'cosine'
                 temperature=0.1, # DPO温度系数
                 num_timesteps=1000,
                 ):   
        super().__init__()
        
        self.lambda_ra = lambda_ra
        self.lambda_sft = lambda_sft
        self.lambda_align = lambda_align
        self.alpha_l = alpha_l
        self.alpha_h = alpha_h
        self.temperature = temperature
        self.num_timesteps = num_timesteps
        self.align_type = align_type

        self.eps = 1e-6
        self.max_std = 1e3
        self.min_std = 1e-3
        self.feature_scale = 0.1
        

    def compute_alpha_from_t(self, t):
        """
        计算噪声水平alpha = 1 - t/num_timesteps
        t越大，alpha越小（噪声越多）
        """
        alpha = 1 - (t.float() / self.num_timesteps)
        return alpha
    
    def _prepare_features(self, geo_latents, rgb_features):
        """
        统一特征格式并进行数值稳定化
        """
        # 处理几何特征
        if geo_latents.dim() == 3:  # [B*NC, num_tokens, D]
            geo = geo_latents.mean(dim=1)  # [B*NC, D]
        elif geo_latents.dim() == 2:  # [B*NC, D]
            geo = geo_latents
        else:
            raise ValueError(f"Unexpected geo_latents dim: {geo_latents.dim()}")
        
        # 处理RGB特征
        if rgb_features.dim() == 3:
            if rgb_features.shape[1] == 6:  # [B, NC, D]
                B, NC, D = rgb_features.shape
                rgb = rgb_features.reshape(B * NC, D)
            else:  # [B*NC, T, D]
                rgb = rgb_features.mean(dim=1)
        elif rgb_features.dim() == 2:
            rgb = rgb_features
        else:
            rgb = rgb_features.mean(dim=[1,2,3]) if rgb_features.dim() > 2 else rgb_features
        
        if geo.shape != rgb.shape:
            if geo.shape[0] > rgb.shape[0]:
                rgb = rgb.repeat_interleave(geo.shape[0] // rgb.shape[0], dim=0)
            else:
                geo = geo.repeat_interleave(rgb.shape[0] // geo.shape[0], dim=0)
        
        # ===== 关键：特征归一化 =====
        geo = F.normalize(geo, dim=-1) * self.feature_scale
        rgb = F.normalize(rgb, dim=-1) * self.feature_scale
        return geo.detach(), rgb.detach()
    
    def compute_alignment_loss(self, geo_latents, rgb_features):
        """✅ 正确的对齐损失，不是MSE"""
        
        geo, rgb = self._prepare_features(geo_latents, rgb_features)

        # ===== 2. 根据类型计算 =====
        if self.align_type == 'kl':
            return self._kl_divergence_stable(geo, rgb, geo_latents, rgb_features)
        elif self.align_type == 'contrastive':
            return self._contrastive_loss_stable(geo, rgb)
        else:
            return self._cosine_loss_stable(geo, rgb)
    
    def _kl_divergence_stable(self, geo, rgb, geo_raw, rgb_raw):
        """数值稳定的KL散度"""
        
        # 计算标准差（使用原始特征）
        if geo_raw.dim() == 3:
            geo_std = geo_raw.std(dim=1)
        else:
            geo_std = torch.ones_like(geo) * 0.1
            
        if rgb_raw.dim() > 2:
            rgb_std = rgb_raw.std(dim=1) if rgb_raw.dim() == 3 else rgb_raw.std(dim=[1,2,3])
        else:
            rgb_std = torch.ones_like(rgb) * 0.1
        
        # 稳定化处理
        geo_std = torch.clamp(geo_std, self.min_std, self.max_std)
        rgb_std = torch.clamp(rgb_std, self.min_std, self.max_std)
        
        # 计算KL散度
        log_ratio = torch.log(geo_std / (rgb_std + self.eps))
        squared_diff = (rgb - geo) ** 2
        second_term = (rgb_std**2 + squared_diff) / (2 * geo_std**2 + self.eps)
        
        loss = log_ratio + second_term - 0.5
        
        # 裁剪异常值
        loss = torch.clamp(loss, -10, 10)
        
        return loss.mean()

    def _contrastive_loss_stable(self, geo, rgb):
        """稳定的对比损失"""
        # 温度参数
        temp = max(self.temperature, 0.1)
        
        # 相似度矩阵
        sim = torch.matmul(geo, rgb.T) / temp
        
        # 数值稳定化
        sim = torch.clamp(sim, -10, 10)
        
        # InfoNCE
        labels = torch.arange(geo.shape[0], device=geo.device)
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def _cosine_loss_stable(self, geo, rgb):
        """稳定的余弦损失"""
        similarity = (geo * rgb).sum(dim=-1)
        # 确保loss在[0,2]范围
        loss = 1.0 - torch.clamp(similarity, -1, 1).mean()
        return loss
    
    
    def compute_ra_dpo_loss(self, 
                           x0_pred,    # 模型修复结果
                           target,          # 真实视频
                           mask,            # 遮罩区域
                           alpha,           # 噪声水平
                           reference_output=None):  # 参考模型输出（可选）
        """
        区域感知DPO损失（LocalDPO公式5-6）
        """
        B, NC, C, T, H, W = target.shape
        N_M = mask.sum() + 1e-8 # 遮罩区域像素数
        ll = mask.norm(p=1) + 1e-8
        
        # 动态加权因子(LocalDPO公式6)
        eta = (alpha - self.alpha_l) / (self.alpha_h - self.alpha_l + 1e-8)
        eta = torch.clamp(eta, 0, 1)
        weight = 1 + eta  # [B]
        
        # 扩展到所有维度
        weight = weight.view(B, 1, 1, 1, 1, 1)
        
        # 区域MSE
        mse_model = ((mask * (x0_pred - target)) ** 2).sum() / (N_M + 1e-8)
        
        if reference_output is not None:
            mse_ref = ((mask * (reference_output - target)) ** 2).sum() / (N_M + 1e-8)
        else:
            # 如果没有参考模型，使用零输出作为参考（LocalDPO实践）
            mse_ref = ((mask * target) ** 2).sum() / (N_M + 1e-8)
        
        # 区域DPO损失（公式5）
        loss_ra = weight * (N_M / (ll + 1e-8)) * (mse_model - mse_ref)
        
        return loss_ra.mean()
    
    def compute_sft_loss(self, model_output, target, mask):
        """
        监督微调损失（只计算遮罩区域）
        """
        return F.mse_loss(
            model_output * mask,
            target * mask
        )
    
    def compute_geo_reward(self, vggt_score):
        """
        VGGT几何一致性奖励
        vggt_score: [B] 0-1之间，越高几何一致性越好
        """
        # DPO风格的奖励：分数越高，损失越小
        return -torch.log(vggt_score + 1e-8).mean()
    
    def forward(self,
                x0_pred,
                target,
                mask,
                t,
                # vggt_scorer=None,
                geometry_latents=None,
                rgb_features=None,
                reference_output=None):
        """
        完整的多视图区域感知DPO损失
        """
        # 计算噪声水平
        alpha = self.compute_alpha_from_t(t)
        
        # 1. 区域DPO损失（核心）
        loss_ra = self.compute_ra_dpo_loss(
            x0_pred, target, mask, alpha, reference_output
        )
        
        # 2. 监督微调损失
        loss_sft = self.compute_sft_loss(x0_pred, target, mask)
        
        # 3. 几何对齐损失
        # if vggt_scorer is not None and video_features is not None:
        #     with torch.no_grad():
        #         geo_score = vggt_scorer(video_features, vggt_features)
        #     loss_geo = self.compute_geo_reward(geo_score)
        
        if geometry_latents is not None and rgb_features is not None:
            loss_align = self.compute_alignment_loss(geometry_latents, rgb_features)
        else:
            loss_align = torch.tensor(0.0, device=x0_pred.device)
        
        # 总损失
        total_loss = (self.lambda_ra * loss_ra + 
                     self.lambda_sft * loss_sft + 
                     self.lambda_align * loss_align)
        
        loss_dict = {
            'loss': total_loss,
            'loss_ra': loss_ra,
            'loss_sft': loss_sft,
            'loss_align': loss_align
        }
        
        return loss_dict