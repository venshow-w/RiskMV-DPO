# filename: losses/localdpo_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiViewRegionAwareDPOLoss(nn.Module):
    """
    创新点3：多视图区域感知DPO损失
    - 修正版本：数值稳定，loss在合理范围
    """
    def __init__(self,
                 lambda_ra=0.01,      # 区域DPO损失权重
                 lambda_sft=1.0,       # 监督微调损失权重
                 lambda_align=0.001,   # VGGT几何奖励权重
                 alpha_l=0.1,          # 低噪声水平阈值
                 alpha_h=0.9,          # 高噪声水平阈值
                 align_type='cosine',  # 对齐类型
                 temperature=0.1,      # 温度系数
                 num_timesteps=1000,
                 beta=0.1,             # DPO beta参数
                 eps=1e-6):   
        super().__init__()
        
        self.lambda_ra = lambda_ra
        self.lambda_sft = lambda_sft
        self.lambda_align = lambda_align
        self.alpha_l = alpha_l
        self.alpha_h = alpha_h
        self.temperature = temperature
        self.num_timesteps = num_timesteps
        self.align_type = align_type
        self.beta = beta
        self.eps = eps
        
        # 数值稳定参数
        self.max_std = 1e3
        self.min_std = 1e-3
        self.feature_scale = 0.1
        
        # 统计信息（用于调试）
        self.register_buffer('running_loss_ra', torch.zeros(1))
        self.register_buffer('running_loss_sft', torch.zeros(1))
        self.register_buffer('running_loss_align', torch.zeros(1))
        self.register_buffer('step_counter', torch.zeros(1))
        
    def compute_alpha_from_t(self, t):
        """计算噪声水平alpha = 1 - t/num_timesteps"""
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
        
        # 确保维度匹配
        if geo.shape != rgb.shape:
            if geo.shape[0] > rgb.shape[0]:
                rgb = rgb.repeat_interleave(geo.shape[0] // rgb.shape[0], dim=0)
            else:
                geo = geo.repeat_interleave(rgb.shape[0] // geo.shape[0], dim=0)
        
        # 特征归一化（关键）
        geo = F.normalize(geo, dim=-1) * self.feature_scale
        rgb = F.normalize(rgb, dim=-1) * self.feature_scale
        
        return geo.detach(), rgb.detach()
    
    def compute_alignment_loss(self, geo_latents, rgb_features):
        """
        几何-外观对齐损失
        输出范围: [0, 1] 之间
        """
        geo, rgb = self._prepare_features(geo_latents, rgb_features)
        
        if self.align_type == 'kl':
            return self._kl_divergence_stable(geo, rgb, geo_latents, rgb_features)
        elif self.align_type == 'contrastive':
            return self._contrastive_loss_stable(geo, rgb)
        else:  # 'cosine' (默认，最稳定)
            return self._cosine_loss_stable(geo, rgb)
    
    def _kl_divergence_stable(self, geo, rgb, geo_raw, rgb_raw):
        """数值稳定的KL散度，输出范围 [0, 5] 左右"""
        
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
        
        # 裁剪异常值并缩放到合理范围
        loss = torch.clamp(loss, -5, 5)
        loss = (loss + 5) / 10  # 缩放到 [0, 1]
        
        return loss.mean()
    
    def _contrastive_loss_stable(self, geo, rgb):
        """稳定的对比损失，输出范围 [0, 5] 左右"""
        temp = max(self.temperature, 0.1)
        
        # 相似度矩阵
        sim = torch.matmul(geo, rgb.T) / temp
        sim = torch.clamp(sim, -10, 10)  # 稳定化
        
        # InfoNCE
        labels = torch.arange(geo.shape[0], device=geo.device)
        loss = F.cross_entropy(sim, labels)
        
        # 缩放到 [0, 1]
        loss = torch.sigmoid(loss)  # 假设原始loss在0-5左右
        
        return loss
    
    def _cosine_loss_stable(self, geo, rgb):
        """稳定的余弦损失，输出范围 [0, 1]"""
        similarity = (geo * rgb).sum(dim=-1)
        # 余弦相似度范围[-1, 1]，loss范围[0, 1]
        loss = (1.0 - similarity) / 2
        loss = torch.clamp(loss, 0, 1).mean()
        return loss
    
    def compute_ra_dpo_loss(self, x0_pred, target, mask, alpha):
        """
        修正的区域DPO损失
        输出范围: [0, 1] 之间
        
        遵循Local DPO论文的核心思想：
        - 用log概率比值代替MSE差值
        - 用log-sigmoid将差值映射到(0,∞)
        """
        B, NC, C, T, H, W = target.shape
        
        # ===== 1. 计算遮罩区域的log概率 =====
        mask_area = mask.sum() + self.eps
        
        # 好回答的log概率（遮罩区域被修复）
        # 使用负MSE作为log概率的代理（越小越好）
        mse_good = ((mask * (x0_pred - target)) ** 2).sum() / mask_area
        log_p_good = -mse_good  # 转换为log概率（越大越好）
        
        # 坏回答的log概率（遮罩区域保持原始状态）
        # 使用随机噪声或原始target作为baseline
        mse_bad = ((mask * target) ** 2).sum() / mask_area
        log_p_bad = -mse_bad
        
        # ===== 2. 参考模型的log概率（使用简单baseline）= 
        # 实际应该用冻结的reference model，这里用target的MSE近似
        log_ref_good = -mse_good.detach()  # 参考模型的好回答log概率
        log_ref_bad = -mse_bad.detach()    # 参考模型的坏回答log概率
        
        # ===== 3. 隐式奖励差 =====
        # R_good - R_bad = (log_p_good - log_ref_good) - (log_p_bad - log_ref_bad)
        reward_gap = (log_p_good - log_ref_good) - (log_p_bad - log_ref_bad)
        
        # 稳定化处理
        reward_gap = torch.clamp(reward_gap, -10, 10)
        
        # ===== 4. 动态权重（噪声水平） =====
        eta = (alpha - self.alpha_l) / (self.alpha_h - self.alpha_l + self.eps)
        eta = torch.clamp(eta, 0, 1)
        weight = 0.1 + 0.9 * eta  # [0.1, 1.0]
        
        # ===== 5. DPO损失 =====
        # -log σ(β * weight * reward_gap)
        # 当reward_gap为正时，损失小；为负时，损失大
        dpo_input = self.beta * weight * reward_gap
        dpo_input = torch.clamp(dpo_input, -10, 10)  # 避免sigmoid饱和
        
        loss_ra = -F.logsigmoid(dpo_input)
        
        # 缩放到合理范围
        loss_ra = loss_ra.mean()
        loss_ra = torch.clamp(loss_ra, 0, 2)  # 确保在0-2之间
        
        return loss_ra
    
    def compute_ra_dpo_loss_original(self, x0_pred, target, mask, alpha, reference_output=None):
        """
        原始Local DPO论文风格的实现
        使用MSE差值但经过sigmoid处理
        """
        B, NC, C, T, H, W = target.shape
        mask_area = mask.sum() + self.eps
        
        # 1. 模型MSE
        mse_model = ((mask * (x0_pred - target)) ** 2).sum() / mask_area
        
        # 2. 参考MSE
        if reference_output is not None:
            mse_ref = ((mask * (reference_output - target)) ** 2).sum() / mask_area
        else:
            # 用target本身作为参考（会导致diff=0）
            mse_ref = ((mask * target) ** 2).sum() / mask_area
        
        # 3. 差值
        diff = mse_model - mse_ref
        
        # 4. 动态权重
        eta = (alpha - self.alpha_l) / (self.alpha_h - self.alpha_l + self.eps)
        eta = torch.clamp(eta, 0, 1)
        weight = 1 + eta  # [1, 2]
        
        # 5. 区域归一化因子
        region_factor = mask_area / (mask.norm(p=1) + self.eps)
        
        # 6. DPO损失（使用sigmoid转换）
        # 将差值通过sigmoid映射到(0,1)，然后取负对数
        dpo_input = weight * region_factor * diff
        dpo_input = torch.clamp(dpo_input, -10, 10)  # 稳定化
        
        loss_ra = -F.logsigmoid(-dpo_input)  # 注意符号
        
        return loss_ra.mean() * 0.01  # 额外缩放
    
    
    def compute_ra_dpo_loss_with_ema(self, x0_pred, target, mask, alpha, 
                           ema_output=None, current_step=None):
        mask_area = mask.sum() + self.eps
        mse_policy = ((mask * (x0_pred - target)) ** 2).sum() / mask_area
        
        # 计算reference MSE
        if ema_output is not None:
            mse_ref = ((mask * (ema_output - target)) ** 2).sum() / mask_area
            # 缓存
            if current_step is not None:
                self.cached_ema_mse = mse_ref.detach()
                self.last_update = current_step
        elif self.cached_ema_mse is not None and current_step - self.last_update < 10:
            mse_ref = self.cached_ema_mse
        else:
            # 回退：用target * 0.5作为参考（经验值）
            mse_ref = ((mask * target) ** 2).sum() / mask_area * 0.5
        
        # DPO损失
        diff = mse_policy - mse_ref
        eta = (alpha - self.alpha_l) / (self.alpha_h - self.alpha_l + self.eps)
        eta = torch.clamp(eta, 0, 1)
        weight = 0.1 + 0.9 * eta
        
        loss_ra = -F.logsigmoid(weight * diff)
        
        return loss_ra.mean() * 0.1
    
    def compute_ra_dpo_loss_with_ref(self, x0_pred, target, mask, alpha, ref_output):
        """
        使用reference output的DPO损失
        """
        mask_area = mask.sum() + self.eps
        
        # policy model的MSE
        mse_policy = ((mask * (x0_pred - target)) ** 2).sum() / mask_area
        
        if ref_output is not None:
            # reference model的MSE
            mse_ref = ((mask * (ref_output - target)) ** 2).sum() / mask_area
        else:
            # 如果没有reference，用target（退化为SFT）
            mse_ref = ((mask * target) ** 2).sum() / mask_area
        
        # 隐式奖励差
        diff = mse_policy - mse_ref
        
        # 动态权重
        eta = (alpha - self.alpha_l) / (self.alpha_h - self.alpha_l + self.eps)
        eta = torch.clamp(eta, 0, 1)
        weight = 0.1 + 0.9 * eta
        
        # DPO损失
        loss_ra = -F.logsigmoid(weight * diff)
        
        return loss_ra.mean() * 0.1  
    
    
    def compute_sft_loss(self, model_output, target, mask):
        """
        监督微调损失（只计算遮罩区域）
        输出范围: [0, 1] 左右（假设像素值在0-1之间）
        """
        loss = F.mse_loss(
            model_output * mask,
            target * mask,
            reduction='sum'
        ) / (mask.sum() + self.eps)
        
        # 如果像素值范围是[-1,1]，MSE范围是[0,4]
        # 缩放到[0,1]
        loss = loss / 4.0
        
        return loss
    
    def update_running_stats(self, loss_ra, loss_sft, loss_align):
        """更新运行统计"""
        self.running_loss_ra = self.running_loss_ra * 0.95 + loss_ra.detach() * 0.05
        self.running_loss_sft = self.running_loss_sft * 0.95 + loss_sft.detach() * 0.05
        self.running_loss_align = self.running_loss_align * 0.95 + loss_align.detach() * 0.05
        self.step_counter += 1
    
    def get_loss_stats(self):
        """获取损失统计"""
        return {
            'loss_ra_avg': self.running_loss_ra.item(),
            'loss_sft_avg': self.running_loss_sft.item(),
            'loss_align_avg': self.running_loss_align.item(),
            'steps': self.step_counter.item()
        }
    
    def forward(self,
                x0_pred,
                target,
                mask,
                t,
                geometry_latents=None,
                rgb_features=None,
                ref_output=None,
                return_stats=False,
                current_step=None):
        """
        完整的多视图区域感知DPO损失
        总损失范围: 大约 0.1 - 2.0
        """
        # 计算噪声水平
        alpha = self.compute_alpha_from_t(t)
        
        # ===== 1. 区域DPO损失 =====
        loss_ra = self.compute_ra_dpo_loss_with_ema(
            x0_pred, target, mask, alpha, ref_output, current_step
        )
        #  loss_ra = self.compute_ra_dpo_loss_with_ref(
        #     x0_pred, target, mask, alpha, ref_output
        # )
        #  确保loss_ra在合理范围
        loss_ra = torch.clamp(loss_ra, 0, 2)
        
        # ===== 2. 监督微调损失 =====
        loss_sft = self.compute_sft_loss(x0_pred, target, mask)
        loss_sft = torch.clamp(loss_sft, 0, 1)
        
        # ===== 3. 几何对齐损失 =====
        if geometry_latents is not None and rgb_features is not None:
            loss_align = self.compute_alignment_loss(geometry_latents, rgb_features)
            loss_align = torch.clamp(loss_align, 0, 1)
        else:
            loss_align = torch.tensor(0.0, device=x0_pred.device)
        
        # ===== 4. 组合损失 =====
        total_loss = (self.lambda_ra * loss_ra + 
                     self.lambda_sft * loss_sft + 
                     self.lambda_align * loss_align)
        
        # 确保总损失在合理范围
        total_loss = torch.clamp(total_loss, 0, 5)
        
        # 更新统计
        self.update_running_stats(loss_ra, loss_sft, loss_align)
        
        loss_dict = {
            'loss': total_loss,
            'loss_ra': loss_ra.detach(),
            'loss_sft': loss_sft.detach(),
            'loss_align': loss_align.detach()
        }
        
        if return_stats:
            loss_dict.update(self.get_loss_stats())
        
        return loss_dict