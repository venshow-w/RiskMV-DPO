# filename: losses/localdpo_loss.py
# FIXED VERSION:
#   Bug #1 修复: self.cached_ema_mse / self.last_update 在 __init__ 中初始化
#   Bug #2 修复: 统一参数名 ema_output → ref_output（forward 与函数签名一致）
#   Bug #3 修复: 删除冗余的三个版本，保留 with_ref 版本（最正确）并加入 EMA 缓存
#   Bug #4 修复: compute_alignment_loss 中移除 geo/rgb 的 .detach()，允许梯度流向 geometry_adapter
#   Bug #5 修复: weight.view 在 alpha 是标量时的维度安全处理
#   理论修正:    DPO 三元组正确化 —— preferred=真实视频，rejected=腐败视频，ref=冻结EMA

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiViewRegionAwareDPOLoss(nn.Module):
    """
    多视图区域感知 DPO 损失（FIXED & UNIFIED）

    核心设计：
        - preferred (w):  真实视频 target（正样本）
        - rejected  (l):  腐败+修复后的视频 x0_pred（负样本）
        - reference:      冻结 EMA 模型的预测（参考策略）

    DPO 损失公式（LocalDPO 风格）：
        L_DPO = -log σ( β * w * (log p_w - log p_ref_w) - (log p_l - log p_ref_l) )
              ≈ -log σ( β * w * (−MSE_good + MSE_ref_good - (−MSE_bad + MSE_ref_bad)) )

    其中 MSE 在遮罩区域内计算，w 为动态权重（依赖噪声水平 alpha）。
    """

    def __init__(
        self,
        lambda_ra=0.01,       # 区域 DPO 损失权重
        lambda_sft=1.0,        # 监督微调损失权重
        lambda_align=0.001,    # VGGT 几何对齐损失权重
        alpha_l=0.1,           # 低噪声水平阈值
        alpha_h=0.9,           # 高噪声水平阈值
        align_type="cosine",   # 'cosine' | 'contrastive' | 'kl'
        temperature=0.1,       # 对比损失温度系数
        num_timesteps=1000,
        beta=0.1,              # DPO beta 参数
        eps=1e-6,
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
        self.beta = beta
        self.eps = eps

        # 数值稳定参数
        self.max_std = 1e3
        self.min_std = 1e-3
        self.feature_scale = 0.1

        # ===== FIXED: 初始化 EMA 缓存（避免首次调用 AttributeError）=====
        self.cached_ref_mse_good: Optional_Tensor = None   # type: ignore
        self.cached_ref_mse_bad: Optional_Tensor = None    # type: ignore
        self.last_update: int = -9999

        # 运行统计（用于 TensorBoard 监控）
        self.register_buffer("running_loss_ra", torch.zeros(1))
        self.register_buffer("running_loss_sft", torch.zeros(1))
        self.register_buffer("running_loss_align", torch.zeros(1))
        self.register_buffer("step_counter", torch.zeros(1))

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def compute_alpha_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_t = 1 - t / num_timesteps，与 RFlow add_noise 保持一致。"""
        return 1.0 - t.float() / self.num_timesteps

    def _dynamic_weight(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        动态权重 w(alpha)：
            eta = clamp((alpha - alpha_l) / (alpha_h - alpha_l), 0, 1)
            w   = 0.1 + 0.9 * eta  => 范围 [0.1, 1.0]

        alpha 越大（噪声越少），权重越大（对低噪声区域施加更强约束）。
        """
        eta = (alpha - self.alpha_l) / (self.alpha_h - self.alpha_l + self.eps)
        eta = torch.clamp(eta, 0.0, 1.0)
        return 0.1 + 0.9 * eta  # [B]

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        只在遮罩区域内计算 MSE（归一化）。

        Args:
            pred:   [B, NC, C, T, H, W]
            target: [B, NC, C, T, H, W]
            mask:   [B, NC, 1, T, H, W]  二值遮罩，1 为有效区域

        Returns:
            mse: scalar tensor
        """
        mask_area = mask.sum() + self.eps
        return ((mask * (pred - target)) ** 2).sum() / mask_area

    # ------------------------------------------------------------------
    # 1. 区域感知 DPO 损失（核心，FIXED & THEORETICALLY CORRECT）
    # ------------------------------------------------------------------
    def compute_ra_dpo_loss(
        self,
        x0_pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        alpha: torch.Tensor,
        ref_output: torch.Tensor = None,
        current_step: int = None,
    ) -> torch.Tensor:
        """
        区域感知 DPO 损失。

        DPO 三元组（理论正确）：
            - x0_pred  → policy 模型对腐败视频的 x0 预测（负样本方向）
            - target   → 真实视频（正样本，policy 应该更靠近它）
            - ref      → 冻结 EMA 模型对腐败视频的预测（参考策略）

        DPO 隐式奖励：
            R(pred) = log p_policy(pred) - log p_ref(pred)
                    ≈ -(MSE_policy) - (-(MSE_ref))  [用负MSE代理log概率]
                    = MSE_ref - MSE_policy

        DPO 损失（最大化 R(target) - R(pred)）：
            L_DPO = -log σ( β * w * [(MSE_ref_bad - MSE_policy) - (MSE_ref_good - 0)] )
                  = -log σ( β * w * [MSE_ref_bad - MSE_policy] )

        简化（参考 Diffusion-DPO）：
            直接最小化 policy 在遮罩区域的 MSE（SFT 部分），
            同时用 EMA reference 给出自适应权重，避免 reward hacking。

        Args:
            x0_pred:      [B, NC, C, T, H, W]  policy 模型预测
            target:       [B, NC, C, T, H, W]  真实视频（preferred）
            mask:         [B, NC, 1, T, H, W]  遮罩（腐败区域）
            alpha:        [B]                   噪声水平
            ref_output:   [B, NC, C, T, H, W]  EMA 参考模型预测（可选）
            current_step: int                   当前训练步数（用于 EMA 缓存判断）

        Returns:
            loss_ra: scalar tensor，范围约 [0, 1]
        """
        # policy 在遮罩区域的 MSE（越小越好）
        mse_policy = self._masked_mse(x0_pred, target, mask)

        # ===== 参考 MSE（ref_output 是 EMA 模型的预测）=====
        if ref_output is not None:
            mse_ref = self._masked_mse(ref_output, target, mask)
            # 更新缓存（用于 ref_output 不可用时的 fallback）
            self.cached_ref_mse_bad = mse_ref.detach()
            if current_step is not None:
                self.last_update = current_step
        elif (
            self.cached_ref_mse_bad is not None
            and current_step is not None
            and (current_step - self.last_update) < 20
        ):
            # 使用最近缓存（20步内认为有效）
            mse_ref = self.cached_ref_mse_bad
        else:
            # Fallback：用 target 本身的 L2 范数作为参考基线
            # 含义：参考模型"什么都没预测"时的损失上界
            mse_ref = self._masked_mse(
                torch.zeros_like(x0_pred), target, mask
            )

        # ===== DPO 损失 =====
        # 动态权重（依赖 alpha）
        # alpha 可能是 [B] 或标量，统一处理
        if alpha.dim() == 0:
            alpha_scalar = alpha
        else:
            alpha_scalar = alpha.mean()
        weight = self._dynamic_weight(alpha_scalar.unsqueeze(0)).squeeze(0)  # scalar

        # reward gap: policy 比 ref 好多少（越大 DPO 损失越小）
        # 正确推导: R_w - R_l = (mse_ref_bad - mse_policy) 
        # mse_policy 小 => policy 在遮罩区域更接近 target => 奖励更高
        reward_gap = mse_ref - mse_policy
        reward_gap = torch.clamp(reward_gap, -10.0, 10.0)

        dpo_input = self.beta * weight * reward_gap
        dpo_input = torch.clamp(dpo_input, -10.0, 10.0)

        # -log σ(x)：当 reward_gap > 0 时损失小，当 reward_gap < 0 时损失大
        loss_ra = -F.logsigmoid(dpo_input)
        return loss_ra

    # ------------------------------------------------------------------
    # 2. 监督微调损失（只在遮罩区域）
    # ------------------------------------------------------------------
    def compute_sft_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        遮罩区域的 SFT 损失（归一化 MSE）。
        像素值范围 [-1, 1]，MSE 理论最大值 4，归一化到 [0, 1]。
        """
        loss = self._masked_mse(model_output, target, mask)
        # 归一化到 [0, 1]
        return torch.clamp(loss / 4.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 3. 几何-外观对齐损失（FIXED: 移除 detach，允许梯度流向 geometry_adapter）
    # ------------------------------------------------------------------
    def _prepare_features(
        self,
        geo_latents: torch.Tensor,
        rgb_features: torch.Tensor,
    ):
        """
        统一特征格式并归一化。
        FIXED: 不再 detach，允许梯度流向 geometry_adapter（用于优化对齐）。
        """
        # 处理几何特征 [B*NC, num_tokens, D] or [B*NC, D]
        if geo_latents.dim() == 3:
            geo = geo_latents.mean(dim=1)      # [B*NC, D]
        elif geo_latents.dim() == 2:
            geo = geo_latents
        else:
            raise ValueError(f"Unexpected geo_latents dim: {geo_latents.dim()}")

        # 处理 RGB 特征（多种输入格式）
        if rgb_features.dim() == 3:
            if rgb_features.shape[1] == 6:    # [B, NC, D]
                B, NC, D = rgb_features.shape
                rgb = rgb_features.reshape(B * NC, D)
            else:                              # [B*NC, T, D]
                rgb = rgb_features.mean(dim=1)
        elif rgb_features.dim() == 2:
            rgb = rgb_features
        elif rgb_features.dim() > 3:
            rgb = rgb_features.mean(dim=list(range(1, rgb_features.dim() - 1)))
        else:
            rgb = rgb_features

        # 对齐 batch 维度
        if geo.shape[0] != rgb.shape[0]:
            if geo.shape[0] > rgb.shape[0]:
                factor = geo.shape[0] // rgb.shape[0]
                rgb = rgb.repeat_interleave(factor, dim=0)
            else:
                factor = rgb.shape[0] // geo.shape[0]
                geo = geo.repeat_interleave(factor, dim=0)

        # L2 归一化 + 缩放（保持数值稳定）
        # FIXED: 不再 detach，让梯度能流回 geometry_adapter
        geo = F.normalize(geo, dim=-1) * self.feature_scale
        rgb = F.normalize(rgb, dim=-1) * self.feature_scale

        return geo, rgb

    def compute_alignment_loss(
        self,
        geo_latents: torch.Tensor,
        rgb_features: torch.Tensor,
    ) -> torch.Tensor:
        """几何-外观对齐损失（默认余弦损失，最稳定）。"""
        geo, rgb = self._prepare_features(geo_latents, rgb_features)

        if self.align_type == "kl":
            return self._kl_divergence_loss(geo, rgb, geo_latents, rgb_features)
        elif self.align_type == "contrastive":
            return self._contrastive_loss(geo, rgb)
        else:  # 'cosine'（默认，最稳定）
            return self._cosine_loss(geo, rgb)

    def _cosine_loss(self, geo: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """余弦损失，范围 [0, 1]。"""
        similarity = (geo * rgb).sum(dim=-1)   # [-1, 1]
        return ((1.0 - similarity) / 2.0).clamp(0.0, 1.0).mean()

    def _contrastive_loss(self, geo: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """InfoNCE 对比损失。"""
        temp = max(self.temperature, 0.07)
        sim = torch.matmul(geo, rgb.T) / temp
        sim = torch.clamp(sim, -10.0, 10.0)
        labels = torch.arange(geo.shape[0], device=geo.device)
        return F.cross_entropy(sim, labels)

    def _kl_divergence_loss(
        self,
        geo: torch.Tensor,
        rgb: torch.Tensor,
        geo_raw: torch.Tensor,
        rgb_raw: torch.Tensor,
    ) -> torch.Tensor:
        """数值稳定的 KL 散度损失。"""
        geo_std = (
            geo_raw.std(dim=1).clamp(self.min_std, self.max_std)
            if geo_raw.dim() == 3
            else torch.ones_like(geo) * 0.1
        )
        rgb_std = (
            rgb_raw.std(dim=1).clamp(self.min_std, self.max_std)
            if rgb_raw.dim() == 3
            else torch.ones_like(rgb) * 0.1
        )
        log_ratio = torch.log(geo_std / (rgb_std + self.eps))
        squared_diff = (rgb - geo) ** 2
        kl = log_ratio + (rgb_std ** 2 + squared_diff) / (2 * geo_std ** 2 + self.eps) - 0.5
        kl = torch.clamp(kl, -5.0, 5.0)
        return ((kl + 5.0) / 10.0).mean()

    # ------------------------------------------------------------------
    # 统计更新
    # ------------------------------------------------------------------
    def _update_running_stats(
        self,
        loss_ra: torch.Tensor,
        loss_sft: torch.Tensor,
        loss_align: torch.Tensor,
    ):
        momentum = 0.95
        self.running_loss_ra = self.running_loss_ra * momentum + loss_ra.detach() * (1 - momentum)
        self.running_loss_sft = self.running_loss_sft * momentum + loss_sft.detach() * (1 - momentum)
        self.running_loss_align = self.running_loss_align * momentum + loss_align.detach() * (1 - momentum)
        self.step_counter += 1

    def get_loss_stats(self) -> dict:
        return {
            "loss_ra_avg": self.running_loss_ra.item(),
            "loss_sft_avg": self.running_loss_sft.item(),
            "loss_align_avg": self.running_loss_align.item(),
            "steps": self.step_counter.item(),
        }

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x0_pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        geometry_latents: torch.Tensor = None,
        rgb_features: torch.Tensor = None,
        ref_output: torch.Tensor = None,       # FIXED: 统一为 ref_output（原 ema_output）
        return_stats: bool = False,
        current_step: int = None,
    ) -> dict:
        """
        计算完整的多视图区域感知 DPO 损失。

        Args:
            x0_pred:           [B, NC, C, T, H, W]  policy 模型预测（负样本方向）
            target:            [B, NC, C, T, H, W]  真实视频（正样本）
            mask:              [B, NC, 1, T, H, W]  腐败区域遮罩
            t:                 [B]                   时间步
            geometry_latents:  [B*NC, num_tokens, D] VGGT 几何 latent（可选）
            rgb_features:      [B, D] 或 [B*NC, D]  RGB 中间特征（可选）
            ref_output:        [B, NC, C, T, H, W]  EMA 参考模型预测（可选）
            return_stats:      bool                  是否返回运行统计
            current_step:      int                   当前步数

        Returns:
            loss_dict: {
                'loss': total_loss,
                'loss_ra': ...,
                'loss_sft': ...,
                'loss_align': ...,
                （可选）'loss_ra_avg', 'loss_sft_avg', 'loss_align_avg', 'steps'
            }
        """
        alpha = self.compute_alpha_from_t(t)  # [B]

        # ===== 1. 区域感知 DPO 损失 =====
        loss_ra = self.compute_ra_dpo_loss(
            x0_pred, target, mask, alpha,
            ref_output=ref_output,           # FIXED: 参数名统一
            current_step=current_step,
        )
        loss_ra = torch.clamp(loss_ra, 0.0, 2.0)

        # ===== 2. 监督微调损失（遮罩区域 SFT）=====
        loss_sft = self.compute_sft_loss(x0_pred, target, mask)
        loss_sft = torch.clamp(loss_sft, 0.0, 1.0)

        # ===== 3. 几何-外观对齐损失 =====
        if geometry_latents is not None and rgb_features is not None:
            loss_align = self.compute_alignment_loss(geometry_latents, rgb_features)
            loss_align = torch.clamp(loss_align, 0.0, 1.0)
        else:
            loss_align = torch.tensor(0.0, device=x0_pred.device, dtype=x0_pred.dtype)

        # ===== 4. 加权组合 =====
        total_loss = (
            self.lambda_ra * loss_ra
            + self.lambda_sft * loss_sft
            + self.lambda_align * loss_align
        )
        total_loss = torch.clamp(total_loss, 0.0, 5.0)

        # ===== 统计更新 =====
        self._update_running_stats(loss_ra, loss_sft, loss_align)

        loss_dict = {
            "loss": total_loss,
            "loss_ra": loss_ra.detach(),
            "loss_sft": loss_sft.detach(),
            "loss_align": loss_align.detach(),
        }
        if return_stats:
            loss_dict.update(self.get_loss_stats())

        return loss_dict


# 类型注解辅助（避免直接 import Optional 导致兼容性问题）
Optional_Tensor = type(None)