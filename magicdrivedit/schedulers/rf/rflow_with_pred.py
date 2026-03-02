# filename: magicdrivedit/schedulers/rflow_with_pred.py
# FIXED VERSION: 修复了 predict_x0 公式与时间轴不一致的问题

from typing import List, Optional
import logging
import torch
import torch.nn.functional as F
from torch.distributions import LogisticNormal
from einops import rearrange

from .rectified_flow import mean_flat, _extract_into_tensor, timestep_transform


class RFlowSchedulerWithPred:
    """
    增强版 RFlowScheduler，支持返回预测的 x0 用于 DPO 损失。

    RFlow 噪声公式（与 add_noise 严格对称）：
        x_t = alpha_t * x0 + (1 - alpha_t) * noise
        其中 alpha_t = 1 - t / num_timesteps，范围 [1/T, 1]

    velocity 定义（dx_t/dt）：
        v = d x_t / d alpha_t * (d alpha_t / d t)
          = (x0 - noise) * (-1 / num_timesteps)
        => 模型预测的 velocity_pred ≈ x0 - noise

    因此 x0 的还原公式：
        x_t = alpha_t * x0 + (1 - alpha_t) * noise
        x_t = alpha_t * x0 + (1 - alpha_t) * (x0 - velocity_pred)
             [noise = x0 - velocity_pred]
        x_t = x0 - (1 - alpha_t) * velocity_pred
        => x0_pred = x_t + (1 - alpha_t) * velocity_pred
    """

    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        cog_style_trans=False,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        if cog_style_trans:
            logging.warning("Use `cog_style_trans`. Please make sure train&inference is consistent!")
        self.cog_style_trans = cog_style_trans

    def _get_alpha(self, t):
        """
        计算 alpha_t = 1 - t / num_timesteps，与 add_noise 严格一致。
        返回 shape: [B]，范围 (0, 1]
        """
        return 1.0 - t.float() / self.num_timesteps

    def predict_x0(self, x_t, velocity_pred, t):
        """
        从预测的 velocity 还原 x0。

        RFlow 推导（与 add_noise 严格对称）：
            add_noise:   x_t = alpha_t * x0 + (1 - alpha_t) * noise
            velocity:    velocity_pred ≈ x0 - noise
                        => noise = x0 - velocity_pred
            代入:        x_t = alpha_t * x0 + (1 - alpha_t) * (x0 - velocity_pred)
                             = x0 - (1 - alpha_t) * velocity_pred
            => x0_pred = x_t + (1 - alpha_t) * velocity_pred

        Args:
            x_t:           [B, C, T, H, W]  加噪后的 latent
            velocity_pred: [B, C, T, H, W]  模型预测的 velocity
            t:             [B]               整数时间步（0 ~ num_timesteps）

        Returns:
            x0_pred: [B, C, T, H, W]
        """
        alpha_t = self._get_alpha(t)                  # [B]，与 add_noise 完全对称
        one_minus_alpha = 1.0 - alpha_t               # [B]
        # 扩展到与 x_t 相同的维度（支持 5D 和 6D tensor）
        for _ in range(x_t.dim() - 1):
            one_minus_alpha = one_minus_alpha.unsqueeze(-1)
        x0_pred = x_t + one_minus_alpha * velocity_pred
        return x0_pred

    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
        return_x0_pred=False,
        return_features=False,
        feature_layers=None,
    ):
        """
        计算训练损失（支持可选的 x0_pred 返回和中间特征返回）。

        Args:
            model:           扩散模型
            x_start:         [B, C*NC, T, H, W]  干净的 latent（正样本或腐败视频）
            model_kwargs:    传给 model.forward 的其他参数
            noise:           可选的预定义噪声
            mask:            帧级时序掩码 [B, T]（scheduler_mask）
            weights:         可选的时间步权重
            t:               可选的预定义时间步
            return_x0_pred:  是否在 terms 中返回 x0_pred（stop-gradient）
            return_features: 是否让模型返回中间层特征
            feature_layers:  指定保存的层索引列表

        Returns:
            terms: dict，包含 'loss'，可选含 'x0_pred'、'features'
        """
        if model_kwargs is None:
            model_kwargs = {}

        # ===== 采样时间步 =====
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(
                    t, model_kwargs,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    cog_style=self.cog_style_trans,
                )

        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        # ===== 添加噪声（RFlow）=====
        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}

        # ===== 模型前向（支持特征返回）=====
        if return_features:
            model_output, features = model(
                x_t, t, **model_kwargs,
                save_features=True,
                feature_layers=feature_layers,
            )
            if features is not None:
                terms["features"] = features
        else:
            model_output = model(x_t, t, **model_kwargs)

        # 处理可能的双通道输出（pred_sigma=True 时）
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output

        # ===== 计算 RFlow MSE 损失（target = x0 - noise = x_start - noise）=====
        target = x_start - noise
        if weights is None:
            loss = mean_flat((velocity_pred - target).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - target).pow(2), mask=mask)
        terms["loss"] = loss

        # ===== 可选：返回 x0_pred（stop-gradient，不参与反向传播）=====
        if return_x0_pred:
            with torch.no_grad():
                x0_pred = self.predict_x0(x_t, velocity_pred.detach(), t)
                terms["x0_pred"] = x0_pred

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        RFlow 噪声添加（兼容 diffusers 接口）：
            x_t = alpha_t * x0 + (1 - alpha_t) * noise
            alpha_t = 1 - t / num_timesteps，范围 [1/T, 1]
        """
        alpha_t = self._get_alpha(timesteps)        # [B]，范围 (0, 1]
        one_minus_alpha = 1.0 - alpha_t             # [B]，范围 [0, 1)

        # 扩展到 [B, C, T, H, W]
        while alpha_t.dim() < original_samples.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            one_minus_alpha = one_minus_alpha.unsqueeze(-1)

        return alpha_t * original_samples + one_minus_alpha * noise