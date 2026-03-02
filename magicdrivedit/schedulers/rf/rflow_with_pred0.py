# filename: magicdrivedit/schedulers/rflow_with_pred.py

from typing import List, Optional
import logging
import torch
import torch.nn.functional as F
from torch.distributions import LogisticNormal
from einops import rearrange

# 导入原有函数
from .rectified_flow import mean_flat, _extract_into_tensor, timestep_transform


class RFlowSchedulerWithPred:
    """
    增强版RFlowScheduler，支持返回预测的x0用于DPO损失
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

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        if cog_style_trans:
            logging.warning("Use `cog_style_trans`. Please make sure train&inference is consistent!")
        self.cog_style_trans = cog_style_trans

    def predict_x0(self, x_t, velocity_pred, t):
        """
        从预测的速度还原x0
        RF公式: x_t = (1-t) * x_0 + t * noise
        所以 x_0_pred = x_t - t * velocity_pred
        """
        t_norm = t.float() / self.num_timesteps
        t_norm = 1 - t_norm  # [1, 1/1000]
        t_norm = t_norm.view(-1, 1, 1, 1, 1)
        
        x0_pred = x_t - t_norm * velocity_pred
        return x0_pred

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, 
                       mask=None, weights=None, t=None,
                       return_x0_pred=False,  # 新增：是否返回x0_pred
                       return_features=False,  # 新增：是否返回中间特征
                       feature_layers=None):   # 新增：指定特征层
        """
        Compute training losses for a single timestep.
        增强版：支持返回x0_pred和中间特征
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, 
                                      num_timesteps=self.num_timesteps, 
                                      cog_style=self.cog_style_trans)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        # 添加噪声
        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)
        
        terms = {}
        
        # ===== 模型前向，根据参数决定是否返回特征 =====
        if return_features:
            # 模型需要支持save_features参数
            model_output, features = model(
                x_t, t, **model_kwargs, 
                save_features=True,
                feature_layers=feature_layers
            )
            if features is not None:
                terms['features'] = features
        else:
            model_output = model(x_t, t, **model_kwargs)
        
        # 处理可能的两通道输出
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        
        # ===== 计算RF损失 =====
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss
        
        # ===== 如果需要，计算并返回x0_pred =====
        if return_x0_pred:
            with torch.no_grad():
                x0_pred = self.predict_x0(x_t, velocity_pred, t)
                terms['x0_pred'] = x0_pred
        
        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise