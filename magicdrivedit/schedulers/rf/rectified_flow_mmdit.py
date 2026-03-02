"""
rectified_flow_mmdit.py

在原版 rectified_flow.py 基础上适配 MagicDrive-MMDiT 的训练接口。

主要改动：
  1. training_losses 中的 model() 调用：
     原版: model(x_t, t, **model_kwargs)
       → timesteps 参数名为 `t`，通过位置参数传入
     MMDiT: model(x, timesteps, y, y_vec, maps, bbox, cams, rel_pos, mv_order_map, ...)
       → 接口完全解包，timesteps 为第二个参数

     由于我们的 MagicDriveMMDiT 使用 **kwargs 接收所有条件，
     实际调用兼容原版方式：model(x_t, t, **model_kwargs)
     只需要 model_kwargs 包含正确的键即可。

  2. x_mask 处理：
     原版使用 mask 参数（(B, T) bool tensor），mean_flat 做带 mask 的均值。
     MMDiT 中 x 的形状是 (B, C*NC, T, H, W)，mask 维度与 T 对齐，逻辑不变。

  3. timestep_transform 中的 num_frames：
     原版从 model_kwargs["num_frames"] 读取。
     MMDiT 不传 num_frames，改为从 x_start.shape[2] 推断（T 维度）。
"""

from typing import List, Optional
import logging

import torch
from torch.distributions import LogisticNormal
from einops import rearrange


def mean_flat(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    对所有非 batch 维度取均值。
    
    Args:
        tensor: 任意形状张量，第0维为 batch
        mask:   (B, T) bool tensor，1=有效帧，0=padding 帧
                要求 tensor.shape[2] == mask.shape[1]（T 维对齐）
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5, f"Expected 5D tensor, got {tensor.dim()}D"
        assert tensor.shape[2] == mask.shape[1], (
            f"T mismatch: tensor.shape[2]={tensor.shape[2]}, mask.shape[1]={mask.shape[1]}"
        )
        # tensor: (B, C, T, H, W) → (B, T, C*H*W) for masked mean
        # 注意：C 维包含 C*NC（多视图通道已经展平）
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]  # 有效 token 数
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]):
    """从 1D 数组中按 timesteps 索引提取，并广播到目标形状。"""
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
    cog_style=False,
    # MMDiT 新增：如果 model_kwargs 里没有 num_frames，从 x_shape 推断
    x_shape=None,
):
    """
    时间步变换（Resolution-aware）。
    
    与原版的差异：
      - MMDiT 的 model_kwargs 里没有 height/width/num_frames 这些字段
      - 改为从 x_shape (B, C*NC, T, H_lat, W_lat) 推断分辨率和帧数
      - 注意：latent 空间的 H/W 需要乘以 patch_size * vae_scale 才是像素空间分辨率
    """
    # Force fp16 → fp32
    t = t.float()

    # ── 分辨率推断 ────────────────────────────────────────────────────────────
    if "height" in model_kwargs and "width" in model_kwargs:
        # 原版路径：model_kwargs 里有像素分辨率
        for key in ["height", "width", "num_frames"]:
            if model_kwargs[key].dtype == torch.float16:
                model_kwargs[key] = model_kwargs[key].float()
        resolution = model_kwargs["height"] * model_kwargs["width"]
        ratio_space = (resolution / base_resolution).sqrt()
        
        if model_kwargs["num_frames"][0] == 1:
            num_frames = torch.ones_like(model_kwargs["num_frames"])
        else:
            if cog_style:
                num_frames = model_kwargs["num_frames"] // 4 + model_kwargs["num_frames"] % 2
            else:
                num_frames = model_kwargs["num_frames"] // 17 * 5
    elif x_shape is not None:
        # MMDiT 路径：从 x_shape 推断
        # x_shape = (B, C*NC, T, H_lat, W_lat)
        # 假设 VAE scale=8, patch_size=2，像素分辨率 = H_lat * 8, W_lat * 8
        # 但这里我们不做严格反推，用 latent 分辨率相对值代替
        B, CNC, T_lat, H_lat, W_lat = x_shape
        
        # latent 面积代表分辨率（相对量，用于比例）
        resolution_lat = torch.tensor(
            H_lat * W_lat, dtype=torch.float32, device=t.device
        ).expand(B)
        base_resolution_lat = base_resolution / (8 * 8)  # latent 空间等效基准
        ratio_space = (resolution_lat / base_resolution_lat).sqrt()

        # 帧数推断（latent T）
        if T_lat == 1:
            num_frames = torch.ones(B, device=t.device)
        else:
            if cog_style:
                # CogVideoX VAE: T_lat = T_pixel // 4 + T_pixel % 2（近似）
                num_frames = torch.full((B,), T_lat, dtype=torch.float32, device=t.device)
            else:
                num_frames = torch.full((B,), T_lat, dtype=torch.float32, device=t.device)
    else:
        # fallback：不做变换
        return t

    assert (num_frames >= 1).all(), "num_frames cannot be less than 1"
    ratio_time = (num_frames / base_num_frames).sqrt()

    t_norm = t / num_timesteps                         # [0, 1]
    ratio = ratio_space * ratio_time * scale
    assert (ratio > 0).all(), "ratio cannot be 0"
    new_t = ratio * t_norm / (1 + (ratio - 1) * t_norm)
    return new_t * num_timesteps


class RFlowSchedulerMMDiT:
    """
    Rectified Flow Scheduler，适配 MagicDrive-MMDiT 接口。
    
    与原版 RFlowScheduler 的差异：
      1. training_losses 中的 timestep_transform 支持从 x_start.shape 推断分辨率
      2. model() 调用方式不变（通过 **model_kwargs 传参），但 model_kwargs 的键不同
      3. 支持可变长度视频的 x_mask 处理（逻辑与原版一致）
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
        self.sample_method = sample_method

        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"

        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        if cog_style_trans:
            logging.warning(
                "Use `cog_style_trans`. Please make sure train & inference is consistent!"
            )
        self.cog_style_trans = cog_style_trans

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,         # (B, C*NC, T, H_lat, W_lat)
        model_kwargs: dict = None,
        noise: torch.Tensor = None,
        mask: torch.Tensor = None,      # (B, T) bool，可变长度视频帧 mask
        weights: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> dict:
        """
        计算单步训练 loss。
        
        与原版的差异：
          - timestep_transform 传入 x_shape=x_start.shape（无需 height/width/num_frames）
          - model(x_t, t, **model_kwargs) 调用不变
          - model_kwargs 包含 MMDiT 所需的所有条件键

        Args:
            model:        MagicDriveMMDiT（或其 booster 包装）
            x_start:      干净样本，(B, C*NC, T, H_lat, W_lat)
            model_kwargs: 包含 y, y_vec, maps, bbox, cams, rel_pos,
                          mv_order_map, drop_cond_mask, drop_frame_mask,
                          x_mask（可选），first_frame_latent（可选）
            noise:        噪声（None 则随机生成）
            mask:         (B, T) 帧有效 mask（用于可变长度视频 loss 计算）
            weights:      每步权重（通常 None）
            t:            时间步（None 则随机采样）
        
        Returns:
            dict with key "loss": (B,) float tensor
        """
        if model_kwargs is None:
            model_kwargs = {}

        # ── 采样时间步 ────────────────────────────────────────────────────────
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(
                    0, self.num_timesteps, (x_start.shape[0],), device=x_start.device
                )
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                # MMDiT 适配：传入 x_shape 而非 height/width/num_frames
                t = timestep_transform(
                    t,
                    model_kwargs,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    cog_style=self.cog_style_trans,
                    x_shape=x_start.shape,      # ← 新增
                )

        # ── 生成噪声样本 ──────────────────────────────────────────────────────
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)

        # ── x_mask 处理（可变长度视频）────────────────────────────────────────
        # x_mask: (B, T) bool，0 表示 padding 帧，对应位置保持 t=0 时的 x_t
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            # mask[:, None, :, None, None] → broadcast 到 (B, C*NC, T, H, W)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        # ── 模型前向 ─────────────────────────────────────────────────────────
        # MagicDriveMMDiT.forward(x, timesteps, **model_kwargs)
        # 这里 timesteps = t（float tensor，未归一化）
        # 模型内部会调用 timestep_embedding(timesteps, 256)
        model_output = model(x_t, t, **model_kwargs)

        # 如果模型输出包含 pred_sigma（双通道），只取前一半
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]

        velocity_pred = model_output

        # ── loss 计算（velocity matching）────────────────────────────────────
        # 目标速度: v* = x_start - noise（即 (x1 - x0) 方向）
        target_velocity = x_start - noise

        if weights is None:
            loss = mean_flat((velocity_pred - target_velocity).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t.long(), x_start.shape)
            loss = mean_flat(weight * (velocity_pred - target_velocity).pow(2), mask=mask)

        return {"loss": loss}

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.Tensor,
    ) -> torch.FloatTensor:
        """
        Rectified flow 加噪：x_t = (1 - t/T) * x0 + (t/T) * noise
        等价于 timepoint * x0 + (1 - timepoint) * noise，其中 timepoint = 1 - t/T
        
        与原版完全一致，兼容 diffusers add_noise() 接口。
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [0, 1] → [1, ~0]

        # broadcast: (B,) → (B, C*NC, T, H, W)
        while timepoints.dim() < original_samples.dim():
            timepoints = timepoints.unsqueeze(-1)
        timepoints = timepoints.expand_as(original_samples)

        return timepoints * original_samples + (1 - timepoints) * noise

    def get_sampling_timesteps(self, batch_size: int, device) -> torch.Tensor:
        """
        推理时的时间步序列（从 T 到 0 的线性插值）。
        返回 (num_sampling_steps,) 的时间步序列。
        """
        step_size = self.num_timesteps / self.num_sampling_steps
        timesteps = torch.arange(
            self.num_timesteps, 0, -step_size, device=device
        ).float()
        return timesteps

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        model_kwargs: dict = None,
        guidance_scale: float = 1.0,
        device=None,
        dtype=None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        推理采样（ODE solver，Euler method）。
        
        Args:
            model:    MagicDriveMMDiT
            shape:    输出形状 (B, C*NC, T, H, W)
            model_kwargs: 条件信息
            guidance_scale: 分类器自由引导强度
        
        Returns:
            x_0: 去噪后的样本 (B, C*NC, T, H, W)
        """
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        if dtype is None:
            dtype = next(model.parameters()).dtype

        B = shape[0]
        x = torch.randn(shape, device=device, dtype=dtype)

        # 时间步序列（从 T 降到 0）
        timesteps = self.get_sampling_timesteps(B, device)
        dt = self.num_timesteps / self.num_sampling_steps

        iterator = tqdm(timesteps, desc="Sampling") if progress else timesteps

        for t_val in iterator:
            t = torch.full((B,), t_val, device=device, dtype=dtype)

            # CFG：如果 guidance_scale > 1，做无条件 + 有条件双次推理
            if guidance_scale > 1.0:
                # 有条件
                v_cond = model(x, t, **model_kwargs)
                # 无条件（drop all conditions）
                uncond_kwargs = _make_uncond_kwargs(model_kwargs, B, device, dtype)
                v_uncond = model(x, t, **uncond_kwargs)
                # CFG 混合
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, **model_kwargs)

            if v.shape[1] == 2 * x.shape[1]:
                v = v.chunk(2, dim=1)[0]

            # Euler step: x_{t-dt} = x_t - dt/T * v
            x = x - (dt / self.num_timesteps) * v

        return x


def _make_uncond_kwargs(model_kwargs: dict, B: int, device, dtype) -> dict:
    """
    构造无条件推理的 model_kwargs（将所有条件置零/drop）。
    用于 CFG 推理。
    """
    import torch
    uncond = {}
    for k, v in model_kwargs.items():
        if k in ("y", "y_vec"):
            uncond[k] = torch.zeros_like(v)
        elif k in ("drop_cond_mask",):
            uncond[k] = torch.zeros_like(v)
        elif k in ("drop_frame_mask",):
            uncond[k] = torch.zeros_like(v)
        elif k == "maps":
            uncond[k] = torch.zeros_like(v)
        elif k == "bbox":
            # bbox dict：masks 全为 0（null）
            if v is not None:
                uncond_bbox = {}
                for bk, bv in v.items():
                    if isinstance(bv, torch.Tensor):
                        if bk == "masks":
                            uncond_bbox[bk] = torch.zeros_like(bv)
                        else:
                            uncond_bbox[bk] = bv
                    else:
                        uncond_bbox[bk] = bv
                uncond[k] = uncond_bbox
            else:
                uncond[k] = None
        else:
            uncond[k] = v
    return uncond


# ── 注册到 SCHEDULERS registry ────────────────────────────────────────────────
try:
    from magicdrivedit.registry import SCHEDULERS

    @SCHEDULERS.register_module("rflow_mmdit")
    def build_rflow_mmdit(**kwargs):
        return RFlowSchedulerMMDiT(**kwargs)

except ImportError:
    pass