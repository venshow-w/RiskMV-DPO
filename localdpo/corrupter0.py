# filename: models/localdpo/corrupter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiViewLocalCorrupter(nn.Module):
    """
    创新点2：多视图局部腐败视频构造
    - 对真实视频的遮罩区域添加噪声
    - 用冻结的MagicDrive模型修复
    - 区域融合得到负样本
    """
    def __init__(self, 
                 diffusion_model,
                 noise_schedule='linear',
                 num_timesteps=1000,
                 device='cpu'):
        super().__init__()
        
        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()  # 冻结！
        
         # 噪声调度（修复：确保tensor在正确设备）
        self.register_buffer('betas', self._get_betas(noise_schedule, num_timesteps).to(device))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, 0))
        self.num_timesteps = num_timesteps
    
    def _get_betas(self, schedule, num_timesteps):
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, num_timesteps)
        elif schedule == 'cosine':
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            return torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
    def add_noise(self, video, t, mask=None):
        """
        向视频添加噪声
        video: [B, NC, C, T, H, W] 真实视频
        t: 噪声水平 [B] 0-1000
        mask: 遮罩区域 [B, NC, 1, T, H, W]
        """
        noise = torch.randn_like(video)
        
        sqrt_alphas_cumprod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod[t]) ** 0.5
        
        # 调整维度
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1, 1, 1)
        
        noisy = sqrt_alphas_cumprod * video + sqrt_one_minus_alphas_cumprod * noise
        
        if mask is not None:
            # 只对遮罩区域添加噪声
            noisy = mask * noisy + (1 - mask) * video
            self.last_noise = noise
            self.last_mask = mask
            self.last_t = t
        
        return noisy
    
    def denoise(self, noisy_video, t, condition):
        """
        用MagicDrive模型去噪
        noisy_video: [B, NC, C, T, H, W] 加噪视频
        t: 时间步 [B]
        condition: 文本/控制条件
        """
        # 转换为MagicDrive期望的格式
        B, NC, C, T, H, W = noisy_video.shape
        x = rearrange(noisy_video, 'B NC C T H W -> B (C NC) T H W',B=B,NC=NC,C=C, T=T)
        
        # # 重复条件
        # cond = {}
        # for k, v in condition.items():
        #     if isinstance(v, torch.Tensor):
        #         cond[k] = v.repeat_interleave(NC, dim=0)
        #     else:
        #         cond[k] = v
        
        # 模型前向
        
        with torch.no_grad():
            denoised = self.diffusion_model(
                x, 
                t, #.repeat_interleave(NC),
                **condition
            )
        denoised = rearrange(denoised, 'B (NC C) T H W -> B NC C T H W', NC=NC)
        return denoised
    
    @torch.no_grad()
    def corrupt_and_restore(self, real_video, mask, condition, t=None):
        """
        腐败并修复完整流程
        real_video: 真实视频（正样本）
        mask: 多视图遮罩s
        condition: 生成条件
        t: 噪声水平（随机采样）
        """
        B = real_video.shape[0] 
        device = real_video.device

        if t is None:
            # 随机采样噪声水平（LocalDPO原文）
            t = torch.randint(0, len(self.betas), (B,), device=real_video.device)
        
        # 1. 只对遮罩区域添加噪声
        noisy = self.add_noise(real_video, t, mask)
        
        # 2. 模型修复遮罩区域
        restored = self.denoise(noisy, t, condition)
        
        # 3. 区域融合：遮罩区域用修复结果，非遮罩区域用真实视频
        corrupted = mask * restored + (1 - mask) * real_video
        
        return corrupted, mask, t, self.last_noise