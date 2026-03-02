# filename: models/localdpo/corrupter.py
# FIXED VERSION:
#   Bug #1 修复: 统一使用 RFlow 噪声调度（删除 DDPM betas/alphas_cumprod）
#   Bug #2 修复: 统一通道排列为 (NC C)，与主训练代码保持一致
#   Bug #3 修复: denoise() 传参过滤，避免把不兼容的 model_args 字段传给冻结模型
#   新增:       add_noise 返回 noise，供 DPO 损失使用

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===== 主训练代码中使用的合法 model forward 参数 =====
_VALID_MODEL_ARGS = {
    "y", "mask", "maps", "bbox", "cams", "rel_pos",
    "fps", "height", "width", "num_frames",
    "drop_cond_mask", "drop_frame_mask",
    "mv_order_map", "t_order_map",
    "x_mask", "frame_idx", "first_frame_images", "frames_mask",
}


def _filter_model_args(model_args: dict) -> dict:
    """只保留模型 forward 接受的合法参数，避免传入 booster 特有字段导致报错。"""
    return {k: v for k, v in model_args.items() if k in _VALID_MODEL_ARGS}


class MultiViewLocalCorrupter(nn.Module):
    """
    多视图局部腐败视频构造器（FIXED）

    核心流程：
        1. 用 RFlow 公式对遮罩区域添加噪声（与主训练一致）
        2. 用冻结的 MagicDrive 模型对加噪区域去噪（修复）
        3. 区域融合：遮罩区域用修复结果，非遮罩区域保留真实视频

    修复说明：
        - 删除 DDPM betas/alphas_cumprod，改用 RFlow 噪声公式
        - 统一通道排列：B (NC*C) T H W（与主训练代码对齐）
        - denoise 传参过滤，只传模型接受的字段
    """

    def __init__(
        self,
        diffusion_model,
        num_timesteps=1000,
        device="cpu",
    ):
        super().__init__()

        # 冻结参考模型（不参与梯度计算）
        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        self.num_timesteps = num_timesteps

    # ------------------------------------------------------------------
    # RFlow 噪声工具（与 RFlowSchedulerWithPred.add_noise 严格一致）
    # ------------------------------------------------------------------
    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_t = 1 - t / num_timesteps，范围 (0, 1]，shape [B]"""
        return 1.0 - t.float() / self.num_timesteps

    def add_noise(
        self,
        video: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        使用 RFlow 公式对视频（或遮罩区域）添加噪声。

        Args:
            video: [B, NC, C, T, H, W]  真实视频（多视图格式）
            t:     [B]                   整数时间步
            mask:  [B, NC, 1, T, H, W]  可选，只对遮罩区域加噪

        Returns:
            noisy:  [B, NC, C, T, H, W]  加噪视频
            noise:  [B, NC, C, T, H, W]  使用的随机噪声
        """
        noise = torch.randn_like(video)

        alpha_t = self._get_alpha(t)           # [B]
        one_minus_alpha = 1.0 - alpha_t        # [B]

        # 扩展到 [B, NC, C, T, H, W]
        shape = video.shape  # (B, NC, C, T, H, W)
        alpha_t_exp = alpha_t.view(-1, *([1] * (len(shape) - 1)))
        one_minus_alpha_exp = one_minus_alpha.view(-1, *([1] * (len(shape) - 1)))

        noisy_all = alpha_t_exp * video + one_minus_alpha_exp * noise

        if mask is not None:
            # 只对遮罩区域加噪，非遮罩区域保留原始视频
            noisy = mask * noisy_all + (1.0 - mask) * video
        else:
            noisy = noisy_all

        return noisy, noise

    # ------------------------------------------------------------------
    # 去噪：调用冻结的参考模型
    # ------------------------------------------------------------------
    def denoise(
        self,
        noisy_video: torch.Tensor,
        t: torch.Tensor,
        condition: dict,
    ) -> torch.Tensor:
        """
        用冻结的 MagicDrive 模型对加噪视频去噪。

        Args:
            noisy_video: [B, NC, C, T, H, W]  加噪的多视图视频
            t:           [B]                   时间步
            condition:   dict                  模型条件（经过过滤）

        Returns:
            denoised:    [B, NC, C, T, H, W]  去噪结果（velocity 预测 → 还原 x0）
        """
        B, NC, C, T, H, W = noisy_video.shape

        # ===== FIXED: 统一通道排列为 (NC C)，与主训练代码一致 =====
        # 主训练代码：x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...")
        # 但实际上模型期望 B (NC*C) T H W（因为在model.forward里会再次重排）
        # 严格对齐主训练循环中的排列方式
        x_input = rearrange(noisy_video, "B NC C T H W -> B (NC C) T H W")

        # ===== FIXED: 过滤 model_args，只传模型接受的字段 =====
        safe_condition = _filter_model_args(condition)

        with torch.no_grad():
            # 模型输出的是 velocity_pred（shape 与 x_input 相同）
            velocity_pred = self.diffusion_model(x_input, t, **safe_condition)

        # 处理双通道输出（pred_sigma=True）
        if velocity_pred.shape[1] == 2 * x_input.shape[1]:
            velocity_pred = velocity_pred.chunk(2, dim=1)[0]

        # ===== 用 RFlow 还原 x0 =====
        # x0_pred = x_t + (1 - alpha_t) * velocity_pred
        alpha_t = self._get_alpha(t)
        one_minus_alpha = 1.0 - alpha_t
        one_minus_alpha_exp = one_minus_alpha.view(-1, *([1] * (x_input.dim() - 1)))
        x0_pred_flat = x_input + one_minus_alpha_exp * velocity_pred

        # 还原多视图格式
        denoised = rearrange(x0_pred_flat, "B (NC C) T H W -> B NC C T H W", NC=NC)
        return denoised

    # ------------------------------------------------------------------
    # 完整腐败 + 修复流程
    # ------------------------------------------------------------------
    @torch.no_grad()
    def corrupt_and_restore(
        self,
        real_video: torch.Tensor,
        mask: torch.Tensor,
        condition: dict,
        t: torch.Tensor = None,
    ):
        """
        腐败并修复完整流程。

        Args:
            real_video:  [B, NC, C, T, H, W]  真实视频（正样本）
            mask:        [B, NC, 1, T, H, W]  多视图运动遮罩
            condition:   dict                  生成条件（model_args）
            t:           [B]（可选）            噪声水平，None 则随机采样

        Returns:
            corrupted:   [B, NC, C, T, H, W]  腐败+修复后的视频（负样本）
            mask:        [B, NC, 1, T, H, W]  使用的遮罩
            t:           [B]                   使用的时间步
            noise:       [B, NC, C, T, H, W]  使用的噪声
        """
        B = real_video.shape[0]
        device = real_video.device

        if t is None:
            # 随机采样噪声水平（参考 LocalDPO 原文，偏向中等噪声水平）
            t = torch.randint(
                self.num_timesteps // 4,
                self.num_timesteps * 3 // 4,
                (B,),
                device=device,
            )

        # 1. 对遮罩区域添加 RFlow 噪声
        noisy, noise = self.add_noise(real_video, t, mask)

        # 2. 用冻结模型修复遮罩区域
        restored = self.denoise(noisy, t, condition)

        # 3. 区域融合：遮罩区域用修复结果，非遮罩区域用真实视频
        #    这样负样本只在遮罩区域与正样本不同（局部DPO的核心思想）
        corrupted = mask * restored + (1.0 - mask) * real_video

        return corrupted, mask, t, noise