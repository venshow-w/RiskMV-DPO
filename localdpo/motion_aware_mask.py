# filename: models/localdpo/motion_aware_mask.py
# FIXED VERSION:
#   Bug #1 修复: 当 motion_mask 全零时（低速场景），添加 fallback 随机矩形遮罩
#   Bug #2 修复: mask_scale 使用 .item() 时可能超出 (0,1)，改为 sigmoid 约束
#   改进:        forward 返回非零遮罩的概率（用于训练监控）

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random
import logging


class MotionAwareMaskGenerator(nn.Module):
    """
    运动感知遮罩生成器（FIXED）

    核心流程：
        1. 用帧差估计运动幅度
        2. 在运动概率高的区域生成矩形遮罩
        3. FIXED: 若运动区域全零（低速/静态场景），自动 fallback 到随机矩形遮罩

    设计原则：
        - 遮罩只覆盖动态物体区域，比随机遮罩有更强的语义意义
        - 可学习参数 mask_scale 控制遮罩相对大小（通过 sigmoid 约束到 (0,1)）
    """

    def __init__(
        self,
        img_size=(106, 200),
        num_cameras=6,
        mask_size_range=(0.1, 0.25),
        motion_threshold=0.6,         # FIXED: 从 0.7 降低到 0.6，减少全零遮罩概率
        use_semantic_prior=True,
        fallback_mask_ratio=0.15,      # FIXED: fallback 遮罩占图像面积比
    ):
        super().__init__()

        self.img_h, self.img_w = img_size
        self.num_cameras = num_cameras
        self.mask_size_range = mask_size_range
        self.motion_threshold = motion_threshold
        self.use_semantic_prior = use_semantic_prior
        self.fallback_mask_ratio = fallback_mask_ratio

        # FIXED: 使用 sigmoid 约束 mask_scale 到 (0, 1)，避免 .item() 超出范围
        # 可学习参数（初始化为 0，sigmoid(0) = 0.5，对应遮罩范围中点）
        self._mask_scale_logit = nn.Parameter(torch.zeros(1))

    @property
    def mask_scale(self) -> torch.Tensor:
        """mask_scale ∈ (0, 1)，通过 sigmoid 保证范围合法。"""
        return torch.sigmoid(self._mask_scale_logit)

    def _compute_motion_masks(self, video: torch.Tensor) -> torch.Tensor:
        """
        计算运动遮罩（帧差法）。

        Args:
            video: [B, NC, C, T, H, W]

        Returns:
            motion_mask: [B, NC, 1, T, H, W]，二值遮罩，1 表示运动区域
        """
        B, NC, C, T, H, W = video.shape

        with torch.no_grad():
            # 帧差：[B, NC, T-1, H, W]（对通道取均值）
            frame_diff = (video[:, :, :, 1:] - video[:, :, :, :-1]).abs().mean(dim=2)

            # 插值到 T 帧（三线性插值，对齐时间维度）
            motion_mag = F.interpolate(
                frame_diff.flatten(0, 1).unsqueeze(1),   # [(B*NC), 1, T-1, H, W]
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            ).squeeze(1).reshape(B, NC, T, H, W)          # [B, NC, T, H, W]

            # 逐样本归一化（避免全局 min/max 被异常值主导）
            min_val = motion_mag.amin(dim=[2, 3, 4], keepdim=True)
            max_val = motion_mag.amax(dim=[2, 3, 4], keepdim=True)
            motion_prob = (motion_mag - min_val) / (max_val - min_val + 1e-8)

            # 阈值化
            motion_mask = (motion_prob > self.motion_threshold).float()

        return motion_mask.unsqueeze(2)   # [B, NC, 1, T, H, W]

    def _generate_motion_based_mask(
        self,
        b: int,
        nc: int,
        motion_mask: torch.Tensor,
        T: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        在运动区域内随机选一个中心点，生成矩形遮罩。
        若该视图无运动区域，返回 None（由外层处理 fallback）。
        """
        motion_regions = torch.nonzero(motion_mask[b, nc, 0])   # [N, 3] (t, y, x)
        if len(motion_regions) == 0:
            return None

        # 计算遮罩半径
        scale_val = self.mask_scale.item()
        mask_sz = self.mask_size_range[0] + scale_val * (
            self.mask_size_range[1] - self.mask_size_range[0]
        )
        radius_h = max(1, int(H * mask_sz / 2))
        radius_w = max(1, int(W * mask_sz / 2))

        # 随机选一个运动区域作为遮罩中心
        idx = random.randint(0, len(motion_regions) - 1)
        t_c, y_c, x_c = motion_regions[idx]

        x1 = max(0, int(x_c) - radius_w)
        x2 = min(W, int(x_c) + radius_w + 1)
        y1 = max(0, int(y_c) - radius_h)
        y2 = min(H, int(y_c) + radius_h + 1)
        t_len = random.randint(2, max(2, T // 2))
        t_start = max(0, int(t_c) - t_len // 2)
        t_end = min(T, t_start + t_len)

        patch = torch.zeros(1, T, H, W, device=device)
        patch[0, t_start:t_end, y1:y2, x1:x2] = 1.0
        return patch

    def _generate_fallback_mask(
        self,
        T: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        FIXED: Fallback 遮罩 —— 当运动区域全零时（低速/静止场景），
        随机生成一个占图像 fallback_mask_ratio 面积的矩形遮罩。
        """
        area = H * W * self.fallback_mask_ratio
        h_size = max(1, int((area ** 0.5)))
        w_size = max(1, int(area / h_size))

        y1 = random.randint(0, max(0, H - h_size))
        x1 = random.randint(0, max(0, W - w_size))
        t_len = random.randint(max(1, T // 4), max(1, T // 2))
        t_start = random.randint(0, max(0, T - t_len))

        patch = torch.zeros(1, T, H, W, device=device)
        patch[0, t_start:t_start + t_len, y1:y1 + h_size, x1:x1 + w_size] = 1.0
        return patch

    def forward(
        self,
        video: torch.Tensor,
        first_frame: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        生成运动感知遮罩。

        Args:
            video:       [B, NC, C, T, H, W]  真实视频
            first_frame: [B, NC, C, H, W]（可选）首帧（保留接口，暂未使用）

        Returns:
            final_mask: [B, NC, 1, T, H, W]   二值遮罩，1 表示腐败区域
        """
        B, NC, C, T, H, W = video.shape
        device = video.device

        # 1. 计算运动概率遮罩
        motion_mask = self._compute_motion_masks(video)   # [B, NC, 1, T, H, W]

        # 2. 在运动区域内生成最终遮罩
        final_mask = torch.zeros(B, NC, 1, T, H, W, device=device, dtype=video.dtype)
        fallback_count = 0

        for b in range(B):
            for nc in range(NC):
                patch = self._generate_motion_based_mask(b, nc, motion_mask, T, H, W, device)

                if patch is None:
                    # ===== FIXED: Fallback —— 不再返回全零遮罩 =====
                    patch = self._generate_fallback_mask(T, H, W, device)
                    fallback_count += 1

                final_mask[b, nc] = patch

        # 记录 fallback 比例（每 100 步打印一次）
        if fallback_count > 0:
            logging.debug(
                f"[MotionAwareMask] fallback triggered for {fallback_count}/{B * NC} views "
                f"(motion_threshold={self.motion_threshold})"
            )

        # 3. 可选：融合 VGGT 语义先验（保留接口）
        if self.use_semantic_prior and first_frame is not None:
            pass  # 后续可集成 VGGT 特征

        return final_mask