"""
magicdrive_stdit3_v13.py
====================================
MagicDriveV2 × OpenSora v1.3 (STDiT3-XL/2)

替换 OpenSora 2.0 (MMDiT/Flux, ~11B) → OpenSora v1.3 (STDiT3-XL/2, ~600M)
保持所有控制条件：maps / cam_params / bboxes / text

架构说明
─────────────────────────────────────────────────────────────────────────────
原始 MagicDriveSTDiT3 已经就是基于 STDiT3-XL/2 的，本文件做了以下改动：

1. 彻底移除 MMDiT/Flux 依赖，回归 STDiT3 骨干
2. 修复 load_from_stdit3_pretrained：支持直接从 OpenSora v1.3 的
   checkpoint 文件（.pt / .pth）加载，不再需要 HuggingFace 格式
3. 新增 set_trainable_parameters(mode) 方法，对齐前面 MMDiT 版本接口
4. 修复 EMA 构建：build_ema_on_cpu_v2 已在 ema_utils.py 中实现
5. 所有控制条件 (maps/cam/bbox/text) 的编码路径与原版完全一致

参数量对比
─────────────────────────────────────────────────────────────────────────────
OpenSora v2.0 MMDiT (Flux-large)  ≈ 11B   → 单卡 GPU 22 GB (bf16)
OpenSora v1.3 STDiT3-XL/2        ≈ 0.6B  → 单卡 GPU  1.2 GB (bf16)
MagicDrive 控制条件模块           ≈ 0.2B
总计 (STDiT3 版)                  ≈ 0.8B  → 单卡 GPU  1.6 GB (bf16)

6卡 + 控制条件 + 激活值 + 优化器状态，完全在 A100 80GB 内

使用方法
─────────────────────────────────────────────────────────────────────────────
在 config 中:
    model = dict(
        type="MagicDriveSTDiT3v13-XL/2",
        from_opensora13_pretrained="/path/to/OpenSora-STDiT-v3/model.pt",
        ...  # 其余参数与原版 MagicDriveSTDiT3-XL/2 完全相同
    )
"""

import os
import logging
import random
from typing import Optional, Dict

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel

from magicdrivedit.acceleration.checkpoint import auto_grad_checkpoint
from magicdrivedit.acceleration.communications import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from magicdrivedit.acceleration.parallel_states import get_sequence_parallel_group
from magicdrivedit.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    MultiHeadAttention,
    SeqParallelMultiHeadAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from magicdrivedit.registry import MODELS
from magicdrivedit.utils.ckpt_utils import load_checkpoint
from magicdrivedit.utils.misc import warn_once

from .embedder import MapControlTempEmbedding
from .utils import zero_module, load_module


# ─────────────────────────────────────────────────────────────────────────────
# Block：与原版 MultiViewSTDiT3Block 完全一致，无改动
# （完整保留，保证 state_dict key 名称对齐 OpenSora v1.3 权重）
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewSTDiT3Block(nn.Module):
    """
    STDiT3 block with multi-view cross-attention.
    与原始 MagicDriveSTDiT3 中的 MultiViewSTDiT3Block 完全相同，
    保证权重加载时 key 名字一致。
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        sequence_parallelism_temporal=True,
        rope=None,
        qk_norm=False,
        temporal=False,
        is_control_block=False,
        use_st_cross_attn=False,
        skip_cross_view=False,
        first_frame_condition=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.is_control_block = is_control_block
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        assert not use_st_cross_attn, "STDiT3 does not support st_cross_attn."
        self.use_st_cross_attn = use_st_cross_attn
        self.skip_cross_view = skip_cross_view or self.temporal
        self.first_frame_condition = first_frame_condition

        if enable_sequence_parallelism:
            attn_cls = fmha_cls = SeqParallelMultiHeadAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = fmha_cls = MultiHeadAttention
            mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(
            hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel
        )
        if temporal:
            _this_attn_cls = attn_cls if sequence_parallelism_temporal else Attention
        else:
            _this_attn_cls = attn_cls
        self.attn = _this_attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
            is_cross_attention=False,
        )

        _this_attn_cls = MultiHeadCrossAttention if sequence_parallelism_temporal else mha_cls
        self.cross_attn = _this_attn_cls(hidden_size, num_heads)

        self.norm2 = get_layernorm(
            hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )

        if not self.skip_cross_view:
            self.norm3 = get_layernorm(
                hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel
            )
            _this_attn_cls = Attention if sequence_parallelism_temporal else fmha_cls
            self.cross_view_attn = _this_attn_cls(
                hidden_size,
                num_heads=num_heads,
                qk_norm=True,
                enable_flash_attn=enable_flash_attn,
                enable_xformers=enable_xformers,
                is_cross_attention=True,
            )
            self.mva_proj = zero_module(nn.Linear(hidden_size, hidden_size))
        else:
            self.mva_proj = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5
        )
        if not self.skip_cross_view:
            self.scale_shift_table_mva = nn.Parameter(
                torch.randn(3, hidden_size) / hidden_size ** 0.5
            )
        if is_control_block:
            self.after_proj = zero_module(nn.Linear(hidden_size, hidden_size))
        else:
            self.after_proj = None

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        return rearrange(x, "B T S C -> B (T S) C")

    def _construct_attn_input_from_map(self, h, order_map, cat_seq=False):
        B = len(h)
        h_q, h_kv, back_order = [], [], []
        for target, values in order_map.items():
            if cat_seq:
                h_q.append(h[:, target])
                h_kv.append(torch.cat([h[:, v] for v in values], dim=1))
                back_order += [target] * B
            else:
                for neighbor in values:
                    h_q.append(h[:, target])
                    h_kv.append(h[:, neighbor])
                    back_order += [target] * B
        h_q = torch.cat(h_q, dim=0)
        h_kv = torch.cat(h_kv, dim=0)
        back_order = torch.LongTensor(back_order)
        return h_q, h_kv, back_order

    def forward(self, x, y, t, mask=None, x_mask=None, t0=None,
                T=None, S=None, NC=None, mv_order_map=None, t_order_map=None):
        B, N, C = x.shape
        assert (N == T * S) and (B % NC == 0)
        b = B // NC

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = repeat(
            self.scale_shift_table[None] + t.reshape(b, 6, -1),
            "b ... -> (b NC) ...", NC=NC,
        ).chunk(6, dim=1)
        if x_mask is not None:
            (shift_msa_zero, scale_msa_zero, gate_msa_zero,
             shift_mlp_zero, scale_mlp_zero, gate_mlp_zero) = repeat(
                self.scale_shift_table[None] + t0.reshape(b, 6, -1),
                "b ... -> (b NC) ...", NC=NC,
            ).chunk(6, dim=1)

        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # self-attention (spatial or temporal)
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)
        x = x + self.drop_path(x_m_s)

        # cross-attention with text / condition tokens
        assert mask is None
        if y.shape[1] == 1:
            x_c = self.cross_attn(x, y[:, 0], mask)
        elif y.shape[1] == T:
            x_c = rearrange(x, "B (T S) C -> (B T) S C", T=T, S=S)
            y_c = rearrange(y, "B T L C -> (B T) L C", T=T)
            x_c = self.cross_attn(x_c, y_c, mask)
            x_c = rearrange(x_c, "(B T) S C -> B (T S) C", T=T, S=S)
        else:
            raise RuntimeError(f"unsupported y.shape[1]={y.shape[1]}")
        x = x + x_c

        # multi-view cross-attention
        if not self.skip_cross_view:
            assert mv_order_map is not None
            shift_mva, scale_mva, gate_mva = repeat(
                self.scale_shift_table_mva[None] + t[:, :3].reshape(b, 3, -1),
                "b ... -> (b NC) ...", NC=NC,
            ).chunk(3, dim=1)
            if x_mask is not None:
                shift_mva_zero, scale_mva_zero, gate_mva_zero = repeat(
                    self.scale_shift_table_mva[None] + t0[:, :3].reshape(b, 3, -1),
                    "b ... -> (b NC) ...", NC=NC,
                ).chunk(3, dim=1)

            x_v = t2i_modulate(self.norm3(x), shift_mva, scale_mva)
            if x_mask is not None:
                x_v_zero = t2i_modulate(self.norm3(x), shift_mva_zero, scale_mva_zero)
                x_v = self.t_mask_select(x_mask, x_v, x_v_zero, T, S)

            x_mv = rearrange(x_v, "(B NC) (T S) C -> (B T) NC S C", NC=NC, T=T)
            x_targets, x_neighbors, cam_order = self._construct_attn_input_from_map(
                x_mv, mv_order_map, cat_seq=False
            )
            raw_out = self.cross_view_attn(x_targets, x_neighbors)
            out_mv = torch.zeros_like(x_mv)
            for cam_i in range(NC):
                attn_out_mv = rearrange(
                    raw_out[cam_order == cam_i],
                    "(n_neighbors b) ... -> b n_neighbors ...",
                    b=B // NC * T,
                )
                out_mv[:, cam_i] = torch.sum(attn_out_mv, dim=1)
            out_mv = rearrange(out_mv, "(B T) NC S C -> (B NC) (T S) C", T=T)

            x_v_s = gate_mva * out_mv
            if x_mask is not None:
                x_v_s_zero = gate_mva_zero * out_mv
                x_v_s = self.t_mask_select(x_mask, x_v_s, x_v_s_zero, T, S)
            x = x + self.mva_proj(self.drop_path(x_v_s))

        # MLP
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)
        breakpoint()
        x_m = self.mlp(x_m)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)
        x = x + self.drop_path(x_m_s)

        if self.is_control_block:
            return x, self.after_proj(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class MagicDriveSTDiT3v13Config(PretrainedConfig):
    model_type = "MagicDriveSTDiT3v13"

    def __init__(
        self,
        # STDiT3-XL/2 default dims
        input_size=(1, 32, 32),
        input_sq_size=512,
        force_pad_h_for_sp_size=None,
        simulate_sp_size=[],
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        freeze_y_embedder=False,
        # magicdrive blocks
        with_temp_block=True,
        freeze_x_embedder=False,
        freeze_old_embedder=False,
        freeze_temporal_blocks=False,
        freeze_old_params=False,
        zero_and_train_embedder=None,
        only_train_base_blocks=False,
        only_train_temp_blocks=False,
        qk_norm_trainable=False,
        sequence_parallelism_temporal=False,
        control_depth=13,
        use_x_control_embedder=False,
        use_st_cross_attn=False,
        uncond_cam_in_dim=(3, 7),
        cam_encoder_cls=None,
        cam_encoder_param={},
        frame_emb_cls=None,
        frame_emb_param={},
        bbox_embedder_cls=None,
        bbox_embedder_param={},
        map_embedder_cls=None,
        map_embedder_param={},
        map_embedder_downsample_rate=4,
        micro_frame_size=17,
        control_skip_cross_view=True,
        control_skip_temporal=True,
        first_frame_condition=False,
        # ── v1.3 新增 ──
        trainable_mode="new_only",   # "new_only" | "all" | "control"
        drop_cond_ratio=0.15,
        drop_frame_ratio=0.4,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.force_pad_h_for_sp_size = force_pad_h_for_sp_size
        self.simulate_sp_size = simulate_sp_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.enable_xformers = enable_xformers
        self.freeze_y_embedder = freeze_y_embedder
        self.first_frame_condition = first_frame_condition
        self.with_temp_block = with_temp_block
        self.freeze_x_embedder = freeze_x_embedder
        self.freeze_old_embedder = freeze_old_embedder
        self.freeze_temporal_blocks = freeze_temporal_blocks
        self.freeze_old_params = freeze_old_params
        self.zero_and_train_embedder = zero_and_train_embedder
        self.only_train_base_blocks = only_train_base_blocks
        self.only_train_temp_blocks = only_train_temp_blocks
        self.qk_norm_trainable = qk_norm_trainable
        self.sequence_parallelism_temporal = sequence_parallelism_temporal
        self.control_depth = control_depth
        self.use_x_control_embedder = use_x_control_embedder
        self.use_st_cross_attn = use_st_cross_attn
        self.uncond_cam_in_dim = uncond_cam_in_dim
        self.cam_encoder_cls = cam_encoder_cls
        self.cam_encoder_param = cam_encoder_param
        self.frame_emb_cls = frame_emb_cls
        self.frame_emb_param = frame_emb_param
        self.bbox_embedder_cls = bbox_embedder_cls
        self.bbox_embedder_param = bbox_embedder_param
        self.map_embedder_cls = map_embedder_cls
        self.map_embedder_param = map_embedder_param
        self.map_embedder_downsample_rate = map_embedder_downsample_rate
        self.micro_frame_size = micro_frame_size
        self.control_skip_cross_view = control_skip_cross_view
        self.control_skip_temporal = control_skip_temporal
        self.trainable_mode = trainable_mode
        self.drop_cond_ratio = drop_cond_ratio
        self.drop_frame_ratio = drop_frame_ratio
        super().__init__(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class MagicDriveSTDiT3v13(PreTrainedModel):
    """
    MagicDriveV2 with OpenSora v1.3 (STDiT3-XL/2) backbone.

    相比原版 MagicDriveSTDiT3 的改动：
    1. 新增 set_trainable_parameters(mode) —— 对齐 MMDiT 版本接口
    2. 新增 load_from_opensora13_pretrained —— 直接从 .pt 文件加载
    3. config 新增 trainable_mode 参数
    4. 其余结构与原版完全一致（block / embedder / control 全部保留）
    """
    config_class = MagicDriveSTDiT3v13Config

    def __init__(self, config: MagicDriveSTDiT3v13Config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.first_frame_condition = config.first_frame_condition

        self.depth = config.depth
        self.control_depth = config.control_depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        self.enable_flash_attn = config.enable_flash_attn
        self.enable_xformers = config.enable_xformers
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism
        self.sequence_parallelism_temporal = config.sequence_parallelism_temporal

        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(self.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)
        self.force_pad_h_for_sp_size = config.force_pad_h_for_sp_size
        self.simu_sp_size = config.simulate_sp_size

        # ── embeddings ──────────────────────────────────────────────────────
        self.x_embedder = PatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )
        self.fps_embedder = SizeEmbedder(self.hidden_size)

        if config.use_x_control_embedder:
            self.x_control_embedder = PatchEmbed3D(
                self.patch_size, self.in_channels, self.hidden_size
            )
        else:
            self.x_control_embedder = None

        self.register_buffer("base_token", torch.randn(self.hidden_size))

        # condition encoders
        self.camera_embedder = load_module(config.cam_encoder_cls)(
            out_dim=config.hidden_size, **config.cam_encoder_param
        )
        self.frame_embedder = load_module(config.frame_emb_cls)(
            out_dim=config.hidden_size, **config.frame_emb_param
        )
        self.bbox_embedder = load_module(config.bbox_embedder_cls)(
            **config.bbox_embedder_param
        )
        self.controlnet_cond_embedder = load_module(config.map_embedder_cls)(
            conditioning_embedding_channels=self.hidden_size // 2,
            **config.map_embedder_param,
        )
        self.micro_frame_size = config.micro_frame_size
        self.controlnet_cond_embedder_temp = MapControlTempEmbedding(
            self.hidden_size, config.map_embedder_downsample_rate
        )
        self.controlnet_cond_patchifier = PatchEmbed3D(
            self.patch_size, self.hidden_size, self.hidden_size
        )

        # ── base blocks ──────────────────────────────────────────────────────
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, self.depth)]
        _block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            enable_flash_attn=self.enable_flash_attn,
            enable_xformers=self.enable_xformers,
            enable_layernorm_kernel=self.enable_layernorm_kernel,
            enable_sequence_parallelism=self.enable_sequence_parallelism,
            sequence_parallelism_temporal=self.sequence_parallelism_temporal,
            qk_norm=config.qk_norm,
        )
        self.base_blocks_s = nn.ModuleList([
            MultiViewSTDiT3Block(
                drop_path=drop_path[i],
                use_st_cross_attn=config.use_st_cross_attn,
                **_block_kwargs,
            )
            for i in range(self.depth)
        ])
        if config.with_temp_block:
            self.base_blocks_t = nn.ModuleList([
                MultiViewSTDiT3Block(
                    drop_path=drop_path[i],
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                    **_block_kwargs,
                )
                for i in range(self.depth)
            ])
        else:
            self.base_blocks_t = None

        # ── control blocks ───────────────────────────────────────────────────
        self.before_proj = zero_module(nn.Linear(self.hidden_size, self.hidden_size))
        ctrl_drop_path = [
            x.item() for x in torch.linspace(0, config.drop_path, self.control_depth)
        ]
        self.control_blocks_s = nn.ModuleList([
            MultiViewSTDiT3Block(
                drop_path=ctrl_drop_path[i],
                is_control_block=True,
                use_st_cross_attn=config.use_st_cross_attn,
                skip_cross_view=config.control_skip_cross_view,
                **_block_kwargs,
            )
            for i in range(self.control_depth)
        ])
        if config.control_skip_temporal:
            self.control_blocks_t = None
        else:
            self.control_blocks_t = nn.ModuleList([
                MultiViewSTDiT3Block(
                    drop_path=ctrl_drop_path[i],
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                    is_control_block=True,
                    **_block_kwargs,
                )
                for i in range(self.control_depth)
            ])

        # ── final layer ──────────────────────────────────────────────────────
        self.final_layer = T2IFinalLayer(
            self.hidden_size, np.prod(self.patch_size), self.out_channels
        )

        # ── init ─────────────────────────────────────────────────────────────
        self.initialize_weights()

        # ── freeze/train policy ──────────────────────────────────────────────
        if config.freeze_y_embedder:
            for p in self.y_embedder.parameters():
                p.requires_grad = False
        if config.freeze_x_embedder:
            for p in self.x_embedder.parameters():
                p.requires_grad = False

        # apply trainable_mode (mirrors MMDiT version interface)
        self.set_trainable_parameters(config.trainable_mode)

    # ─────────────────────────────────────────────────────────────────────────
    # Trainable parameter control
    # ─────────────────────────────────────────────────────────────────────────

    def set_trainable_parameters(self, mode: str):
        """
        控制哪些参数参与训练，与 MMDiT 版本接口一致。

        mode:
          "new_only"     ── 只训练 MagicDrive 新增部分（cross_view_attn +
                            control_blocks + 所有条件编码器），冻结骨干
          "control"      ── 只训练 control_blocks + 条件编码器
          "all"          ── 全量训练
          "freeze_base"  ── 冻结 base_blocks，其余可训练
        """
        logging.info(f"[MagicDriveSTDiT3v13] set_trainable_parameters(mode={mode!r})")

        if mode == "all":
            for p in self.parameters():
                p.requires_grad = True
            return

        # 先全部冻结
        for p in self.parameters():
            p.requires_grad = False

        # 永远可训练：MagicDrive 新增模块
        _new_modules = [
            self.control_blocks_s,
            self.before_proj,
            self.camera_embedder,
            self.frame_embedder,
            self.bbox_embedder,
            self.controlnet_cond_embedder,
            self.controlnet_cond_embedder_temp,
            self.controlnet_cond_patchifier,
        ]
        if self.control_blocks_t is not None:
            _new_modules.append(self.control_blocks_t)
        if self.x_control_embedder is not None:
            _new_modules.append(self.x_control_embedder)
        for mod in _new_modules:
            for p in mod.parameters():
                p.requires_grad = True

        if mode == "control":
            return

        # "new_only" 或 "freeze_base"：还要打开 cross_view_attn
        if mode in ("new_only", "freeze_base"):
            for block in self.base_blocks_s:
                if hasattr(block, "cross_view_attn"):
                    for p in block.norm3.parameters():
                        p.requires_grad = True
                    for p in block.cross_view_attn.parameters():
                        p.requires_grad = True
                    for p in block.mva_proj.parameters():
                        p.requires_grad = True
                    block.scale_shift_table_mva.requires_grad = True

        if mode == "freeze_base":
            # 除 base_blocks 外都解冻
            for p in self.t_embedder.parameters():
                p.requires_grad = True
            for p in self.t_block.parameters():
                p.requires_grad = True
            for p in self.fps_embedder.parameters():
                p.requires_grad = True
            for p in self.y_embedder.parameters():
                p.requires_grad = True
            for p in self.final_layer.parameters():
                p.requires_grad = True
            if self.base_blocks_t is not None:
                for p in self.base_blocks_t.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logging.info(
            f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
            f"({trainable/total*100:.1f}%)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Weight initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        def _zero_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for block in self.base_blocks_s:
            _zero_init(block.mva_proj)
            assert block.after_proj is None

        if self.base_blocks_t is not None:
            for block in self.base_blocks_t:
                assert block.mva_proj is None
                assert block.after_proj is None
                _zero_init(block.attn.proj)
                _zero_init(block.cross_attn.proj)
                _zero_init(block.mlp.fc2)
            logging.info("base_blocks_t uses zero init.")

        for block in self.control_blocks_s:
            _zero_init(block.mva_proj)
            _zero_init(block.after_proj)
        if self.control_blocks_t is not None:
            for block in self.control_blocks_t:
                assert block.mva_proj is None
                _zero_init(block.after_proj)

        _zero_init(self.before_proj)
        _zero_init(self.bbox_embedder.final_proj)
        _zero_init(self.camera_embedder.after_proj)
        _zero_init(self.frame_embedder.final_proj)

        w = self.controlnet_cond_patchifier.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.bbox_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.bbox_embedder.mlp.fc2.weight, std=0.02)
        nn.init.normal_(self.frame_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.frame_embedder.mlp.fc2.weight, std=0.02)
        nn.init.normal_(self.camera_embedder.emb2token.weight, std=0.02)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility helpers (identical to original MagicDriveSTDiT3)
    # ─────────────────────────────────────────────────────────────────────────

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        return T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]

    def sample_box_latent(self, n_boxes, generator=None):
        if self.bbox_embedder.mean_var is None:
            return None
        return torch.randn(
            (n_boxes, self.bbox_embedder.box_latent_shape[1]), generator=generator
        )

    def encode_text(self, y, mask=None, drop_cond_mask=None):
        if drop_cond_mask is not None:
            y = self.y_embedder(y, False, force_drop_ids=1 - drop_cond_mask)
        else:
            y = self.y_embedder(y, False)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y_lens = [i + 1 for i in mask.sum(dim=1).tolist()]
            max_len = int(min(max(y_lens), y.shape[2]))
            if drop_cond_mask is not None and not drop_cond_mask.all():
                assert max_len == y.shape[2]
            y = y.squeeze(1)[:, :max_len]
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1)
        return y, y_lens

    def encode_box(self, bboxes, drop_mask):
        B, T, seq_len = bboxes["bboxes"].shape[:3]
        bbox_embedder_kwargs = {k: v.clone() for k, v in bboxes.items()}
        drop_mask = repeat(drop_mask, "B T -> B T S", S=seq_len)
        _null_mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _null_mask[bbox_embedder_kwargs["masks"] == 0] = 0
        _mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _mask[bbox_embedder_kwargs["masks"] == -1] = 0
        _mask[
            torch.logical_and(
                bbox_embedder_kwargs["masks"] == 1, drop_mask == 0
            )
        ] = 0
        return self.bbox_embedder(
            bboxes=bbox_embedder_kwargs["bboxes"],
            classes=bbox_embedder_kwargs["classes"].type(torch.int32),
            null_mask=_null_mask,
            mask=_mask,
            box_latent=bbox_embedder_kwargs.get("box_latent", None),
        )

    def encode_cam(self, cam, embedder, drop_mask):
        B, T, S = cam.shape[:3]
        NC = B // drop_mask.shape[0]
        mask = repeat(drop_mask, "b T -> (b NC T S)", NC=NC, S=S)
        cam = rearrange(cam, "B T S ... -> (B T S) ...")
        cam_emb, _ = embedder.embed_cam(cam, mask, T=T, S=S)
        return cam_emb

    def encode_cond_sequence(self, bbox, cams, rel_pos, y, mask,
                              drop_cond_mask, drop_frame_mask):
        b = len(y)
        NC, T = cams.shape[0] // b, cams.shape[1]
        cond = []

        y, _ = self.encode_text(y, mask, drop_cond_mask)
        y = repeat(y, "b ... -> (b NC) ...", NC=NC)

        if bbox is not None:
            drop_box_mask = torch.logical_and(
                drop_cond_mask[:, None], drop_frame_mask
            )
            drop_box_mask = repeat(drop_box_mask, "b ... -> (b NC) ...", NC=NC)
            bbox_emb = self.encode_box(bbox, drop_mask=drop_box_mask)
            bbox_emb = self.base_token[None, None, None] + bbox_emb
            cond.append(bbox_emb)

        cam_emb = self.encode_cam(
            cams[:, 0:1], self.camera_embedder,
            repeat(drop_cond_mask, "b -> b T", T=1),
        )
        frame_emb = self.encode_cam(rel_pos, self.frame_embedder, drop_frame_mask)
        cam_emb = rearrange(cam_emb, "(B 1 S) ... -> B 1 S ...", S=cams.shape[2])
        cam_emb = self.base_token[None, None, None] + cam_emb
        frame_emb = self.base_token[None, None, None] + frame_emb

        cam_emb = repeat(cam_emb, "B 1 S ... -> B T S ...", T=frame_emb.shape[1])
        y = repeat(y, "B ... -> B T ...", T=frame_emb.shape[1])
        cond = [frame_emb, cam_emb, y] + cond
        cond = torch.cat(cond, dim=2)
        return cond, None

    def encode_map(self, maps, NC, h_pad_size, x_shape):
        b, T = maps.shape[:2]
        maps = rearrange(maps, "b T ... -> (b T) ...")
        controlnet_cond = self.controlnet_cond_embedder(maps)
        controlnet_cond = rearrange(controlnet_cond, "(b T) C ... -> b C T ...", T=T)
        if self.micro_frame_size is None:
            controlnet_cond = self.controlnet_cond_embedder_temp(controlnet_cond)
        else:
            z_list = []
            for i in range(0, controlnet_cond.shape[2], self.micro_frame_size):
                z = self.controlnet_cond_embedder_temp(
                    controlnet_cond[:, :, i: i + self.micro_frame_size]
                )
                z_list.append(z)
            controlnet_cond = torch.cat(z_list, dim=2)

        if controlnet_cond.shape[-3:] != x_shape[-3:]:
            warn_once(
                f"Interpolating map cond from {controlnet_cond.shape[-3:]} "
                f"to {x_shape[-3:]}"
            )
            if DEVICE_TYPE == "npu":
                dtype = controlnet_cond.dtype
                controlnet_cond = controlnet_cond.to(torch.float32)
            if (
                np.prod(x_shape[-3:]) > np.prod([33, 106, 200])
                and controlnet_cond.shape[0] > 1
            ):
                _cc = []
                for ci in range(controlnet_cond.shape[0]):
                    _cc.append(F.interpolate(controlnet_cond[ci: ci + 1], x_shape[-3:]))
                controlnet_cond = torch.cat(_cc, dim=0)
            else:
                controlnet_cond = F.interpolate(controlnet_cond, x_shape[-3:])
            if DEVICE_TYPE == "npu":
                controlnet_cond = controlnet_cond.to(dtype)

        if h_pad_size > 0:
            hx_pad_size = h_pad_size * self.patch_size[1]
            controlnet_cond = F.pad(controlnet_cond, (0, 0, 0, hx_pad_size))
        controlnet_cond = self.controlnet_cond_patchifier(controlnet_cond)
        controlnet_cond = repeat(controlnet_cond, "b ... -> (b NC) ...", NC=NC)
        return controlnet_cond

    def prepare_text_embedding(self, text_encoder):
        @torch.no_grad()
        def text_to_embedding(text):
            ret = text_encoder.encode(text)
            hidden_state, _ = self.encode_text(ret["y"], mask=None)
            return hidden_state[:, : int(ret["mask"].sum(dim=1))]

        _training = self.training
        self.training = False
        self.bbox_embedder.prepare(text_to_embedding)
        self.base_token[:] = text_to_embedding("").squeeze()
        self.training = _training

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self, x, timestep, y, maps, bbox, cams, rel_pos, fps,
        height, width,
        drop_cond_mask=None,
        drop_frame_mask=None,
        mv_order_map=None,
        t_order_map=None,
        mask=None,
        x_mask=None,
        first_frame_latent=None,
        frames_mask=None,
        **kwargs,
    ):
        dtype = self.x_embedder.proj.weight.dtype
        B, real_T = x.size(0), rel_pos.size(1)

        if drop_cond_mask is None:
            drop_cond_mask = torch.ones((B,), device=x.device, dtype=x.dtype)
        if drop_frame_mask is None:
            drop_frame_mask = torch.ones((B, real_T), device=x.device, dtype=x.dtype)

        NC = len(mv_order_map)
        x = x.to(dtype)
        x = rearrange(x, "B (C NC) T ... -> (B NC) C T ...", NC=NC)

        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # dynamic size
        _, _, Tx, Hx, Wx = x.size()
        x_in_shape = x.shape
        T, H, W = self.get_dynamic_size(x)
        S = H * W

        # SP padding
        h_pad_size = 0
        if self.training:
            _simu_sp_size = self.simu_sp_size
        else:
            _simu_sp_size = []

        if self.force_pad_h_for_sp_size is not None:
            if S % self.force_pad_h_for_sp_size != 0:
                h_pad_size = self.force_pad_h_for_sp_size - H % self.force_pad_h_for_sp_size
        elif len(_simu_sp_size) > 0:
            if self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
                sp_size = dist.get_world_size(get_sequence_parallel_group())
                possible = [s for s in _simu_sp_size if s >= sp_size]
            else:
                possible = _simu_sp_size
            simu_sp_size = random.choice(possible)
            if S % simu_sp_size != 0:
                h_pad_size = simu_sp_size - H % simu_sp_size
        elif self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if S % sp_size != 0:
                h_pad_size = sp_size - H % sp_size

        if h_pad_size > 0:
            hx_pad_size = h_pad_size * self.patch_size[1]
            x = F.pad(x, (0, 0, 0, hx_pad_size))
            H += h_pad_size
            S = H * W
            if self.enable_sequence_parallelism and not self.sequence_parallelism_temporal:
                sp_size = dist.get_world_size(get_sequence_parallel_group())
                assert S % sp_size == 0

        # positional embedding
        base_size = round(S ** 0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # timestep embedding
        t = self.t_embedder(timestep, dtype=x.dtype)
        fps_emb = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps_emb
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0 = self.t_embedder(torch.zeros_like(timestep), dtype=x.dtype) + fps_emb
            t0_mlp = self.t_block(t0)

        # condition encoding
        y, y_lens = self.encode_cond_sequence(
            bbox, cams, rel_pos, y, mask, drop_cond_mask, drop_frame_mask
        )
        if y.shape[1] != T and y.shape[1] > 1:
            warn_once(f"y length {y.shape[1]} → interpolate to {T}")
            seq_len = y.shape[2]
            y = rearrange(y, "B T L D -> B (L D) T")
            y = F.interpolate(y, T)
            y = rearrange(y, "B (L D) T -> B T L D", L=seq_len)

        c = self.encode_map(maps, NC, h_pad_size, x_in_shape)
        c = rearrange(c, "B (T S) C -> B T S C", T=T)
   
        # patch embedding
        x_b = self.x_embedder(x)
        # if first_frame_latent is not None and self.first_frame_condition:
        #     first_frame_latent = rearrange(
        #         first_frame_latent, "B (C NC) T ... -> (B NC) C T ...", NC=NC
        #     )
        #     f_tokens = self.x_embedder(first_frame_latent)
        #     num_f0 = f_tokens.shape[1]
        #     inj_mask = repeat(drop_cond_mask, "b -> (b nc) 1 1", nc=NC)
        #     x_b[:, :num_f0] = inj_mask * f_tokens + (1.0 - inj_mask) * x_b[:, :num_f0]

        x_b = rearrange(x_b, "B (T S) C -> B T S C", T=T, S=S)
        x_b = x_b + pos_emb

        if self.x_control_embedder is None:
            x_c = x_b
        else:
            x_c = self.x_control_embedder(x)
            x_c = rearrange(x_c, "B (T S) C -> B T S C", T=T, S=S)
            x_c = x_c + pos_emb

        c = x_c + self.before_proj(c)
        x = x_b

        # sequence parallelism split
        if self.enable_sequence_parallelism:
            assert not self.sequence_parallelism_temporal
            x = split_forward_gather_backward(
                x, get_sequence_parallel_group(), dim=2, grad_scale="down"
            )
            c = split_forward_gather_backward(
                c, get_sequence_parallel_group(), dim=2, grad_scale="down"
            )
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        c = rearrange(c, "B T S C -> B (T S) C", T=T, S=S)

        # ── transformer blocks ───────────────────────────────────────────────
        if x_mask is not None:
            x_mask = repeat(x_mask, "b ... -> (b NC) ...", NC=NC)

        # control depth: base + control in parallel, skip injected
        for i in range(self.control_depth):
            x = auto_grad_checkpoint(
                self.base_blocks_s[i],
                x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC,
                mv_order_map, t_order_map,
            )
            c, c_skip = auto_grad_checkpoint(
                self.control_blocks_s[i],
                c, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC,
                mv_order_map, t_order_map,
            )
            x = x + c_skip
            if self.base_blocks_t is not None:
                x = auto_grad_checkpoint(
                    self.base_blocks_t[i],
                    x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC,
                    mv_order_map, t_order_map,
                )
            if self.control_blocks_t is not None:
                c, c_skip = auto_grad_checkpoint(
                    self.control_blocks_t[i],
                    c, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC,
                    mv_order_map, t_order_map,
                )
                x = x + c_skip

        # remaining base blocks only
        for i in range(self.control_depth, self.depth):
            x = auto_grad_checkpoint(
                self.base_blocks_s[i],
                x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC,
                mv_order_map, t_order_map,
            )
            if self.base_blocks_t is not None:
                x = auto_grad_checkpoint(
                    self.base_blocks_t[i],
                    x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC,
                    mv_order_map, t_order_map,
                )

        # gather sequence parallelism
        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(
                x, get_sequence_parallel_group(), dim=2, grad_scale="up"
            )
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # final layer
        x = self.final_layer(
            x,
            repeat(t, "b d -> (b NC) d", NC=NC),
            x_mask,
            repeat(t0, "b d -> (b NC) d", NC=NC) if t0 is not None else None,
            T, S,
        )
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        x = x.to(torch.float32)
        x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t, N_h=N_h, N_w=N_w,
            T_p=T_p, H_p=H_p, W_p=W_p,
            C_out=self.out_channels,
        )
        return x[:, :, :R_t, :R_h, :R_w]


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading from OpenSora v1.3
# ─────────────────────────────────────────────────────────────────────────────

def _load_state_dict_from_file(path: str) -> Dict[str, torch.Tensor]:
    """
    从 OpenSora v1.3 checkpoint 文件加载 state_dict。

    支持格式：
    1. 直接 state_dict（每个 key 是参数名，value 是 tensor）
    2. {"model": state_dict, ...}  —— ColossalAI 保存格式
    3. {"state_dict": state_dict}  —— Lightning 格式
    4. HuggingFace 目录（包含 model.safetensors 或 pytorch_model.bin）
    """
    import os, glob

    if os.path.isdir(path):
        # HuggingFace format: directory with model.safetensors or pytorch_model*.bin
        shard_files = sorted(glob.glob(os.path.join(path, "model/pytorch_model-*.pt")))
        if shard_files:
            # ColossalAI sharded checkpoint
            sd = {}
            for sf in shard_files:
                sd.update(torch.load(sf, map_location="cpu"))
            return sd
        safetensor_file = os.path.join(path, "model.safetensors")
        bin_file = os.path.join(path, "pytorch_model.bin")
        model_file = os.path.join(path, "model.pt")
        if os.path.exists(safetensor_file):
            from safetensors.torch import load_file
            return load_file(safetensor_file)
        elif os.path.exists(bin_file):
            return torch.load(bin_file, map_location="cpu")
        elif os.path.exists(model_file):
            return torch.load(model_file, map_location="cpu")
        else:
            # try to load from HF
            from magicdrivedit.utils.ckpt_utils import load_checkpoint as lc
            raise FileNotFoundError(
                f"Cannot find model weights in directory: {path}. "
                "Expected model.safetensors, pytorch_model.bin, or model.pt."
            )
    else:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        elif "state_dict" in ckpt:
            return ckpt["state_dict"]
        # check if it looks like a raw state_dict (first key is a param name)
        first_key = next(iter(ckpt.keys()))
        if "." in first_key:
            return ckpt
        # fallback
        return ckpt
    return ckpt


def load_from_opensora13_pretrained(
    model: MagicDriveSTDiT3v13,
    from_pretrained: str,
):
    """
    从 OpenSora v1.3 (STDiT3) 预训练权重初始化 MagicDriveSTDiT3v13。

    OpenSora v1.3 的 STDiT3-XL/2 结构：
      - x_embedder, t_embedder, t_block, y_embedder, fps_embedder  → 直接对应
      - final_layer                                                  → 直接对应
      - pos_embed                                                    → 直接对应
      - spatial_blocks[i]   →  base_blocks_s[i]
      - temporal_blocks[i]  →  base_blocks_t[i]

    MagicDrive 新增模块（random init 保持不变）：
      - base_blocks_s[i].{norm3, cross_view_attn, mva_proj, scale_shift_table_mva}
      - before_proj
      - control_blocks_s / control_blocks_t（从 spatial/temporal_blocks 复制权重）
      - camera_embedder, frame_embedder, bbox_embedder
      - controlnet_cond_embedder, controlnet_cond_embedder_temp, controlnet_cond_patchifier
      - x_control_embedder（从 x_embedder 复制权重）
    """
    logging.info(f"Loading OpenSora v1.3 pretrained weights from: {from_pretrained}")
    state_dict = _load_state_dict_from_file(from_pretrained)

    # ── Step 1: 加载所有能对上的权重（helper modules）────────────────────────
    # 原始 STDiT3 的 key 名字与 MagicDriveSTDiT3v13 大部分一致
    # （因为 base 模块名字相同）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    _expected_missing = []
    _unexpected = []
    for k in missing:
        # 这些 key 是 MagicDrive 新增的，缺失是正常的
        if any(
            k.startswith(prefix)
            for prefix in [
                "base_blocks_s.", "base_blocks_t.",
                "control_blocks_s.", "control_blocks_t.",
                "before_proj",
                "camera_embedder", "frame_embedder", "bbox_embedder",
                "controlnet_cond_", "x_control_embedder",
                "base_token",
            ]
        ):
            _expected_missing.append(k)
        else:
            logging.warning(f"  Unexpected missing key: {k}")

    for k in unexpected:
        if any(
            k.startswith(prefix)
            for prefix in ["spatial_blocks.", "temporal_blocks."]
        ):
            _unexpected.append(k)
        else:
            logging.warning(f"  Truly unexpected key: {k}")

    logging.info(
        f"Step 1 (helper modules): "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # ── Step 2: spatial_blocks → base_blocks_s ──────────────────────────────
    spatial_sd = {
        k[len("spatial_blocks."):]: v
        for k, v in state_dict.items()
        if k.startswith("spatial_blocks.")
    }
    if spatial_sd:
        m, u = model.base_blocks_s.load_state_dict(spatial_sd, strict=False)
        logging.info(
            f"Step 2 (spatial_blocks → base_blocks_s): "
            f"missing={len(m)}, unexpected={len(u)}"
        )
        # missing keys in base_blocks_s are cross_view_attn params (new), OK
    else:
        logging.warning(
            "No 'spatial_blocks.' keys found in checkpoint. "
            "Trying 'blocks.' prefix (PixArt-style)..."
        )
        blocks_sd = {
            k[len("blocks."):]: v
            for k, v in state_dict.items()
            if k.startswith("blocks.")
        }
        if blocks_sd:
            m, u = model.base_blocks_s.load_state_dict(blocks_sd, strict=False)
            logging.info(
                f"  Loaded from 'blocks.' → base_blocks_s: "
                f"missing={len(m)}, unexpected={len(u)}"
            )

    # ── Step 3: temporal_blocks → base_blocks_t ─────────────────────────────
    temporal_sd = {
        k[len("temporal_blocks."):]: v
        for k, v in state_dict.items()
        if k.startswith("temporal_blocks.")
    }
    if temporal_sd and model.base_blocks_t is not None:
        m, u = model.base_blocks_t.load_state_dict(temporal_sd, strict=False)
        logging.info(
            f"Step 3 (temporal_blocks → base_blocks_t): "
            f"missing={len(m)}, unexpected={len(u)}"
        )

    # ── Step 4: control_blocks 从 spatial/temporal_blocks 复制权重 ───────────
    # 控制分支与骨干分支结构相同，用 spatial 权重初始化（strict=False，
    # 多出的 after_proj 保持 zero-init）
    if spatial_sd:
        # 只取 control_depth 层
        ctrl_spatial_sd = {}
        for k, v in spatial_sd.items():
            # key 格式: "{layer_idx}.{param_name}"
            parts = k.split(".")
            if parts[0].isdigit():
                layer_idx = int(parts[0])
                if layer_idx < model.control_depth:
                    ctrl_spatial_sd[k] = v
        m, u = model.control_blocks_s.load_state_dict(ctrl_spatial_sd, strict=False)
        logging.info(
            f"Step 4 (spatial_blocks[:ctrl] → control_blocks_s): "
            f"missing={len(m)}, unexpected={len(u)}"
        )

    if temporal_sd and model.control_blocks_t is not None:
        ctrl_temporal_sd = {}
        for k, v in temporal_sd.items():
            parts = k.split(".")
            if parts[0].isdigit() and int(parts[0]) < model.control_depth:
                ctrl_temporal_sd[k] = v
        m, u = model.control_blocks_t.load_state_dict(ctrl_temporal_sd, strict=False)
        logging.info(
            f"Step 4 (temporal_blocks[:ctrl] → control_blocks_t): "
            f"missing={len(m)}, unexpected={len(u)}"
        )

    # ── Step 5: x_control_embedder 从 x_embedder 复制权重 ────────────────────
    if model.x_control_embedder is not None:
        x_emb_sd = {
            k[len("x_embedder."):]: v
            for k, v in state_dict.items()
            if k.startswith("x_embedder.")
        }
        if x_emb_sd:
            model.x_control_embedder.load_state_dict(x_emb_sd, strict=True)
            logging.info("Step 5: x_embedder → x_control_embedder (strict)")

    logging.info("OpenSora v1.3 weights loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Registry entry point
# ─────────────────────────────────────────────────────────────────────────────

@MODELS.register_module("MagicSoraSTDiT3v13-XL/2")
def MagicSoraSTDiT3v13_XL_2(
    from_pretrained=None,          # 已有 MagicDrive checkpoint（严格加载）
    from_opensora13_pretrained=None,  # OpenSora v1.3 原版权重（宽松加载）
    force_huggingface=False,
    trainable_mode="new_only",
    **kwargs,
):
    """
    构建 MagicDriveSTDiT3v13 (STDiT3-XL/2 backbone)。

    优先级：
    1. from_pretrained        → 直接严格加载已训练好的 MagicDrive 权重
    2. from_opensora13_pretrained → 从 OpenSora v1.3 骨干初始化
    3. 两者都不提供           → 随机初始化
    """
    kwargs.setdefault("trainable_mode", trainable_mode)
    config = MagicDriveSTDiT3v13Config(
        depth=28,
        hidden_size=1152,
        patch_size=(1, 2, 2),
        num_heads=16,
        **kwargs,
    )
  
    model = MagicDriveSTDiT3v13(config)

    if from_pretrained is not None:
        if os.path.exists(from_pretrained) and not force_huggingface:
            load_checkpoint(model, from_pretrained, strict=True)
            logging.info(f"Loaded MagicDrive checkpoint (strict): {from_pretrained}")
        else:
            # HuggingFace format
            model = MagicDriveSTDiT3v13.from_pretrained(from_pretrained, **kwargs)
            logging.info(f"Loaded from HuggingFace: {from_pretrained}")

    elif from_opensora13_pretrained is not None:
        model = load_from_opensora13_pretrained(model, from_opensora13_pretrained)

    else:
        logging.info("No pretrained weights provided; using random initialization.")

    return model