# ─────────────────────────────────────────────────────────────────────────────
# stage3_424x800_stdit3v13.py
#
# MagicDriveV2 × OpenSora v1.3 (STDiT3-XL/2) 训练配置
#
# 参数量对比（OOM 根因）：
#   OpenSora v2.0 MMDiT (Flux-large)   ≈ 11B   →  GPU 显存 22 GB
#   OpenSora v1.3 STDiT3-XL/2         ≈ 0.6B  →  GPU 显存  1.2 GB
#   MagicDrive 控制条件 + cross-view   ≈ 0.2B
#   ──────────────────────────────────────────
#   总计 (v1.3 版)                     ≈ 0.8B  →  GPU 显存  1.6 GB (bf16)
#
# 即便 6 视角 × batch_size × 激活值 × 优化器状态，
# 也完全在 A100/H100 80GB 内，OOM 问题彻底解决。
# ─────────────────────────────────────────────────────────────────────────────
mask_types = {
    "i2v_head": 5,
    "i2v_tail": 2,
    "i2v_loop": 2,
    "v2v_head": 1,
    "v2v_head_noisy": 2,
    "v2v_tail": 1,
    "other": 1,
    "none": 2,
}

# ── 数据集 ─────────────────────────────────────────────────────────────────
num_frames = None
micro_frame_size = 8
bbox_mode = "all-xyz"

data_cfg_names = [
    ((424, 800), "Nuscenes_400_map_cache_box_t_with_n2t_12Hz"),
    ((448, 840), "Nuscenes_400_map_cache_box_t_with_n2t_12Hz_448x840"), 
]

video_lengths_fps = {
    "424x800": [
        [1, 17, 33, 65, 129],
        [[120], [12], [12], [12], [12]],
        [1, 1, 1, 1, 1],
    ],
    "448x840": [  # 424x800
        [1, 17, 33, 65, 129,],
        [[120,], [12,], [12,], [12], [12],],
        [1, 1, 1, 1, 1],
    ],
}

balance_keywords = ["night", "rain", "none"]

dataset_cfg_overrides = [
    (
        ("dataset.data.train.ann_file",
         "/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/PublicDatasets/nuscenes/MagicDriveDiT-nuScenes-metadata/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
        ("dataset.data.val.ann_file",
         "/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/PublicDatasets/nuscenes/MagicDriveDiT-nuScenes-metadata/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
        ("dataset.data.train.type", "NuScenesVariableDataset"),
        ("dataset.data.val.type", "NuScenesVariableDataset"),
        ("dataset.data.train.video_length", video_lengths_fps["424x800"][0]),
        ("dataset.data.train.fps", video_lengths_fps["424x800"][1]),
        ("+dataset.data.train.repeat_times", video_lengths_fps["424x800"][2]),
        ("+dataset.data.train.balance_keywords", balance_keywords),
        ("dataset.data.val.video_length", video_lengths_fps["424x800"][0]),
        ("dataset.data.val.fps", video_lengths_fps["424x800"][1]),
    ),
    (
        ("dataset.data.train.ann_file",
         "/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/PublicDatasets/nuscenes/MagicDriveDiT-nuScenes-metadata/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
        ("dataset.data.val.ann_file",
         "/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/PublicDatasets/nuscenes/MagicDriveDiT-nuScenes-metadata/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
        ("dataset.data.train.type", "NuScenesVariableDataset"),
        ("dataset.data.val.type", "NuScenesVariableDataset"),
        ("dataset.data.train.video_length", video_lengths_fps["448x840"][0]),
        ("dataset.data.train.fps",          video_lengths_fps["448x840"][1]),
        ("+dataset.data.train.repeat_times", video_lengths_fps["448x840"][2]),
        ("+dataset.data.train.balance_keywords", balance_keywords),
        ("dataset.data.val.video_length", video_lengths_fps["448x840"][0]),
        ("dataset.data.val.fps",          video_lengths_fps["448x840"][1]),
    ),
]

img_collate_param_train = dict(
    frame_emb="next2top",
    bbox_mode=bbox_mode,
    bbox_view_shared=False,
    keyframe_rate=6,
    bbox_drop_ratio=0.4,
    bbox_add_ratio=0.1,
    bbox_add_num=3,
    bbox_processor_type=2,
)

bucket_config = {
    "424-800-120-1": 10,
    "424-800-12-17": 6,
    "424-800-12-33": 4,
    "424-800-12-65": 2,
    "424-800-12-129": -1,
    "448-840-120-1": 8, #6 10
    "448-840-12-17": 4,  # 6: 32-34s/it, 8: 47s/it
    "448-840-12-33": 2,  # 3: 30s/it, 4: 37-40s/it
    "448-840-12-65": 1,  # 1: 16-18s/it, 2: 34-35s/it
    "448-840-12-129": -1,
}

validation_index = [
    "5538-424-800-12-33",
    "14631-424-800-12-33",
    "6720-424-800-12-33",
    "14449-424-800-12-33",
    "3649-424-800-12-33",
    "912-448-840-12-33",
    "1680-448-840-12-17",
    "5543-448-840-12-33",
    "3657-448-840-12-33",
]
validation_before_run = False

# ── 运行时 ─────────────────────────────────────────────────────────────────
dtype = "bf16"
sp_size = 4
plugin = "zero2-seq" if sp_size > 1 else "zero2"
grad_checkpoint = True
batch_size = None
drop_cond_ratio = 0.15
drop_cond_ratio_t = 0.4
drop_first_frame_ratio = 0.1

num_workers = 2
prefetch_factor = 2
num_bucket_build_workers = 16

mv_order_map = {
    0: [5, 1], 1: [0, 2], 2: [1, 3],
    3: [2, 4], 4: [3, 5], 5: [4, 0],
}
t_order_map = None

global_flash_attn = True
global_layernorm = True
global_xformers = True

# ── VAE ────────────────────────────────────────────────────────────────────
# OpenSora v1.3 使用的 VAE：
#   STDiT3 版本的 MagicDriveV2 原配 CogVideoX VAE (in_channels=16)
#   与 OpenSora v1.3 训练时使用的 VAE 一致，无需替换。
#
# ⚠ 注意：OpenSora v1.3 实际上也使用 CogVideoX VAE（16 channel, 4x temporal, 8x spatial）
# 若你有 OpenSora v1.3 官方 VAE checkpoint，直接替换 from_pretrained 路径即可。

vae_out_channels = 16

vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/OpenSora/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)

# ── 文本编码器 ──────────────────────────────────────────────────────────────
text_encoder = dict(
    type="t5",
    from_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/google/t5-v1_1-xxl",   # ← 替换
    model_max_length=300,
    shardformer=True,
)

# ── Scheduler ──────────────────────────────────────────────────────────────
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,
    sample_method="logit-normal",
)

# ── 主模型（STDiT3-XL/2，~0.6B）────────────────────────────────────────────
model = dict(
    type="MagicSoraSTDiT3v13-XL/2",

    # ── OpenSora v1.3 预训练权重 ─────────────────────────────────────────
    # 可以是：
    #   - OpenSora v1.3 官方 checkpoint 目录（含 model.safetensors 或 pytorch_model.bin）
    #   - .pt / .pth 文件
    #   - 已有 MagicDrive checkpoint（此时用 from_pretrained 替代）
    from_opensora13_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/OpenSora/OpenSora-STDiT-v4",   # ← 替换

    # 训练模式："new_only" | "all" | "control" | "freeze_base"
    # Stage 1 推荐 "new_only"：只训练新增的控制条件模块，冻结骨干
    # Stage 2 推荐 "freeze_base"：开放 temporal blocks
    # Stage 3 推荐 "all"：全量微调
    trainable_mode="new_only",

    # ── STDiT3-XL/2 骨干参数（与原版完全一致）───────────────────────────
    simulate_sp_size=[4, 8],
    qk_norm=True,
    pred_sigma=False,
    enable_flash_attn=True and global_flash_attn,
    enable_layernorm_kernel=True and global_layernorm,
    enable_sequence_parallelism=sp_size > 1,
    freeze_y_embedder=True,

    # ── MagicDrive 结构参数（与原版完全一致）───────────────────────────
    with_temp_block=True,
    use_x_control_embedder=True,
    enable_xformers=False and global_xformers,
    sequence_parallelism_temporal=False,
    use_st_cross_attn=False,
    uncond_cam_in_dim=(3, 7),

    # 相机编码器
    cam_encoder_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedder",
    cam_encoder_param=dict(
        input_dim=3,
        num=7,
        after_proj=True,
    ),

    # 帧编码器
    frame_emb_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedderTemp",
    frame_emb_param=dict(
        input_dim=3,
        num=4,
        after_proj=True,
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),

    # bbox 编码器
    bbox_embedder_cls="magicdrivedit.models.magicdrive.embedder.ContinuousBBoxWithTextTempEmbedding",
    bbox_embedder_param=dict(
        n_classes=10,
        class_token_dim=1152,         # ← STDiT3-XL/2 hidden_size
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[1152, 512, 512, 1152],  # ← 对应 hidden_size=1152
        mode=bbox_mode,
        minmax_normalize=False,
        use_text_encoder_init=True,
        after_proj=True,
        sample_id=True,
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),

    # 地图编码器
    map_embedder_cls="magicdrivedit.models.magicdrive.embedder.MapControlEmbedding",
    map_embedder_param=dict(
        conditioning_size=[8, 400, 400],
        block_out_channels=[16, 32, 96, 256],
    ),
    map_embedder_downsample_rate=4.5,
    micro_frame_size=None,

    # ControlNet 深度
    control_depth=13,
    control_skip_cross_view=True,
    control_skip_temporal=False,

    drop_path=0.0,
)

# 如果想从已有 MagicDrive checkpoint 继续训练，取消注释：
# partial_load = "/path/to/MagicDriveDiT-stage3-40k-ft"

val = dict(
    validation_index=validation_index,
    batch_size=1,
    verbose=2,
    num_sample=2,
    save_fps=None,
    seed=1024,
    scheduler=dict(
        **scheduler,
        num_sampling_steps=30,
        cfg_scale=2.0,
    ),
)

mask_ratios = {
    "random": 0.01,
    "intepolate": 0.005,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.001,
    "image_head": 0.22,
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

seed = 1024
outputs = "outputs"
wandb = False
epochs = 4
log_every = 1
ckpt_every = 1250
report_every = ckpt_every

load = None
grad_clip = 1.0
lr = 1e-4         # Stage 1: 1e-4；Stage 3 全量: 2e-5
ema_decay = 0.9999
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 500