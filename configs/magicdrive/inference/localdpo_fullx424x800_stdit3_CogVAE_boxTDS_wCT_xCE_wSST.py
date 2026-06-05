# Inference config for RiskMV-DPO (LocalDPO fine-tuned MagicDrive)
# Override checkpoint path via: --cfg-options model.from_pretrained=/path/to/ckpt

fps = 12
frame_interval = 1
save_fps = 12
validation_index = [0]
num_sample = 1
batch_size = 1
dtype = "bf16"

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,
    num_sampling_steps=30,
    cfg_scale=2.0,
)

# Dataset settings
num_frames = 65
data_cfg_name = "Nuscenes_400_map_cache_box_t_with_n2t_12Hz"
bbox_mode = 'all-xyz'
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
dataset_cfg_overrides = (
    ("dataset.data.val.type", "NuScenesTvalDataset"),
    ("dataset.data.train.ann_file", "data/nuscenes/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
    ("dataset.data.val.ann_file", "data/nuscenes/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
    ("+dataset.data.val.start_on_firstframe", True),
    ("+dataset.data.val.micro_frame_size", 8),
    ("+dataset.data.val.info_file_from_planner", "data/nuscenes/info_10.json"),
)

sp_size = 1
plugin = "zero2-seq" if sp_size > 1 else "zero2"
grad_checkpoint = False
drop_cond_ratio = 0.15
num_workers = 0
num_bucket_build_workers = 16

mv_order_map = {
    0: [5, 1], 1: [0, 2], 2: [1, 3],
    3: [2, 4], 4: [3, 5], 5: [4, 0],
}
t_order_map = None
global_flash_attn = True
global_layernorm = True
global_xformers = True
micro_frame_size = None
vae_out_channels = 16

model = dict(
    type="MagicDriveSTDiT3-XL/2-localdpo",
    simulate_sp_size=[4, 8],
    qk_norm=True,
    pred_sigma=False,
    enable_flash_attn=True and global_flash_attn,
    enable_layernorm_kernel=True and global_layernorm,
    enable_sequence_parallelism=sp_size > 1,
    freeze_y_embedder=True,
    with_temp_block=True,
    use_x_control_embedder=True,
    enable_xformers=False and global_xformers,
    sequence_parallelism_temporal=False,
    use_st_cross_attn=False,
    uncond_cam_in_dim=(3, 7),
    cam_encoder_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedder",
    cam_encoder_param=dict(input_dim=3, num=7, after_proj=True),
    bbox_embedder_cls="magicdrivedit.models.magicdrive.embedder.ContinuousBBoxWithTextTempEmbedding",
    bbox_embedder_param=dict(
        n_classes=10, class_token_dim=1152, trainable_class_token=False,
        embedder_num_freq=4, proj_dims=[1152, 512, 512, 1152], mode=bbox_mode,
        minmax_normalize=False, use_text_encoder_init=True, after_proj=True,
        sample_id=True, num_heads=8, mlp_ratio=4.0, qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True, time_downsample_factor=4.5,
    ),
    map_embedder_cls="magicdrivedit.models.magicdrive.embedder.MapControlEmbedding",
    map_embedder_param=dict(
        conditioning_size=[8, 400, 400],
        block_out_channels=[16, 32, 96, 256],
    ),
    map_embedder_downsample_rate=4.5,
    micro_frame_size=micro_frame_size,
    frame_emb_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedderTemp",
    frame_emb_param=dict(
        input_dim=3, num=4, after_proj=True, num_heads=8, mlp_ratio=4.0,
        qk_norm=True, enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True, time_downsample_factor=4.5,
    ),
    control_skip_cross_view=True,
    control_skip_temporal=False,
    from_pretrained="???",
    use_vggt_adapter=True,
    vggt_checkpoint="pretrained/dggt/model_latest_waymo.pt",
    freeze_vggt=True,
    vggt_feat_dim=3072,
    num_geo_tokens=16,
    geo_adapter_layers=2,
    geo_adapter_heads=16,
)

vae = dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="pretrained/CogVideoX-2b",
    subfolder="vae",
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
)
text_encoder = dict(
    type="t5",
    from_pretrained="pretrained/google/t5-v1_1-xxl",
    model_max_length=300,
)

mask_ratios = {
    "random": 0.01, "intepolate": 0.002, "quarter_random": 0.002,
    "quarter_head": 0.002, "quarter_tail": 0.002, "quarter_head_tail": 0.002,
    "image_random": 0.0, "image_head": 0.22, "image_tail": 0.005,
    "image_head_tail": 0.005,
}

seed = 42
outputs = "outputs/inference/localdpo"
wandb = False
epochs = 1
log_every = 1
ckpt_every = 500
load = None
grad_clip = 1.0
lr = 2e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 500
force_image = True
split_6views_for_image = True
