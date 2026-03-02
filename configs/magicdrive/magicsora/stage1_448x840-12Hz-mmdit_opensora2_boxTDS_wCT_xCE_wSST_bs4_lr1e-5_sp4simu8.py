# ─────────────────────────────────────────────────────────────────────────────
# stage3_448x840_mmdit_flux_6cam_12Hz.py
#
# MagicDrive-MMDiT 训练配置
# 骨干: OpenSora2 MMDiT (Flux-style DoubleStream + SingleStream + RoPE)
# 控制: MagicDrive ControlNet 分支 (map / bbox / cam / frame)
# 场景: nuScenes 6相机多视图视频生成, 424x800 分辨率, 12Hz
# ─────────────────────────────────────────────────────────────────────────────

# ── 数据集基础设置（与原版一致） ──────────────────────────────────────────────
num_frames = None          # None = 可变长度模式
micro_frame_size = 8       # VAE 时序分块大小
bbox_mode = 'all-xyz'

data_cfg_names = [
    # ((224, 400), "Nuscenes_map_cache_box_t_with_n2t_12Hz"),
    ((448, 840), "Nuscenes_400_map_cache_box_t_with_n2t_12Hz"),
]

video_lengths_fps = {
    # "224x400": [
    #     [1, 9, 17, 33, 65,],
    #     [[120,], [12,], [12], [12], [12]],
    # ],
    "448x840": [
        [1, 9, 17, 33, 65, 129],
        [[120,],[12,], [12,], [12,], [12,], [12,]],
        [1, 1, 1, 1, 1, 1],    # repeat times
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
    # "224-400-12-17": 2,
    "448-840-120-1":  -1,
    "448-840-12-9":   1, # 17
    "448-840-12-17": -1,
    "448-840-12-33":   -1,
    "448-840-12-65":   -1,
    "448-840-12-129": -1,   # 跳过（显存不足时设为 -1）
}

validation_index = [
    # "5538-448-840-12-33",
    # "14631-448-840-12-33",
    # "6720-448-840-12-33",
    # "14449-448-840-12-33",
    # "3649-448-840-12-33",
    # "912-448-840-12-33",
    "1680-448-840-12-9",
    "5543-448-840-12-9",
    # "3657-448-840-12-33",
]
validation_before_run = False

# ── 运行时设置 ────────────────────────────────────────────────────────────────
dtype = "bf16"
sp_size = 4
plugin = "zero2-seq" if sp_size > 1 else "zero2"
grad_checkpoint = True
batch_size = None          # 由 bucket_config 决定
drop_cond_ratio = 0.15
drop_cond_ratio_t = 0.4
drop_first_frame_ratio = 0.1   # 首帧引导随机屏蔽比例

# 加速
num_workers = 2
prefetch_factor = 2
num_bucket_build_workers = 16

# ── 多视图设置 ────────────────────────────────────────────────────────────────
# nuScenes 6相机环形排列邻居关系
# 0=front, 1=front-right, 2=front-left, 3=back, 4=back-right, 5=back-left
# (按原版 MagicDrive 顺序)
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
t_order_map = None  # MMDiT 不需要，保留字段兼容 runner

# ── VAE ──────────────────────────────────────────────────────────────────────
vae_out_channels = 64 #16   #16 for CogVideoX VAE  64 for hunyuan vae 输出通道数

vae = dict(
    type="hunyuan_vae",
    from_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/Open-Sora-v2/hunyuan_vae.safetensors",
    # subfolder="vae",
    # micro_frame_size=micro_frame_size,
    # micro_batch_size=1,
)
dropout_ratio = {  # probability for dropout text embedding
    "t5": 0.31622777,
    "clip": 0.31622777,
}
# ── T5 文本编码器（序列上下文，输入 txt 流）────────────────────────────────────
text_encoder = dict(
    type="t5",
    # from_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/Open-Sora-v2/google/t5-v1_1-xxl",
    from_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/google/t5-v1_1-xxl",
    model_max_length=512,
    shardformer=True,
)

# ── CLIP 文本编码器（池化向量，输入 vec 流）─────────────────────────────────────
# MMDiT (Flux) 需要两路文本：T5 序列 + CLIP 池化向量
# 如果没有 CLIP 模型，设为 None，训练器会用全零替代（vec 流退化）
clip_text_encoder = dict(
    from_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/Open-Sora-v2/openai/clip-vit-large-patch14",
    max_length=77,
)

# ── Scheduler (Rectified Flow) ────────────────────────────────────────────────
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,       # CogVideoX 风格时间步变换
    sample_method="logit-normal",
)

# ── 主模型：MagicDrive-MMDiT ──────────────────────────────────────────────────
model = dict(
    type="MagicSora",

    # ── OpenSora2/Flux 骨干参数 ──────────────────────────────────────────────
    # 这些参数对应 Flux-dev/schnell large 规格
    in_channels=vae_out_channels,               # 16 
    vec_in_dim=768,                             # CLIP-L 维度
    context_in_dim=4096,                        # T5-XXL 维度
    hidden_size=3072,                           # Flux large hidden dim
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,                                   # DoubleStreamBlock 层数
    depth_single_blocks=38,                     # SingleStreamBlock 层数
    axes_dim=[16, 56, 56],                      # RoPE 各轴维度，和=pe_dim=3072/24=128
    theta=10000,
    qkv_bias=True,
    guidance_embed=True,                        # CFG distillation (guidance vec)
    fused_qkv=True,
    patch_size=2,

    # ── 多视图 & 时间注意力 ──────────────────────────────────────────────────
    num_cameras=6,
    temporal_every_n_blocks=1,                  # 每 N 个 DoubleStream block 加一个 TemporalSelfAttention
                                                # 1=每层(最强), 2=隔层, 0=不加(退化到v1)

    # ── ControlNet 分支 ──────────────────────────────────────────────────────
    control_depth=9,                            # 前9层 DoubleStream 做控制分支
    control_single_depth=0,                     # SingleStream 控制分支层数（0=不加）
    control_skip_cross_view=True,               # 控制分支不做跨视图注意力（节省显存）

    # ── 训练参数控制 ─────────────────────────────────────────────────────────
    # "new_only": 只训练新增模块 (cross_view/temporal/control/embedders) ← 推荐第一阶段
    # "control":  只训练控制分支 + 条件编码器
    # "all":      全量微调
    trainable_mode="new_only",

    # ── 预训练权重 ────────────────────────────────────────────────────────────
    # 方式1: 从 OpenSora2 预训练权重初始化骨干（首次训练用这个）
    from_opensora2_pretrained="/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/Open-Sora-v2/opensora2_flux_pretrained.pt",
    # 方式2: 从之前的 MagicDriveMMDiT checkpoint 继续训练（断点续训用 load=）
    # from_pretrained=None,

    # ── 条件编码器：相机参数 ─────────────────────────────────────────────────
    cam_encoder_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedder",
    cam_encoder_param=dict(
        input_dim=3,
        num=7,
        after_proj=True,
    ),

    # ── 条件编码器：帧间位姿 ─────────────────────────────────────────────────
    frame_emb_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedderTemp",
    frame_emb_param=dict(
        input_dim=3,
        num=4,
        after_proj=True,
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False,
        enable_xformers=True,
        enable_layernorm_kernel=True,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),

    # ── 条件编码器：Bounding Box ─────────────────────────────────────────────
    bbox_embedder_cls="magicdrivedit.models.magicdrive.embedder.ContinuousBBoxWithTextTempEmbedding",
    bbox_embedder_param=dict(
        n_classes=10,
        class_token_dim=3072,               # 与 hidden_size 匹配（原版1152，这里3072）
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[3072, 512, 512, 3072],   # 与 hidden_size 匹配
        mode=bbox_mode,
        minmax_normalize=False,
        use_text_encoder_init=True,
        after_proj=True,
        sample_id=True,
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False,
        enable_xformers=True,
        enable_layernorm_kernel=True,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),

    # ── 条件编码器：BEV Map ──────────────────────────────────────────────────
    map_embedder_cls="magicdrivedit.models.magicdrive.embedder.MapControlEmbedding",
    map_embedder_param=dict(
        conditioning_size=[8, 400, 400],
        block_out_channels=[16, 32, 96, 256],
        # conditioning_embedding_channels 由 hidden_size//2=1536 自动设置
    ),
    map_embedder_downsample_rate=4.5,
    micro_frame_size=None,                  # 地图编码时不分块（或设为 micro_frame_size）

    # ── 杂项 ─────────────────────────────────────────────────────────────────
    drop_path=0.0,
    grad_ckpt=True,
)

# ── 部分加载（用已有 MagicDriveDiT 权重初始化相同部分）────────────────────────
# 如果有之前训练好的 MagicDriveSTDiT3 权重，可以用来初始化条件编码器部分
# partial_load = '/path/to/MagicDriveDiT-stage3-40k-ft'
partial_load = None

# ── 验证设置 ──────────────────────────────────────────────────────────────────
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

# ── 帧遮盖 mask（用于可变长度视频训练）────────────────────────────────────────
mask_ratios = {
    "random": 0.01,
    "intepolate": 0.005,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.001,
    "image_head": 0.22,         # 首帧 mask（用于条件生成训练）
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

# ── 日志 & 优化 ───────────────────────────────────────────────────────────────
seed = 1024
outputs = "outputs"
wandb = False
epochs = 4
log_every = 1
ckpt_every = 250 * 5
report_every = ckpt_every

load = None
grad_clip = 1.0
lr = 1e-4           # 建议：new_only 模式用 1e-4，all 模式用 2e-5
ema_decay = 0.9999
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 500  # MMDiT 骨干较大，建议更长 warmup