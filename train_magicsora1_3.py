"""
train_magicdrive_stdit3v13.py
═════════════════════════════════════════════════════════════════════════════
MagicDriveV2 × OpenSora v1.3 (STDiT3-XL/2) 训练脚本

基于原版 train_magicdrive.py，改动点：

1. 模型构建：使用 MagicDriveSTDiT3v13-XL/2（注册于 magicdrive_stdit3_v13.py）
   - 去掉 MMDiT 相关的 CLIP encoder
   - 去掉 y_vec（STDiT3 不需要 CLIP 池化向量）
   - 完整保留 t5 text encoder，maps/cam/bbox 条件不变

2. EMA 修复：使用 build_ema_on_cpu_v2 替换 deepcopy OOM 方案
   - EMA 常驻 CPU fp32，GPU 节省 ~1.6 GB（原版 ~1.2 GB × 2 倍）

3. model_kwargs：去掉 MMDiT 专用的 guidance / y_vec，恢复原版接口
   - fps / height / width 保留（STDiT3 位置编码需要）
   - t_order_map 保留（STDiT3 时序块需要）

4. 其余逻辑（数据加载、SP、ZeRO、验证、checkpoint）与原版完全一致
═════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import random
import logging
import warnings
from contextlib import nullcontext
from datetime import timedelta
from pprint import pformat

sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
if not torch.cuda.is_available() or DEVICE_TYPE == "npu":
    USE_NPU = True
    os.environ["DEVICE_TYPE"] = "npu"
    DEVICE_TYPE = "npu"
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False

import magicdrivedit.utils.module_contrib  # noqa: side-effects
import torch.distributed as dist
from einops import rearrange, repeat
import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm
from mmcv.parallel import DataContainer

from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger("shapely.geos").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.INFO)

from magicdrivedit.acceleration.checkpoint import set_grad_checkpoint
from magicdrivedit.acceleration.parallel_states import (
    get_data_parallel_group,
    get_sequence_parallel_group,
)
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.ckpt_utils import (
    load,
    model_gathering,
    model_sharding,
    record_model_param_shape,
    save,
    prepare_ckpt,
    RandomStateManager,
)
from magicdrivedit.utils.config_utils import (
    define_experiment_workspace,
    parse_configs,
    save_training_config,
    merge_dataset_cfg,
    mmengine_conf_get,
    mmengine_conf_set,
)
from magicdrivedit.utils.lr_scheduler import LinearWarmupLR, MultiStepWithLinearWarmupLR
from magicdrivedit.utils.misc import (
    Timer,
    all_reduce_mean,
    reset_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent
)
from magicdrivedit.utils.train_utils import (
    MaskGenerator,
    create_colossalai_plugin,
    update_ema,
    run_validation,
    sp_vae,
    get_mask_cond,
    get_mask_index,
)

# ── EMA 修复（CPU fp32，解决 deepcopy OOM）────────────────────────────────
from ema_utils import (
    build_ema_on_cpu_v2,
    update_ema_cpu,
    model_sharding_ema,
    model_gathering_ema,
)

from mmcv.parallel import DataContainer

# ── 注册 MagicDriveSTDiT3v13 ─────────────────────────────────────────────
# from magicdrive_stdit3_v13 import (   # noqa: ensure registration
#     MagicDriveSTDiT3v13,
#     MagicDriveSTDiT3v13Config,
# )

from opensora.registry import MODELS as MODELS2
from opensora.registry import build_module as build_module2
# ─────────────────────────────────────────────────────────────────────────────
# model_kwargs 构建（STDiT3 接口，与原版 train_magicdrive.py 一致）
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_validation_stdit3(
    cfg, text_encoder, vae, model,
    device, dtype, val_dataloader, coordinator,
    global_step, exp_dir,
):
    """验证函数，接口与原版 run_validation 一致。"""
    model.eval()
    with torch.no_grad():
        run_validation(
            cfg, text_encoder, vae, model,
            device, dtype, val_dataloader, coordinator,
            global_step, exp_dir,
        )
    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.set_grad_enabled(True)
    cfg = parse_configs(training=True)
    cfg_dtype = cfg.get("dtype", "bf16")
    dtype = to_torch_dtype(cfg_dtype)
    verbose_mode = cfg.get("verbose", False)
    enable_debug = cfg.get("debug", False)
    # exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    # logger = reset_logger(exp_dir, enable_debug)

    # 数据集配置（支持可变分辨率，与原版一致）
    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets, val_datasets = [], []
        for idx, (res, data_cfg_name) in enumerate(cfg.data_cfg_names):
            overrides = cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)[idx]
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        cfg.dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        cfg.val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:
        cfg.dataset, cfg.val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames
        )
    
    # ── 分布式初始化 ──────────────────────────────────────────────────────────
    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)

    set_seed(cfg.get("seed", 1024))
    torch.cuda.manual_seed_all(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    coordinator._local_rank = (
        int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
    )
    device = get_current_device()

    # ── 实验目录 ──────────────────────────────────────────────────────────────
    if cfg.get("overfit", None) is not None:
        cfg.tag = (f"{cfg.tag}_" if cfg.get("tag", "") != "" else "") + "overfit-" + str(cfg.overfit)
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    coordinator.block_all()
    if coordinator.is_node_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    log = reset_logger(exp_dir, enable_debug)
    log.info("Experiment directory: %s", exp_dir)
    log.info("Configuration:\n%s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)

    # ── ColossalAI booster ────────────────────────────────────────────────────
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
        overlap_allgather=cfg.get("overlap_allgather", False),
        verbose=verbose_mode,
    )
    if not hasattr(plugin, "pg_mesh"):
        plugin.pg_mesh = None
        plugin.destroy_mesh_process_groups = lambda: None
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # =========================================================================
    # 数据集
    # =========================================================================
    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    # logger.info("Building dataset...")
    # == build dataset ==
    
    dataset = build_module(cfg.dataset, DATASETS)
    if cfg.get("overfit", None) is not None:
        _overfit_idxs = random.sample(range(len(dataset)), cfg.overfit)
        # logger.info(f"Overfit on: {_overfit_idxs}")
        overfit_idxs = []
        for _ in range(cfg.epochs):
            overfit_idxs += _overfit_idxs
            random.shuffle(_overfit_idxs)
        cfg.epochs = 1
        dataset = torch.utils.data.Subset(dataset, overfit_idxs)
    # logger.info("Dataset contains %s samples.", len(dataset))
    
    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size= cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 1), ###### 4
        seed=cfg.get("seed", 1024),
        shuffle=True if cfg.get("overfit", None) is None else False,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # val
    if cfg.get("overfit", None) is not None:
        # first n samples, actually this is all unique samples.
        val_dataset = torch.utils.data.Subset(dataset, list(range(cfg.overfit)))
    else:
        val_dataset = build_module(cfg.val_dataset, DATASETS)
        if cfg.val.validation_index != "all":
            if len(cfg.val.validation_index) < get_data_parallel_group().size():
                if isinstance(cfg.val.validation_index[0], int):
                    # we use max world size 32 before, keep the same.
                    cfg.val.validation_index += random.sample(
                        list(set(range(len(val_dataset))) - set(cfg.val.validation_index)),
                        min(get_data_parallel_group().size(), 32) - len(cfg.val.validation_index),
                    )
                    # for larger than 32, add them one-by-one.
                    if get_data_parallel_group().size() > 32:
                        while len(cfg.val.validation_index) < get_data_parallel_group().size():
                            cfg.val.validation_index += random.sample(
                                list(set(range(len(val_dataset)))
                                     - set(cfg.val.validation_index)), 1,
                            )
                else:
                    while len(cfg.val.validation_index) < get_data_parallel_group().size():
                        new_key = val_dataset.rand_another_key()
                        if new_key not in cfg.val.validation_index:
                            cfg.val.validation_index.append(new_key)
                logging.info(f"validation_index rewrite as: {cfg.val.validation_index}")
            val_dataset = torch.utils.data.Subset(
                val_dataset, cfg.val.validation_index)
        else:
            raise NotImplementedError()
    # logger.info("Val Dataset contains %s samples.", len(val_dataset))
    dataloader_args['shuffle'] = False
    dataloader_args['dataset'] = val_dataset
    dataloader_args['batch_size'] = cfg.val.get("batch_size", 1)
    dataloader_args['num_workers'] = cfg.val.get("num_workers", 1) # 2
    val_dataloader, val_sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })


    # =========================================================================
    # 模型构建
    # =========================================================================
    log.info("Building models...")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # T5 文本编码器
    text_encoder = build_module(
        cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype
    )
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # VAE
    vae = build_module2(cfg.get("vae", None), MODELS2)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
    vae_out_channels = cfg.get("vae_out_channels", 16)

    # 主模型（MagicDriveSTDiT3v13）
    model_cfg = cfg.model.copy()
    trainable_mode = model_cfg.pop("trainable_mode", "new_only")
    latent_size = (None, None, None)
   
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
        )
        .to(device, dtype)
        .train()
    )

    # trainable_mode 已在 config 中设置，这里可以覆盖
    model.set_trainable_parameters(trainable_mode)
    log.info(f"Trainable mode: {trainable_mode}")

    # 文本 embedding 初始化（bbox_embedder 依赖 T5）
    if text_encoder is not None:
        model.prepare_text_embedding(text_encoder)

    # 部分加载 checkpoint
    if cfg.get("partial_load", None) and not cfg.get("load", None):
        load_dir = cfg.partial_load
        if os.path.isdir(load_dir):
            from glob import glob
            weight = {}
            for path in glob(os.path.join(load_dir, "model/pytorch_model-*")):
                weight.update(torch.load(path, map_location="cpu"))
        else:
            weight = torch.load(load_dir, map_location="cpu")
        m, u = model.load_state_dict(weight, strict=False)
        log.info(f"[partial_load] missing={len(m)}, unexpected={len(u)}")
        del weight

    model_numel, model_numel_trainable = get_model_numel(model)
    log.info(
        "[Model] Trainable: %s | Frozen: %s | Total: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel - model_numel_trainable),
        format_numel_str(model_numel),
    )

    # ── EMA（CPU fp32，无 OOM）────────────────────────────────────────────────
    ema = build_ema_on_cpu_v2(model)
    ema_shape_dict = record_model_param_shape(ema)
    log.info("EMA model built on CPU (fp32). GPU memory saved: ~%.1f GB",
             sum(p.numel() for p in ema.parameters()) * 4 / 1e9)

    # Scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # Optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)
    milestones_lr = cfg.get("milestones_lr", None)
    if warmup_steps is None:
        lr_scheduler = None
    elif milestones_lr is None:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)
    else:
        lr_scheduler = MultiStepWithLinearWarmupLR(
            optimizer, warmup_steps=warmup_steps, milestones_lr=milestones_lr
        )

    # ── 梯度 checkpoint ───────────────────────────────────────────────────────
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)

    # ── booster.boost ─────────────────────────────────────────────────────────
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )

    # ── checkpoint 加载（resume）─────────────────────────────────────────────
    start_epoch = start_step = 0
    if cfg.get("load", None) is not None:
        log.info("Loading checkpoint from %s", cfg.load)
        ret = load(
            booster, cfg.load,
            model=model, ema=ema, optimizer=optimizer,
            lr_scheduler=lr_scheduler, sampler=sampler,
        )
        if ret is not None:
            start_epoch, start_step, *_ = ret

    # debug 用：保存初始 checkpoint
    if cfg.get("save_init_ckpt", False):
        save_dir = save(
            booster, exp_dir,
            model=model, ema=ema, optimizer=optimizer,
            lr_scheduler=lr_scheduler, sampler=sampler,
            epoch=start_epoch, step=start_step,
            global_step=start_epoch * num_steps_per_epoch + start_step,
            batch_size=cfg.get("batch_size", None),
        )
        log.info("Initial checkpoint saved to %s", save_dir)

    model_sharding_ema(ema)  # CPU EMA 无需 shard，空操作

    # 训练前 validation
    if cfg.get("validation_before_run", False) and val_dataloader is not None:
        with RandomStateManager(verbose=True):
            coordinator.block_all()
            _run_validation_stdit3(
                cfg, text_encoder, vae, model,
                device, dtype, val_dataloader, coordinator,
                start_epoch * num_steps_per_epoch + start_step, exp_dir,
            )
            coordinator.block_all()

    # =========================================================================
    # 训练循环
    # =========================================================================
    log.info("Starting training from epoch %d, step %d", start_epoch, start_step)

    # mask generator（时序 mask）
    mask_generator = None
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    timers = {
        name: Timer(name)
        for name in ["data", "encode", "diffusion", "backward", "update_ema",
                     "reduce_loss", "misc"]
    }
    record_time = cfg.get("record_time", False)
    timer_list = []

    # == mask for i2v ==
    mask_types = cfg.get("mask_types", None)
    if mask_types is not None:
        mask_randgen = random.Random(dist.get_rank())

    for epoch in range(start_epoch, cfg.get("epochs", 1)):
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        log.info("Epoch %d / %d", epoch + 1, cfg.get("epochs", 1))

        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            running_loss = log_step = acc_step = 0
            optimizer.zero_grad()

            for step in pbar:
                # ── Step A: 数据 ─────────────────────────────────────────────
                with timers["data"] as t_data:
                    batch = next(dataloader_iter)
                if record_time:
                    timer_list.append(t_data)
                B, T, NC = batch["pixel_values"].shape[:3]
                # ── Step B: 编码 ─────────────────────────────────────────────
                with timers["encode"] as t_enc:
                    with torch.no_grad():
                        # VAE 编码
                        if cfg.get("load_video_features", False):
                            x = batch["pixel_values"].to(device, dtype)
                            if "first_frame" in batch:
                                first_frame_latent = batch["first_frame"].to(device, dtype)
                            else:
                                first_frame_latent = None
                        else:
                            raw_video = move_to(batch["pixel_values"], device, dtype) # shape torch.Size([24, 3, 17, 448, 840])
                            with RandomStateManager(verbose=verbose_mode):
                                x = rearrange(raw_video, "B T NC C ... -> (B NC) C T ...") # x shape torch.Size([24, 3, 17, 448, 840])
                                x = sp_vae(
                                    x, vae.encode,
                                    get_sequence_parallel_group()
                                ) # x shape torch.Size([24, 16, 5, 56, 105])
                                x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC) 
                                if "first_frame" in batch:
                                    first_frame_latent = sp_vae(
                                        move_to(batch["first_frame"], device, dtype),
                                        vae.encode,
                                        get_sequence_parallel_group(),
                                    )
                                    first_frame_latent = rearrange(first_frame_latent, "(B NC) C T ... -> B (C NC) T ...", NC=NC) 
                                else:
                                    first_frame_latent = None

                        # T5 文本编码
                        y = batch.pop("captions")[0]
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            ret = text_encoder.encode(y)
                            model_args = {k: v for k, v in ret.items()}

                # == prepare i2v&v2v mask_index ==
            
                # num_frames = x.shape[2]
                # latent_t = vae.get_latent_size(x.shape[2:])[0]
                # mask_index = []
                # text_uncond_prob = cfg.model.get("class_dropout_prob", 0.1)
                # if mask_types is not None:
                #     mask_cond = get_mask_cond(mask_randgen, mask_types)
                #     if num_frames > 1:  # NOTE: only use mask_indx for video
                #         mask_index = get_mask_index(mask_cond, latent_t)
                #         if len(mask_index) > 0:
                #             text_uncond_prob = 0.0

                maps = batch.pop("bev_map_with_aux").to(device, dtype)  # B, T, C, H, W
                bbox = batch.pop("bboxes_3d_data")
                # B len list (T, NC=1, len, 8, 3)
                bbox = [bbox_i.data for bbox_i in bbox]
                # B, T, NC, len, 8, 3
                # TODO: `bbox` has redundancy on `NC` dim. They are direct
                # copies and should be differentiate through mask.
                bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
                if bbox is not None:
                    bbox = add_box_latent(bbox, B, NC, T, model.module.sample_box_latent)

                    for k, v in bbox.items():
                        bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7

                if record_time:
                    timer_list.append(t_enc)

                # ── Step C: 条件 drop mask ───────────────────────────────────
                B = x.shape[0]
                NC = len(cfg.get("mv_order_map", {1: []}))

                drop_cond_ratio = cfg.get("drop_cond_ratio", 0.15)
                drop_frame_ratio = cfg.get("drop_cond_ratio_t", 0.4)
                
                drop_cond_mask = (
                    torch.rand(B, device=device) >= drop_cond_ratio
                ).float()

                real_T = batch["frame_emb"].shape[1]
                drop_frame_mask = (
                    torch.rand(B, real_T, device=device) >= drop_frame_ratio
                ).float()

                # ── Step D: 时序 mask ────────────────────────────────────────
                x_mask = None
                if mask_generator is not None:
                    # x shape: (B, C*NC, T, H, W)，latent T = (T_pixel-1)//4+1
                    latent_T = x.shape[2]
                    if latent_T > 1:
                        x_mask = mask_generator.get_masks(x)

                # ── Step E: latent shape 信息 ────────────────────────────────
                _, CNC, latent_T, latent_H, latent_W = x.shape
                H_pixel = latent_H * 8   # VAE 空间压缩比 8
                W_pixel = latent_W * 8

                # ── Step F: model_kwargs 构建 ────────────────────────────────
                # B, T, NC, 3, 7
                cams = batch.pop("camera_param").to(device, dtype)
                cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
                rel_pos = batch.pop("frame_emb").to(device, dtype)
                rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4
                
                model_args["maps"] = maps
                model_args["bbox"] = bbox
                model_args["cams"] = cams
                model_args["rel_pos"] = rel_pos
                model_args["drop_cond_mask"] = drop_cond_mask
                model_args["drop_frame_mask"] = drop_frame_mask
                model_args["fps"] = batch.pop('fps')
                model_args["height"] = batch.pop("height")
                model_args["width"] = batch.pop("width")
                model_args["num_frames"] = batch.pop("num_frames")
                model_args = move_to(model_args, device=device, dtype=dtype)
                # no need to move these
                model_args["mv_order_map"] = cfg.get("mv_order_map")
                model_args["t_order_map"] = cfg.get("t_order_map")
                model_kwargs = {
                    **model_args,
                    "drop_cond_mask": drop_cond_mask,
                    "drop_frame_mask": drop_frame_mask,
                    "x_mask": x_mask,
                    "first_frame_latent": first_frame_latent,
                }
                
                # ── Step G: 扩散 loss ────────────────────────────────────────
                with timers["diffusion"] as t_diff:
                    loss_dict = scheduler.training_losses(
                        model, x, model_kwargs, mask=x_mask
                    )
                    loss = loss_dict["loss"].mean()
                if record_time:
                    timer_list.append(t_diff)

                # ── Step H: 反向传播 ─────────────────────────────────────────
                with timers["backward"] as t_bwd:
                    booster.backward(loss, optimizer)
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(t_bwd)

                # ── Step I: EMA 更新 ─────────────────────────────────────────
                with timers["update_ema"] as t_ema:
                    update_ema_cpu(
                        ema, model.module,
                        decay=cfg.get("ema_decay", 0.9999),
                    )
                if record_time:
                    timer_list.append(t_ema)

                # ── Step J: 统计 / log ───────────────────────────────────────
                with timers["reduce_loss"] as t_reduce:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(t_reduce)

                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": avg_loss, "step": str(step),
                        "global_step": str(global_step), "lr": lr,
                    })
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    tb_writer.add_scalar("avg_loss", avg_loss, global_step)
                    tb_writer.add_scalar("lr", lr, global_step)
                    running_loss = 0.0
                    log_step = 0

                # ── Step K: Checkpoint 保存 ──────────────────────────────────
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering_ema(ema, ema_shape_dict)  # 空操作（CPU EMA）
                    save_dir = save(
                        booster, exp_dir,
                        model=model, ema=ema, optimizer=optimizer,
                        lr_scheduler=lr_scheduler, sampler=sampler,
                        epoch=epoch, step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if dist.get_rank() == 0:
                        model_sharding_ema(ema)  # 空操作（CPU EMA）
                    log.info(
                        "Checkpoint saved at epoch %d, step %d, global_step %d → %s",
                        epoch, step + 1, global_step + 1, save_dir,
                    )

                # ── Step L: Validation ───────────────────────────────────────
                report_every = cfg.get("report_every", 0)
                if (
                    report_every > 0
                    and (global_step + 1) % report_every == 0
                    and val_dataloader is not None
                ):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    with RandomStateManager(verbose=True):
                        coordinator.block_all()
                        _run_validation_stdit3(
                            cfg, text_encoder, vae, model,
                            device, dtype, val_dataloader, coordinator,
                            global_step + 1, exp_dir,
                        )
                        coordinator.block_all()
                    torch.cuda.empty_cache()

        # epoch 结束，重置 start_step
        start_step = 0

    log.info("Training complete.")


if __name__ == "__main__":
    main()