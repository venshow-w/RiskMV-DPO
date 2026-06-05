import os
from contextlib import nullcontext
import sys
import random
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
import torch.nn.functional as F
if not torch.cuda.is_available() or DEVICE_TYPE == 'npu':
    USE_NPU = True
    os.environ['DEVICE_TYPE'] = "npu"
    DEVICE_TYPE = "npu"
    print("Enable NPU!")
    try:
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import magicdrivedit.utils.module_contrib

import torch.distributed as dist
from einops import rearrange, repeat
import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm
from mmcv.parallel import DataContainer

import logging
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('shapely.geos').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.INFO)
logging.getLogger('magicdrivedit.models.vae.vae_cogvideox').setLevel(logging.WARNING)

from magicdrivedit.acceleration.checkpoint import set_grad_checkpoint
from magicdrivedit.acceleration.parallel_states import get_data_parallel_group, get_sequence_parallel_group
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save, prepare_ckpt, RandomStateManager
from magicdrivedit.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
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
    add_box_latent,
)
from magicdrivedit.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema, run_validation, sp_vae

# from localdpo.multiview_mask import MultiViewSpatioTemporalMask, SemanticAwareMaskGenerator
from localdpo.corrupter import MultiViewLocalCorrupter
# from localdpo.vggt_scorer import VGGTGeoScorer, VGGTGeometryAdapter
from localdpo.localdpo_loss import MultiViewRegionAwareDPOLoss
from localdpo.motion_aware_mask import MotionAwareMaskGenerator


def build_motion_mask_generator(cfg, device):
    """构建运动感知遮罩生成器"""
    cfg_dtype = cfg.get("dtype", "bf16")
    dtype = torch.bfloat16 if cfg_dtype == "bf16" else torch.float16
    mask_gen = MotionAwareMaskGenerator(
        img_size=(448, 840),
        num_cameras=6,
        mask_size_range=cfg.get('mask_size_range', (0.1, 0.25)),
        motion_threshold=cfg.get('motion_threshold', 0.6),   # FIXED: 降低默认阈值
        use_semantic_prior=cfg.get('use_semantic_prior', True),
        fallback_mask_ratio=cfg.get('fallback_mask_ratio', 0.15),  # FIXED: 新增 fallback 参数
    ).to(device, dtype=dtype)
    mask_gen.train()
    return mask_gen


def collect_trainable_params(model, motion_mask_gen, cfg):
    """
    统一收集可训练参数，避免重复。
    """
    trainable_params = []
    param_ids = set()

    def add_param(param, name):
        if param.requires_grad and id(param) not in param_ids:
            trainable_params.append(param)
            param_ids.add(id(param))

    # 1. 运动遮罩生成器
    for name, param in motion_mask_gen.named_parameters():
        param.requires_grad = True
        add_param(param, f"motion_mask.{name}")

    # 2. 几何适配器
    if hasattr(model, 'geometry_adapter'):
        for name, param in model.geometry_adapter.named_parameters():
            param.requires_grad = True
            add_param(param, f"geo_adapter.{name}")

    # 3. GAM 模块
    for module_name, module in model.named_modules():
        if 'gam' in module_name.lower() and hasattr(module, 'parameters'):
            for name, param in module.named_parameters():
                param.requires_grad = True
                add_param(param, f"gam.{module_name}.{name}")

    return trainable_params


# FIXED: 用于传给冻结 EMA 模型的合法参数集合
_VALID_MODEL_ARGS = {
    "y", "mask", "maps", "bbox", "cams", "rel_pos",
    "fps", "height", "width", "num_frames",
    "drop_cond_mask", "drop_frame_mask",
    "mv_order_map", "t_order_map",
    "x_mask", "frame_idx", "first_frame_images", "frames_mask",
}


def _filter_model_args(model_args: dict) -> dict:
    """过滤 model_args，只保留模型 forward 接受的字段。"""
    return {k: v for k, v in model_args.items() if k in _VALID_MODEL_ARGS}


def get_ema_output(ema, ema_shape_dict, x_corrupted, t_noise, model_args):
    """
    用 EMA 模型进行一次 forward（作为 DPO reference）。
    FIXED: 过滤 model_args，避免传入不兼容的 booster 字段。
    """
    ema = ema.to(torch.bfloat16)

    # FIXED: 数据类型安全处理
    safe_model_args = _filter_model_args(model_args)
    for key in ["height", "width", "num_frames"]:
        if key in safe_model_args and hasattr(safe_model_args[key], 'dtype'):
            if safe_model_args[key].dtype == torch.float16:
                safe_model_args[key] = safe_model_args[key].float()

    with torch.no_grad():
        ema_output = ema(x_corrupted, t_noise, **safe_model_args)

    return ema_output


def get_compressed_frame_idx(T_original, device):
    """获取 VAE 压缩后的帧索引。"""
    T_latent = T_original // 4 + 1
    compressed_idx = []
    for t in range(T_latent):
        if t == 0:
            compressed_idx.append(0)
        else:
            original_frame = (t - 1) * 4 + 2
            compressed_idx.append(original_frame)
    return torch.tensor(compressed_idx, device=device)


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    cfg = parse_configs(training=True)
    if cfg.get("vsdebug", False):
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
        cfg.record_time = True
    enable_debug = cfg.get("debug", False)
    if enable_debug:
        cfg.outputs = os.path.join(cfg.get("outputs", "outputs"), "debug")
        cfg.ckpt_every = 50
        cfg.record_time = True
    verbose_mode = cfg.get("verbose_mode", False)
    if verbose_mode:
        cfg.record_time = True
    record_time = cfg.get("record_time", False)

    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets = []
        val_datasets = []
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
            cfg.num_frames)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    if USE_NPU:
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)

    set_seed(cfg.get("seed", 1024))
    torch.cuda.manual_seed_all(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    coordinator._local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
    device = get_current_device()

    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    coordinator.block_all()
    if coordinator.is_node_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    logger = reset_logger(exp_dir, enable_debug)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    logger.info(f"ColossalAI version: {colossalai.__version__}")
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)

    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
        overlap_allgather=cfg.get("overlap_allgather", False),
        verbose=verbose_mode,
    )
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")

    def build_real_dataset(cfg):
        from magicdrivedit.utils.config_utils import merge_dataset_cfg
        if cfg.get("num_frames", None) is None:
            num_data_cfgs = len(cfg.data_cfg_names)
            datasets = []
            for idx, (res, data_cfg_name) in enumerate(cfg.data_cfg_names):
                overrides = cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)[idx]
                dataset, _ = merge_dataset_cfg(cfg, data_cfg_name, overrides)
                datasets.append((res, dataset))
            cfg.dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        else:
            cfg.dataset, _ = merge_dataset_cfg(
                cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
                cfg.num_frames)
        return build_module(cfg.dataset, DATASETS)

    dataset = build_real_dataset(cfg)
    if cfg.get("overfit", None) is not None:
        _overfit_idxs = random.sample(range(len(dataset)), cfg.overfit)
        logger.info(f"Overfit on: {_overfit_idxs}")
        overfit_idxs = []
        for _ in range(cfg.epochs):
            overfit_idxs += _overfit_idxs
            random.shuffle(_overfit_idxs)
        cfg.epochs = 1
        dataset = torch.utils.data.Subset(dataset, overfit_idxs)
    logger.info("Dataset contains %s samples.", len(dataset))

    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 1),
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

    if cfg.get("overfit", None) is not None:
        val_dataset = torch.utils.data.Subset(dataset, list(range(cfg.overfit)))
    else:
        val_dataset = build_module(cfg.val_dataset, DATASETS)
        if cfg.val.validation_index != "all":
            if len(cfg.val.validation_index) < get_data_parallel_group().size():
                if isinstance(cfg.val.validation_index[0], int):
                    cfg.val.validation_index += random.sample(
                        list(set(range(len(val_dataset))) - set(cfg.val.validation_index)),
                        min(get_data_parallel_group().size(), 32) - len(cfg.val.validation_index),
                    )
                    if get_data_parallel_group().size() > 32:
                        while len(cfg.val.validation_index) < get_data_parallel_group().size():
                            cfg.val.validation_index += random.sample(
                                list(set(range(len(val_dataset))) - set(cfg.val.validation_index)), 1,
                            )
                else:
                    while len(cfg.val.validation_index) < get_data_parallel_group().size():
                        new_key = val_dataset.rand_another_key()
                        if new_key not in cfg.val.validation_index:
                            cfg.val.validation_index.append(new_key)
                logging.info(f"validation_index rewrite as: {cfg.val.validation_index}")
            val_dataset = torch.utils.data.Subset(val_dataset, cfg.val.validation_index)
        else:
            raise NotImplementedError()
    logger.info("Val Dataset contains %s samples.", len(val_dataset))
    dataloader_args['shuffle'] = False
    dataloader_args['dataset'] = val_dataset
    dataloader_args['batch_size'] = cfg.val.get("batch_size", 1)
    dataloader_args['num_workers'] = cfg.val.get("num_workers", 1)
    val_dataloader, val_sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()

    latent_size = (None, None, None)
    vae_out_channels = cfg.get("vae_out_channels", 4)

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

    model.prepare_text_embedding(text_encoder)
    if cfg.get("partial_load", None) and not cfg.get("load", None):
        load_dir = cfg.partial_load
        if os.path.isdir(load_dir):
            from glob import glob
            weight = {}
            for path in glob(os.path.join(load_dir, "model/pytorch_model-*")):
                weight.update(torch.load(path, map_location="cpu"))
        else:
            weight = torch.load(load_dir, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)
        logger.info(f"[partial load] Missing keys: {missing_keys}")
        logger.info(f"[partial load] Unexpected keys: {unexpected_keys}")
        del weight, missing_keys, unexpected_keys

    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Fix: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel - model_numel_trainable),
        format_numel_str(model_numel),
    )

    # ==================== LocalDPO 模块初始化 ====================
    logger.info("Initializing LocalDPO modules...")

    # 1. 运动感知遮罩生成器
    motion_mask_gen = build_motion_mask_generator(cfg, device)

    # 2. 局部腐败器（FIXED: 使用修复后的 corrupter）
    corrupter = MultiViewLocalCorrupter(
        diffusion_model=model.module if hasattr(model, 'module') else model,
        num_timesteps=cfg.get("num_timesteps", 1000),
        device=device,
    ).to(device)

    # 3. DPO 损失函数（FIXED: 使用修复后的 loss）
    dpo_loss_fn = MultiViewRegionAwareDPOLoss(
        lambda_ra=cfg.get("lambda_ra", 0.01),
        lambda_sft=cfg.get("lambda_sft", 1.0),
        lambda_align=cfg.get("lambda_align", 0.001),
        align_type=cfg.get('align_type', 'cosine'),
        alpha_l=cfg.get("alpha_l", 0.1),
        alpha_h=cfg.get("alpha_h", 0.9),
        temperature=cfg.get("temperature", 0.1),
        num_timesteps=cfg.get("num_timesteps", 1000),
        beta=0.1,
    ).to(device)

    # 4. 可训练参数收集
    trainable_params = collect_trainable_params(model, motion_mask_gen, cfg)
    total_params = sum(p.numel() for p in trainable_params)
    logger.info(f"\n{'='*50}")
    logger.info(f"总计可训练参数: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数列表长度: {len(trainable_params)}")
    logger.info(f"{'='*50}\n")
    assert len(trainable_params) > 0, "错误：可训练参数列表为空！"

    # EMA
    ema = deepcopy(model).to(torch.bfloat16).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    optimizer = HybridAdam(
        trainable_params,
        adamw_mode=True,
        lr=cfg.get("lr", 2e-5),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)
    milestones_lr = cfg.get("milestones_lr", None)
    if warmup_steps is None:
        lr_scheduler = None
    else:
        if milestones_lr is None:
            lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)
        else:
            lr_scheduler = MultiStepWithLinearWarmupLR(
                optimizer, milestones_lr=milestones_lr, warmup_steps=warmup_steps)

    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # ======================================================
    # 4. distributed training preparation
    # ======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    drop_cond_ratio = cfg.get("drop_cond_ratio", 0.0)
    drop_cond_ratio_t = cfg.get("drop_cond_ratio_t", 0.4)
    running_loss = 0.0
    running_loss_ra = 0.0
    running_loss_align = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=None if cfg.get("reset_lr", False) or cfg.get("start_from_scratch", False) else lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
            local_master=coordinator.is_node_master(),
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
            if cfg.get("reset_lr", False) and lr_scheduler:
                total_step = start_epoch * num_steps_per_epoch + start_step
                lr_scheduler.last_epoch = total_step
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    if enable_debug:
        save_dir = save(
            booster, exp_dir, model=model, ema=ema, optimizer=optimizer,
            lr_scheduler=lr_scheduler, sampler=sampler, epoch=start_epoch,
            step=start_step, global_step=start_epoch * num_steps_per_epoch + start_step,
            batch_size=cfg.get("batch_size", None),
        )
        logger.info(f"Save your model to {save_dir} before training.")

    model_sharding(ema)

    if cfg.get("validation_before_run", False):
        with RandomStateManager(verbose=True):
            coordinator.block_all()
            run_validation(
                cfg.val, text_encoder, vae, model, device, dtype,
                val_dataloader, coordinator,
                start_epoch * num_steps_per_epoch + start_step,
                exp_dir, cfg.mv_order_map, cfg.t_order_map,
            )
            val_sampler.reset()

    with RandomStateManager(verbose=True):
        print(f"{torch.randn(3)} {torch.randn(3, device=get_current_device())} "
              f"on rank {dist.get_rank()} "
              f"dp_rank {dist.get_rank(get_data_parallel_group())}")

    # ======================================================
    # 5. training loop
    # ======================================================
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    coordinator.block_all()
    timers = {}
    timer_keys = [
        "move_data", "encode", "move_data2", "mask",
        "localdpo_corrupt", "diffusion", "dpo_loss",
        "backward", "update_ema", "reduce_loss", "misc",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=None)
        else:
            timers[key] = nullcontext()

    # EMA full（用于 DPO reference，与保存 checkpoint 的 ema 分开）
    ema_full = deepcopy(model).to(device, dtype=torch.bfloat16)
    requires_grad(ema_full, False)
    ema_full.eval()
    update_ema(ema_full, model, decay=cfg.get("ema_decay", 0.9999), sharded=False)

    for epoch in range(start_epoch, cfg_epochs):
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                if verbose_mode:
                    logger.info(f"Dataloader returns data! step={step}")
                B, T, NC = batch["pixel_values"].shape[:3]
                logging.debug(f"bs = {B}; t = {T}; shape = {batch['pixel_values'].shape}")
                timer_list = []

                with timers["move_data"] as move_data_t:
                    x = batch.pop("pixel_values").to(device, dtype)
                    x = rearrange(x, "B T NC C ... -> (B NC) C T ...")
                    y = batch.pop("captions")[0]
                    first_frame_images = batch.pop("first_frames").to(device, dtype)
                    maps = batch.pop("bev_map_with_aux").to(device, dtype)
                    bbox = batch.pop("bboxes_3d_data")
                    cams = batch.pop("camera_param").to(device, dtype)
                    rel_pos = batch.pop("frame_emb").to(device, dtype)

                    bbox = [bbox_i.data for bbox_i in bbox]
                    bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
                    if bbox is not None:
                        bbox = add_box_latent(bbox, B, NC, T, model.module.sample_box_latent)
                        for k, v in bbox.items():
                            bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")

                    cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")
                    rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)
                if record_time:
                    timer_list.append(move_data_t)

                with timers["encode"] as encode_t:
                    with torch.no_grad():
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            with RandomStateManager(verbose=verbose_mode):
                                x = sp_vae(x, vae.encode, get_sequence_parallel_group())

                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            ret = text_encoder.encode(y)
                            model_args = {k: v for k, v in ret.items()}
                if record_time:
                    timer_list.append(encode_t)

                with timers["move_data2"] as move_data_t:
                    drop_cond_mask = torch.ones((B,))
                    drop_frame_mask = torch.ones((B, T))
                    if drop_cond_ratio > 0:
                        for bs in range(B):
                            if random.random() < drop_cond_ratio:
                                drop_cond_mask[bs] = 0
                                drop_frame_mask[bs, :] = 0
                                model_args["mask"][bs] = 1
                                continue
                            t_ids = random.sample(range(1, T - 1), int(drop_cond_ratio_t * (T - 2)))
                            drop_frame_mask[bs, t_ids] = 0

                    T_latent = T // 4 + 1
                    frame_idx = torch.arange(T_latent, device=device).view(1, 1, T_latent).expand(B, NC, -1)
                    frame_idx = rearrange(frame_idx, 'B NC T -> (B NC) T')

                    model_args["frame_idx"] = frame_idx
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
                    model_args["mv_order_map"] = cfg.get("mv_order_map")
                    model_args["t_order_map"] = cfg.get("t_order_map")
                if record_time:
                    timer_list.append(move_data_t)

                with timers["mask"] as mask_t:
                    x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
                    scheduler_mask = None
                    if cfg.get("mask_ratios", None) is not None:
                        scheduler_mask = mask_generator.get_masks(x)
                        model_args["x_mask"] = scheduler_mask
                if record_time:
                    timer_list.append(mask_t)

                model_args['frames_mask'] = torch.zeros_like(x)
                model_args['first_frame_images'] = first_frame_images

                # ============ LocalDPO 腐败流程 ============
                real_video = rearrange(x, "B (C NC) T H W -> B NC C T H W", NC=NC)

                # 生成运动感知遮罩
                motion_mask = motion_mask_gen(real_video, first_frame_images)

                # 腐败并修复（FIXED: 使用统一 RFlow 噪声）
                with timers["localdpo_corrupt"] as corrupt_t:
                    corrupted_video, mask_corrupt, t_noise, _ = corrupter.corrupt_and_restore(
                        real_video,
                        motion_mask,
                        model_args,   # corrupter 内部会过滤合法字段
                    )
                if record_time:
                    timer_list.append(corrupt_t)

                x_corrupted = rearrange(corrupted_video, "B NC C T H W -> B (C NC) T H W")

                # ============ 扩散模型前向 ============
                with timers["diffusion"] as loss_t:
                    loss_dict = scheduler.training_losses(
                        model, x_corrupted, model_args,
                        mask=scheduler_mask,
                        return_x0_pred=True,
                        return_features=True,
                        feature_layers=list(range(20, 28)),
                    )
                    original_loss = loss_dict['loss'].detach().clone()
                if record_time:
                    timer_list.append(loss_t)

                # ============ LocalDPO 损失 ============
                with timers["dpo_loss"] as dpo_t:
                    if 'x0_pred' in loss_dict:
                        x0_pred = loss_dict['x0_pred']
                        model_output_video = rearrange(
                            x0_pred, "B (C NC) T H W -> B NC C T H W", NC=NC
                        )

                        # 7.1 几何 latent（通过 geometry_adapter）
                        geometry_latents = None
                        if hasattr(model.module, 'geometry_adapter'):
                            # FIXED: 直接调用 geometry_adapter，而不是 extract_geometry_latents
                            # 确保梯度能流向 geometry_adapter
                            geometry_latents = model.module.geometry_adapter(
                                first_frame_images,
                                drop_cond_mask=drop_cond_mask,
                            )

                        # 7.2 RGB 特征（用于对齐损失）
                        if 'features' in loss_dict and geometry_latents is not None:
                            rgb_features = loss_dict['features']   # [B*NC, D]
                        else:
                            # Fallback：对腐败视频做全局平均池化
                            rgb_features = x0_pred.mean(dim=[2, 3, 4])  # [B, C*NC] → pool

                        # 7.3 EMA reference（FIXED: 使用修复后的 get_ema_output）
                        ema_output_raw = get_ema_output(
                            ema_full, ema_shape_dict, x_corrupted, t_noise, model_args
                        )
                        # 处理双通道输出
                        if ema_output_raw.shape[1] == 2 * x_corrupted.shape[1]:
                            ema_output_raw = ema_output_raw.chunk(2, dim=1)[0]

                        # EMA velocity → x0_pred（用 RFlow 还原）
                        alpha_t = 1.0 - t_noise.float() / cfg.get("num_timesteps", 1000)
                        one_minus_alpha = (1.0 - alpha_t).view(-1, *([1] * (x_corrupted.dim() - 1)))
                        ema_x0_pred_flat = x_corrupted + one_minus_alpha * ema_output_raw
                        ema_output_video = rearrange(
                            ema_x0_pred_flat, "B (NC C) T H W -> B NC C T H W", NC=NC
                        )

                        # 7.4 DPO 损失（FIXED: 参数名统一为 ref_output）
                        dpo_loss_dict = dpo_loss_fn(
                            x0_pred=model_output_video,
                            target=real_video,
                            mask=motion_mask,
                            t=t_noise,
                            geometry_latents=geometry_latents,
                            rgb_features=rgb_features,
                            ref_output=ema_output_video,   # FIXED: 统一为 ref_output
                            return_stats=True,
                            current_step=step,
                        )

                        # 7.5 动态混合损失权重
                        use_mixed_loss = cfg.get("use_mixed_loss", True)
                        if use_mixed_loss:
                            sft_value = dpo_loss_dict['loss_sft'].item()
                            if sft_value > 0.8:
                                dpo_weight = 0.005
                            elif sft_value > 0.5:
                                dpo_weight = 0.01
                            else:
                                dpo_weight = 0.05
                            loss_dict['loss'] = original_loss + dpo_weight * dpo_loss_dict['loss']
                        else:
                            loss_dict['loss'] = dpo_loss_dict['loss']

                        loss_dict['loss_ra'] = dpo_loss_dict['loss_ra']
                        loss_dict['loss_sft'] = dpo_loss_dict['loss_sft']
                        loss_dict['loss_align'] = dpo_loss_dict['loss_align']
                        if 'loss_ra_avg' in dpo_loss_dict:
                            loss_dict['loss_ra_avg'] = dpo_loss_dict['loss_ra_avg']
                            loss_dict['loss_sft_avg'] = dpo_loss_dict['loss_sft_avg']
                            loss_dict['loss_align_avg'] = dpo_loss_dict['loss_align_avg']
                if record_time:
                    timer_list.append(dpo_t)

                # ============ 反向传播 ============
                coordinator.block_all()
                if verbose_mode:
                    logger.info(f"Start model backward step! step={step}, loss={loss_dict['loss']}")

                # FIXED: 修正日志中未定义 loss 变量的 Bug
                if step % 100 == 0 and coordinator.is_master():
                    logger.info(f"\n=== Loss Values (Step {step}) ===")
                    logger.info(f"  Original diffusion loss: {original_loss:.4f}")
                    logger.info(f"  DPO loss_ra: {loss_dict.get('loss_ra', torch.tensor(0.0)).item():.4f}")
                    logger.info(f"  DPO loss_sft: {loss_dict.get('loss_sft', torch.tensor(0.0)).item():.4f}")
                    logger.info(f"  DPO loss_align: {loss_dict.get('loss_align', torch.tensor(0.0)).item():.4f}")
                    # FIXED: 使用正确的变量名（loss_dict['loss'] 而非 loss）
                    logger.info(f"  Total loss: {loss_dict['loss'].item():.4f}")
                    if 'loss_ra_avg' in loss_dict:
                        logger.info(f"  Avg loss_ra: {loss_dict['loss_ra_avg']:.4f}")
                        logger.info(f"  Avg loss_sft: {loss_dict['loss_sft_avg']:.4f}")
                        logger.info(f"  Avg loss_align: {loss_dict['loss_align_avg']:.4f}")
                    logger.info("=" * 40)

                with timers["backward"] as backward_t:
                    # FIXED: total_loss 从 loss_dict 取，避免 loss 未定义
                    total_loss = loss_dict["loss"].mean()
                    booster.backward(loss=total_loss, optimizer=optimizer)
                    optimizer.step()
                    if enable_debug:
                        for n, p in model.named_parameters():
                            if not (p == p).all():
                                logger.info(f"Got nan on {n}")
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                with timers["update_ema"] as ema_t:
                    # 更新 checkpoint EMA
                    update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                    # 更新 reference EMA（同步更新）
                    update_ema(ema_full, model, decay=cfg.get("ema_decay", 0.9999), sharded=False)
                if record_time:
                    timer_list.append(ema_t)

                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(total_loss)
                    running_loss += total_loss.item()
                    running_loss_ra += loss_dict.get('loss_ra', torch.tensor(0.0)).mean().item()
                    running_loss_align += loss_dict.get('loss_align', torch.tensor(0.0)).mean().item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                if record_time:
                    misc_t = timers['misc'].__enter__()
                    timer_list.append(misc_t)

                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    avg_loss_ra = running_loss_ra / log_step
                    avg_loss_align = running_loss_align / log_step
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": avg_loss,
                        "loss_ra": f"{avg_loss_ra:.4f}",
                        "loss_align": f"{avg_loss_align:.4f}",
                        "step": str(step),
                        "global_step": str(global_step),
                        "lr": lr,
                    })
                    tb_writer.add_scalar("loss", total_loss.item(), global_step)
                    tb_writer.add_scalar("avg_loss", avg_loss, global_step)
                    tb_writer.add_scalar("loss/ra_dpo", avg_loss_ra, global_step)
                    tb_writer.add_scalar("loss/align", avg_loss_align, global_step)
                    tb_writer.add_scalar("lr", lr, global_step)
                    running_loss = 0.0
                    running_loss_ra = 0.0
                    running_loss_align = 0.0
                    log_step = 0

                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster, exp_dir, model=model, ema=ema, optimizer=optimizer,
                        lr_scheduler=lr_scheduler, sampler=sampler, epoch=epoch,
                        step=step + 1, global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    import torch.nn as nn
                    if isinstance(motion_mask_gen, nn.Module):
                        torch.save(
                            motion_mask_gen.state_dict(),
                            os.path.join(save_dir, "motion_mask_gen.pt"),
                        )
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch, step + 1, global_step + 1, save_dir,
                    )

                report_every = cfg.get("report_every", 0)
                if report_every > 0 and (global_step + 1) % report_every == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    val_dir = run_validation(
                        cfg.val, text_encoder, vae, model, device, dtype,
                        val_dataloader, coordinator, global_step + 1,
                        exp_dir, cfg.mv_order_map, cfg.t_order_map,
                    )
                    val_sampler.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                if record_time:
                    misc_t.__exit__(*sys.exc_info())
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    log_str += f"Total: {sum([t.elapsed_time for t in timer_list]):.3f}s"
                    logger.info(log_str)

                if enable_debug and step > 50:
                    break

        if enable_debug:
            break
        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()