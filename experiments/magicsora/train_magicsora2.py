"""
train_magicdrive_mmdit.py

MagicDrive-MMDiT 多视图视频生成训练脚本。
基于原 train_magicdrive.py 修改，适配 OpenSora2 (MMDiT/Flux) 骨干网络。

主要改动（相对于原 train_magicdrive.py）：
  1. 模型构建：使用 MagicDriveMMDiT 替换 MagicDriveSTDiT3
     - 去掉 input_size / caption_channels / model_max_length 传参（MMDiT 内部处理）
     - 增加 from_opensora2_pretrained 权重加载路径
     - 增加 trainable_mode 控制哪些参数参与训练

  2. 模型 forward 参数适配：
     - 去掉 fps / height / width / num_frames（MMDiT 不需要 pos-embed resize）
     - 去掉 t_order_map（MMDiT 内部用 RoPE，不需要帧间序列对）
     - 增加 y_vec（CLIP 池化向量，MMDiT vec 流需要）
     - mask / x_mask 语义保持不变

  3. Scheduler 调用适配：
     - RFlowScheduler.training_losses 签名不变，但 model_kwargs 字段更新

  4. 文本编码适配：
     - 需要同时获得 T5 隐层（y）和 CLIP 池化向量（y_vec）
     - 增加 clip_text_encoder 构建与编码逻辑

  5. 训练阶段控制（trainable_mode）：
     - "new_only": 只训练新增模块（cross_view / temporal / control / embedders）
     - "control": 只训练控制分支 + 条件编码器
     - "all": 全量微调

  6. EMA OOM 修复（本次改动）：
     原始写法 `deepcopy(model).to(torch.float32).to(device)` 会在 GPU 上产生
     三阶段显存峰值（model bf16 + EMA bf16副本 + EMA fp32 = ~114 GB），
     单张 A100/H100 80GB OOM。
     修复：EMA 全程在 CPU fp32，通过 build_ema_on_cpu_v2 / update_ema_cpu 实现，
     GPU 显存节省 ~57 GB，且与 save/load checkpoint 接口完全兼容。
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
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('shapely.geos').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.INFO)
logging.getLogger('magicdrivedit.models.vae.vae_cogvideox').setLevel(logging.WARNING)

from magicdrivedit.acceleration.checkpoint import set_grad_checkpoint
from magicdrivedit.acceleration.parallel_states import (
    get_data_parallel_group, get_sequence_parallel_group
)
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.ckpt_utils import (
    load, model_gathering, model_sharding, record_model_param_shape,
    save, prepare_ckpt, RandomStateManager,
)
from magicdrivedit.utils.config_utils import (
    define_experiment_workspace, parse_configs, save_training_config,
    merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set,
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
    add_box_latent,
)
from magicdrivedit.utils.train_utils import (
    MaskGenerator, create_colossalai_plugin, update_ema,
    run_validation, sp_vae,
)

# ── EMA 显存修复工具（替换 deepcopy OOM 方案）─────────────────────────────────
from ema_utils import (
    build_ema_on_cpu_v2,
    update_ema_cpu,
    model_sharding_ema,
    model_gathering_ema,
)
from opensora.registry import MODELS as MODELS2
from opensora.registry import build_module as build_module2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP 文本编码器封装（获取池化向量 y_vec，供 MMDiT vec 流使用）
# ─────────────────────────────────────────────────────────────────────────────

class CLIPTextEncoderWrapper:
    """
    封装 CLIP 文本编码器，提供 encode(texts) → y_vec (B, clip_dim) 接口。
    
    MMDiT（Flux）需要两路文本编码：
      - T5 大模型 → 长序列上下文 y (B, L, 4096)，输入 txt 流
      - CLIP → 池化向量 y_vec (B, 768)，输入 vec 流（与时间步相加）
    
    如果没有 CLIP 编码器（仅用 T5），y_vec 用全零替代，
    模型仍能运行但 vec 流缺少语义信息。
    """
    
    def __init__(self, clip_model_path: str, device, dtype):
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
            self.model = CLIPTextModel.from_pretrained(clip_model_path).to(device, dtype)
            self.model.eval()
            self.device = device
            self.dtype = dtype
            self.available = True
            self.output_dim = self.model.config.hidden_size
            logger.info(f"CLIP text encoder loaded from {clip_model_path}, dim={self.output_dim}")
        except Exception as e:
            logger.warning(f"Failed to load CLIP encoder: {e}. y_vec will be zeros.")
            self.available = False
            self.output_dim = 768  # Flux default
    
    @torch.no_grad()
    def encode(self, texts, device=None):
        """
        texts: list of str, length B
        返回: (B, output_dim) 的池化向量
        """
        if not self.available:
            B = len(texts)
            return torch.zeros(B, self.output_dim, device=device or self.device,
                               dtype=self.dtype)
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=77,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # 使用 pooler_output 或 last_hidden_state[:,0,:]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.to(self.dtype)
        return outputs.last_hidden_state[:, 0].to(self.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# forward 参数构建（适配 MMDiT 接口）
# ─────────────────────────────────────────────────────────────────────────────

def build_model_kwargs_mmdit(
    # 原始数据
    y_t5: torch.Tensor,         # (B, L, 4096) T5 隐层序列
    y_vec: torch.Tensor,        # (B, 768)     CLIP 池化向量
    maps: torch.Tensor,         # (B, T, C, H, W) BEV map
    bbox: dict,                 # collated bbox dict
    cams: torch.Tensor,         # (B*NC, T, 1, 3, 7)
    rel_pos: torch.Tensor,      # (B*NC, T, 1, 4, 4)
    drop_cond_mask: torch.Tensor,
    drop_frame_mask: torch.Tensor,
    mv_order_map: dict,
    x_mask=None,                # (B, T) 帧遮盖 mask（用于可变长度视频）
    first_frame_latent=None,    # (B, C*NC, 1, H, W) 首帧引导
    device=None,
    dtype=None,
):
    """
    构建 MagicDriveMMDiT.forward() 所需的 model_kwargs 字典。
    
    与原版 train_magicdrive.py 的差异：
      ✓ 增加 y_vec (CLIP 池化向量)
      ✓ 去掉 fps / height / width / num_frames（MMDiT 不需要 pos-embed resize）
      ✓ 去掉 t_order_map（MMDiT 不需要帧间序列对）
      ✓ mask → x_mask (语义不变，改名对齐 MMDiT 接口)
    """
    model_args = {
        # 双流文本条件
        "y": y_t5,                        # T5 序列 → txt 流
        "y_vec": y_vec,                   # CLIP 池化 → vec 流
        # 控制条件
        "maps": maps,
        "bbox": bbox,
        "cams": cams,
        "rel_pos": rel_pos,
        # 条件 drop mask
        "drop_cond_mask": drop_cond_mask,
        "drop_frame_mask": drop_frame_mask,
        # 多视图配置
        "mv_order_map": mv_order_map,
        # 时序 mask（可变长度视频）
        "x_mask": x_mask,
        # 首帧引导
        "first_frame_latent": first_frame_latent,
    }
    
    if device is not None and dtype is not None:
        # 只移动张量，跳过 dict/None
        tensor_keys = ["y", "y_vec", "maps", "cams", "rel_pos",
                       "drop_cond_mask", "drop_frame_mask"]
        for k in tensor_keys:
            if model_args[k] is not None:
                model_args[k] = model_args[k].to(device=device, dtype=dtype)
        if bbox is not None:
            for k, v in bbox.items():
                if isinstance(v, torch.Tensor):
                    bbox[k] = v.to(device=device, dtype=dtype)
        if x_mask is not None:
            model_args["x_mask"] = x_mask.to(device=device)
        if first_frame_latent is not None:
            model_args["first_frame_latent"] = first_frame_latent.to(device=device, dtype=dtype)
    
    return model_args


# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # =========================================================================
    # 1. 配置与运行时变量
    # =========================================================================
    cfg = parse_configs(training=True)
    
    # 调试模式
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

    # data config
    if cfg.num_frames is None:  # variable length dataset!
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
    else:  # single dataset!
        cfg.dataset, cfg.val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)

    # dtype
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"]
    dtype = to_torch_dtype(cfg_dtype)

    # NPU 兼容性
    if USE_NPU:
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # ── 分布式初始化 ──────────────────────────────────────────────────────────
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
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
    coordinator._local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
    device = get_current_device()

    # ── 实验目录 ──────────────────────────────────────────────────────────────
    if cfg.get("overfit", None) is not None:
        cfg.tag = f"{cfg.tag}_" if cfg.get("tag", "") != "" else ""
        cfg.tag += "overfit-" + str(cfg.overfit)
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    coordinator.block_all()
    if coordinator.is_node_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # ── logger / tensorboard ─────────────────────────────────────────────────
    log = reset_logger(exp_dir, enable_debug)
    log.info("Experiment directory created at %s", exp_dir)
    log.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    log.info(f"ColossalAI version: {colossalai.__version__}")
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
    if not hasattr(plugin, 'pg_mesh'):
        plugin.pg_mesh = None
        plugin.destroy_mesh_process_groups = lambda: None
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # =========================================================================
    # 2. 数据集与 DataLoader（与原版完全一致）
    # =========================================================================
    log.info("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    if cfg.get("overfit", None) is not None:
        _overfit_idxs = random.sample(range(len(dataset)), cfg.overfit)
        log.info(f"Overfit on: {_overfit_idxs}")
        overfit_idxs = []
        for _ in range(cfg.epochs):
            overfit_idxs += _overfit_idxs
            random.shuffle(_overfit_idxs)
        cfg.epochs = 1
        dataset = torch.utils.data.Subset(dataset, overfit_idxs)
    log.info("Dataset contains %s samples.", len(dataset))

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

    # 验证集
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
    log.info("Val Dataset contains %s samples.", len(val_dataset))

    val_dataloader_args = {**dataloader_args,
                           'shuffle': False,
                           'dataset': val_dataset,
                           'batch_size': cfg.val.get("batch_size", 1),
                           'num_workers': cfg.val.get("num_workers", 1)}
    val_dataloader, val_sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **val_dataloader_args,
    )

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # =========================================================================
    # 3. 模型构建（MagicDriveMMDiT 适配）
    # =========================================================================
    log.info("Building models...")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    # ── T5 文本编码器（与原版一致）────────────────────────────────────────────
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim        # 4096
        text_encoder_model_max_length = text_encoder.model_max_length  # 300
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # ── CLIP 文本编码器（新增，用于 MMDiT vec 流）────────────────────────────
    clip_encoder_path = cfg.get("clip_text_encoder", {}).get(
        "from_pretrained", None
    )
    if clip_encoder_path:
        clip_encoder = CLIPTextEncoderWrapper(clip_encoder_path, device, dtype)
        vec_in_dim = clip_encoder.output_dim
    else:
        # 没有 CLIP 时用零向量，vec_in_dim 从 config 读取
        clip_encoder = None
        vec_in_dim = cfg.get("model", {}).get("vec_in_dim", 768)
        log.warning("No CLIP text encoder configured. y_vec will be zeros (vec flow disabled).")

    # ── VAE ──────────────────────────────────────────────────────────────────
    vae = build_module2(cfg.get("vae", None), MODELS2, device_map=device, torch_dtype=dtype)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
    latent_size = (None, None, None)
    vae_out_channels = cfg.get("vae_out_channels", 16)

    # ── 扩散主模型（MagicDriveMMDiT）─────────────────────────────────────────
    #
    # config 中需要添加的字段（相对于原 MagicDriveSTDiT3 配置）：
    #   type: "MagicDriveMMDiT"
    #   vec_in_dim: 768          # CLIP 维度
    #   context_in_dim: 4096     # T5 维度
    #   hidden_size: 3072        # Flux large
    #   num_heads: 24
    #   depth: 19
    #   depth_single_blocks: 38
    #   axes_dim: [16, 56, 56]
    #   theta: 10000
    #   guidance_embed: true
    #   temporal_every_n_blocks: 1
    #   from_opensora2_pretrained: /path/to/opensora2.pt   ← 新增
    #   trainable_mode: "new_only"                          ← 新增
    #
    model_cfg = cfg.model.copy()
    trainable_mode = model_cfg.pop("trainable_mode", "new_only")

    model = (
        build_module(
            model_cfg,
            MODELS,
            # MMDiT 不需要这些（内部自行处理）
            # input_size / caption_channels / model_max_length 由 config 指定
        )
        .to(device, dtype)
        .train()
    )

    # 设置可训练参数
    model.set_trainable_parameters(trainable_mode)
    log.info(f"Trainable mode: {trainable_mode}")

    # 准备文本嵌入（bbox embedder 需要用 T5 初始化文本令牌）
    model.prepare_text_embedding(text_encoder)

    # 部分加载预训练权重（partial_load，对应原版逻辑）
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
        log.info(f"[partial load] Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
        log.info(f"[partial load] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")
        del weight, missing_keys, unexpected_keys

    model_numel, model_numel_trainable = get_model_numel(model)
    log.info(
        "[Diffusion] Trainable: %s | Frozen: %s | Total: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel - model_numel_trainable),
        format_numel_str(model_numel),
    )

    # ── EMA（修复 OOM：从 GPU deepcopy fp32 → CPU fp32 流式构建）────────────
    #
    # 原始写法的问题：
    #   ema = deepcopy(model).to(torch.float32).to(device)
    #   deepcopy 在 GPU 上先分配 bf16 副本，再转 fp32，三阶段峰值 ≈ 114 GB → OOM
    #
    # 修复：EMA 全程在 CPU fp32，通过 state_dict 流式拷贝，GPU 峰值 ≈ 0
    #   - 节省 GPU 显存：~57 GB（EMA fp32 完全不占 GPU）
    #   - update_ema_cpu：每步从 GPU 流式拷贝参数到 CPU 做 EMA 更新
    #   - model_sharding_ema / model_gathering_ema：空操作（CPU EMA 无需分布式 shard）
    #   - save(ema=ema)：完全兼容，save 内部直接调 ema.state_dict()
    ema = build_ema_on_cpu_v2(model)               # CPU fp32，GPU 占用 = 0
    ema_shape_dict = record_model_param_shape(ema)  # 兼容原版接口

    # ── Scheduler（RFlow，与原版一致）────────────────────────────────────────
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ── Optimizer ────────────────────────────────────────────────────────────
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
            optimizer, milestones_lr=milestones_lr, warmup_steps=warmup_steps
        )

    # 梯度检查点
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =========================================================================
    # 4. 分布式训练准备（ColossalAI booster）
    # =========================================================================
    log.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    # breakpoint()
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    log.info("Boosting model for distributed training")

    # ── 全局变量 ──────────────────────────────────────────────────────────────
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    drop_cond_ratio = cfg.get("drop_cond_ratio", 0.0)
    drop_cond_ratio_t = cfg.get("drop_cond_ratio_t", 0.4)
    drop_first_frame_ratio = cfg.get("drop_first_frame_ratio", 0.1)
    running_loss = 0.0
    log.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # ── 断点续训 ──────────────────────────────────────────────────────────────
    if cfg.get("load", None) is not None:
        log.info("Loading checkpoint")
        ret = load(
            booster, cfg.load,
            model=model, ema=ema, optimizer=optimizer,
            lr_scheduler=None if cfg.get("reset_lr", False) or cfg.get("start_from_scratch", False) else lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
            local_master=coordinator.is_node_master(),
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
            if cfg.get("reset_lr", False) and lr_scheduler:
                total_step = start_epoch * num_steps_per_epoch + start_step
                lr_scheduler.last_epoch = total_step
        log.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    if enable_debug:
        save_dir = save(booster, exp_dir, model=model, ema=ema, optimizer=optimizer,
                        lr_scheduler=lr_scheduler, sampler=sampler,
                        epoch=start_epoch, step=start_step,
                        global_step=start_epoch * num_steps_per_epoch + start_step,
                        batch_size=cfg.get("batch_size", None))
        log.info(f"Save your model to {save_dir} before training.")

    model_sharding_ema(ema)   # CPU EMA 无需 shard，空操作
    if cfg.get("validation_before_run", False):
        with RandomStateManager(verbose=True):
            coordinator.block_all()
            _run_validation_mmdit(
                cfg, text_encoder, clip_encoder, vae, model,
                device, dtype, val_dataloader, coordinator,
                start_epoch * num_steps_per_epoch + start_step,
                exp_dir,
            )
            val_sampler.reset()

    # =========================================================================
    # 5. 训练循环
    # =========================================================================
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    coordinator.block_all()

    # Timer 配置
    timers = {}
    timer_keys = ["move_data", "encode", "move_data2", "mask",
                  "diffusion", "backward", "update_ema", "reduce_loss", "misc"]
    for key in timer_keys:
        timers[key] = Timer(key, coordinator=None) if record_time else nullcontext()

    for epoch in range(start_epoch, cfg_epochs):
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        log.info("Beginning epoch %s...", epoch)

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                # ── 批次基本信息 ──────────────────────────────────────────────
                B, T, NC = batch["pixel_values"].shape[:3]
                logging.debug(f"bs={B}; T={T}; NC={NC}; shape={batch['pixel_values'].shape}")
                timer_list = []

                # ── Step A: 数据搬移 & 重排 ───────────────────────────────────
                with timers["move_data"] as t_move:
                    # 视频像素
                    x = batch.pop("pixel_values").to(device, dtype)
                    x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # (B*NC, C, T, H, W)

                    # 首帧
                    first_frame = batch.pop("first_frames").to(device, dtype)
                    first_frame = rearrange(first_frame, "B T NC C ... -> (B NC) C T ...")

                    # 文本（T5 输入）
                    y_texts = batch.pop("captions")[0]  # list of str, length B

                    # 地图
                    maps = batch.pop("bev_map_with_aux").to(device, dtype)  # (B, T, C, H, W)

                    # Bounding boxes
                    bbox = batch.pop("bboxes_3d_data")
                    bbox = [bbox_i.data for bbox_i in bbox]
                    bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
                    if bbox is not None:
                        bbox = add_box_latent(bbox, B, NC, T, model.module.sample_box_latent)
                        for k, v in bbox.items():
                            bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")

                    # 相机参数
                    cams = batch.pop("camera_param").to(device, dtype)   # (B, T, NC, 3, 7)
                    cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # (B*NC, T, 1, 3, 7)

                    # 相对位姿（帧间）
                    rel_pos = batch.pop("frame_emb").to(device, dtype)   # (B, T, 4, 4)
                    rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)

                if record_time:
                    timer_list.append(t_move)

                # ── Step B: 视觉 & 文本编码 ──────────────────────────────────
                with timers["encode"] as t_enc:
                    with torch.no_grad():
                        # VAE 编码（与原版一致，支持序列并行）
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                            first_frame_latent = first_frame.to(device, dtype)
                        else:
                            with RandomStateManager(verbose=verbose_mode):
                                # breakpoint()
                                x = sp_vae(x, vae.encode, get_sequence_parallel_group())
                                first_frame_latent = sp_vae(
                                    first_frame, vae.encode, get_sequence_parallel_group()
                                )

                        # T5 文本编码：获取隐层序列 y 和 attention mask
                        if cfg.get("load_text_features", False):
                            y_t5 = batch.pop("y").to(device, dtype)
                            text_mask = batch.pop("mask")
                            if isinstance(text_mask, torch.Tensor):
                                text_mask = text_mask.to(device)
                        else:
                            t5_ret = text_encoder.encode(y_texts)
                            # t5_ret 包含 'y': (B,1,L,D) 和 'mask': (B,L)
                            y_t5 = t5_ret["y"].squeeze(1).to(device, dtype)  # (B, L, 4096)
                            text_mask = t5_ret.get("mask", None)

                        # CLIP 文本编码：获取池化向量 y_vec (B, 768)
                        if clip_encoder is not None:
                            y_vec = clip_encoder.encode(y_texts, device=device)  # (B, 768)
                        else:
                            y_vec = torch.zeros(B, vec_in_dim, device=device, dtype=dtype)

                if record_time:
                    timer_list.append(t_enc)

                # ── Step C: 条件 drop mask 构建 ───────────────────────────────
                with timers["move_data2"] as t_move2:
                    # drop_cond_mask: shape (B,), 1=keep, 0=drop 所有条件
                    drop_cond_mask = torch.ones(B, device=device)
                    # drop_frame_mask: shape (B, T), 控制 bbox/rel_pos 的帧级别 drop
                    drop_frame_mask = torch.ones(B, T, device=device)

                    if drop_cond_ratio > 0:
                        for bs in range(B):
                            if random.random() < drop_cond_ratio:
                                # 完全无条件：drop 所有
                                drop_cond_mask[bs] = 0
                                drop_frame_mask[bs, :] = 0
                                if text_mask is not None:
                                    text_mask[bs] = 1  # 无条件时保留全部 token 位置
                                continue
                            # 部分帧 drop（保留首末帧）
                            t_ids = random.sample(
                                range(1, T - 1),
                                int(drop_cond_ratio_t * (T - 2))
                            )
                            drop_frame_mask[bs, t_ids] = 0

                    # 首帧引导的随机屏蔽
                    # x & first_frame_latent 此时都是 (B*NC, C, T, H, W)
                    # 需要转成 (B, C*NC, T, H, W) 供模型使用
                    x_model = rearrange(x, "(B NC) C T H W -> B (C NC) T H W", NC=NC)
                    ff_latent = rearrange(
                        first_frame_latent, "(B NC) C T H W -> B (C NC) T H W", NC=NC
                    )

                    # 随机屏蔽首帧（10%）
                    if random.random() < drop_first_frame_ratio:
                        ff_latent = torch.zeros_like(ff_latent)

                    # 构建 model_kwargs（MMDiT 接口）
                    model_args = build_model_kwargs_mmdit(
                        y_t5=y_t5,
                        y_vec=y_vec,
                        maps=maps,
                        bbox=bbox,
                        cams=cams,
                        rel_pos=rel_pos,
                        drop_cond_mask=drop_cond_mask,
                        drop_frame_mask=drop_frame_mask,
                        mv_order_map=cfg.get("mv_order_map"),
                        first_frame_latent=ff_latent,
                        device=device,
                        dtype=dtype,
                    )

                    # text mask（给 scheduler 使用，如果需要的话）
                    if text_mask is not None:
                        model_args["mask"] = text_mask

                    # x_mask：可变长度视频的帧遮盖 mask（与原版一致）
                    x_mask = None
                    if cfg.get("mask_ratios", None) is not None:
                        x_mask = mask_generator.get_masks(x_model)
                        model_args["x_mask"] = x_mask

                if record_time:
                    timer_list.append(t_move2)

                if verbose_mode:
                    log.info(f"Start model forward step! step={step}")

                # ── Step D: 扩散 loss 计算 ────────────────────────────────────
                # RFlowScheduler.training_losses 接口与原版一致：
                #   training_losses(model, x_start, model_kwargs, mask=x_mask)
                # 
                # 注意：x_model 形状是 (B, C*NC, T, H, W)
                # scheduler 内部调用 model(x_t, t, **model_kwargs)
                # MMDiT.forward(x, timesteps, **model_kwargs)
                with timers["diffusion"] as t_diff:
                    loss_dict = scheduler.training_losses(
                        model, x_model, model_args, mask=x_mask
                    )
                if record_time:
                    timer_list.append(t_diff)

                coordinator.block_all()

                # ── Step E: 反向传播 & 参数更新 ──────────────────────────────
                with timers["backward"] as t_back:
                    loss = loss_dict["loss"].mean()
                    booster.backward(loss=loss, optimizer=optimizer)

                    if verbose_mode:
                        log.info(f"Start model update step! step={step}")
                    optimizer.step()

                    if enable_debug:
                        for n, p in model.named_parameters():
                            if not (p == p).all():
                                log.info(f"Got nan on {n}")
                    optimizer.zero_grad()

                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(t_back)

                # ── Step F: EMA 更新（CPU fp32，流式拷贝）────────────────────
                with timers["update_ema"] as t_ema:
                    update_ema_cpu(ema, model.module, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(t_ema)

                # ── Step G: 日志 & 统计 ───────────────────────────────────────
                with timers["reduce_loss"] as t_reduce:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(t_reduce)

                if record_time:
                    misc_t = timers['misc'].__enter__()
                    timer_list.append(misc_t)

                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "step": str(step),
                        "global_step": str(global_step),
                        "lr": f"{lr:.2e}",
                    })
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    tb_writer.add_scalar("avg_loss", avg_loss, global_step)
                    tb_writer.add_scalar("lr", lr, global_step)
                    running_loss = 0.0
                    log_step = 0

                # ── Step H: Checkpoint 保存 ───────────────────────────────────
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering_ema(ema, ema_shape_dict)  # CPU EMA 无需 gather，空操作
                    save_dir = save(
                        booster, exp_dir,
                        model=model, ema=ema, optimizer=optimizer,
                        lr_scheduler=lr_scheduler, sampler=sampler,
                        epoch=epoch, step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if dist.get_rank() == 0:
                        model_sharding_ema(ema)               # CPU EMA 无需 shard，空操作
                    log.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch, step + 1, global_step + 1, save_dir,
                    )

                # ── Step I: Validation ────────────────────────────────────────
                report_every = cfg.get("report_every", 0)
                if report_every > 0 and (global_step + 1) % report_every == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    _run_validation_mmdit(
                        cfg, text_encoder, clip_encoder, vae, model,
                        device, dtype, val_dataloader, coordinator,
                        global_step + 1, exp_dir,
                    )
                    val_sampler.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                if record_time:
                    misc_t.__exit__(*sys.exc_info())
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for t in timer_list:
                        log_str += f"{t.name}: {t.elapsed_time:.3f}s | "
                    log_str += f"Total: {sum(t.elapsed_time for t in timer_list):.3f}s"
                    log.info(log_str)

                if enable_debug and step > 50:
                    break

        if enable_debug:
            break
        sampler.reset()
        start_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Validation 适配（MMDiT 接口）
# ─────────────────────────────────────────────────────────────────────────────

def _run_validation_mmdit(
    cfg, text_encoder, clip_encoder, vae, model,
    device, dtype, val_dataloader, coordinator,
    global_step, exp_dir,
):
    """
    封装验证逻辑，适配 MMDiT 的推理接口。
    
    与原版 run_validation 的差异：
      - 传入 clip_encoder，推理时生成 y_vec
      - 去掉 t_order_map（MMDiT 不需要）
      - 内部调用 inference 时使用 MMDiT 接口参数
    """
    # 注意：如果 run_validation 内部已经足够通用，可以直接复用。
    # 这里提供一个适配层，确保 mv_order_map 和推理参数正确传入。
    val_dir = run_validation(
        cfg.val,
        text_encoder,
        vae,
        model,
        device,
        dtype,
        val_dataloader,
        coordinator,
        global_step,
        exp_dir,
        cfg.mv_order_map,
        t_order_map=None,           # MMDiT 不需要
        extra_model_kwargs={
            # 推理时的额外参数，会被传入模型 forward
            "_clip_encoder": clip_encoder,
            "_vec_in_dim": cfg.get("model", {}).get("vec_in_dim", 768),
        },
    )
    return val_dir


if __name__ == "__main__":
    main()