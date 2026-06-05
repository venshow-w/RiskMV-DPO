import os
from contextlib import nullcontext
import sys
import numpy as np
import random
from copy import deepcopy
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
        # just before torch_npu, let xformers know there is no gpu
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import magicdrivedit.utils.module_contrib
from torch.utils.data import DataLoader, DistributedSampler
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
from magicdrivedit.acceleration.parallel_states import get_data_parallel_group, get_sequence_parallel_group, set_data_parallel_group
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


def main():
    # 1. 初始化分布式环境 (针对 2*H800)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 2. 加载配置与模型
    cfg = parse_configs(training=False)
    dtype = torch.bfloat16 # H800 推荐使用
    save_root = "/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/PublicDatasets/nuscenes/MagicDriveDiT-nuScenes-metadata/prep_latent_424-800"
    os.makedirs(save_root, exist_ok=True)
   
    # 加载编码器
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)
    # 3. 展开所有桶中的所有 Index
    def build_real_dataset(cfg):
        # 模拟 train_magicdrive.py 中的合并逻辑
        from magicdrivedit.utils.config_utils import merge_dataset_cfg
        
        if cfg.get("num_frames", None) is None:  # 多分辨率模式
            num_data_cfgs = len(cfg.data_cfg_names)
            datasets = []
            for idx, (res, data_cfg_name) in enumerate(cfg.data_cfg_names):
                # 获取对应的 override 配置
                overrides = cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)[idx]
                dataset, _ = merge_dataset_cfg(cfg, data_cfg_name, overrides)
                datasets.append((res, dataset))
            # 显式构造配置字典
            cfg.dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        else:  # 单一分辨率模式
            cfg.dataset, _ = merge_dataset_cfg(
                cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
                cfg.num_frames)
        
        return build_module(cfg.dataset, DATASETS)

    # 在 main 函数中使用：
    cfg = parse_configs(training=False)
    dataset = build_real_dataset(cfg)
    # dataset = build_module(cfg.dataset, DATASETS)
    all_str_indices = []
    # 这一步非常重要：模拟 Sampler 访问的所有可能路径
    buckets = dataset.as_buckets() 
    for bucket_id, idx_list in buckets.items():
        for idx in idx_list:
            all_str_indices.append(f"{idx}-{bucket_id}")

    # 4. 分布式分配任务
    # 使用 DistributedSampler 对字符串列表进行切分
    sampler = DistributedSampler(all_str_indices, shuffle=False)
    # 计算当前卡负责的子集
    indices_to_process = [all_str_indices[i] for i in list(sampler)]

    print(f"Rank {local_rank} processing {len(indices_to_process)} samples...")

    # 5. 提取循环
    with torch.no_grad():
        for str_idx in tqdm(indices_to_process, disable=(local_rank != 0)):
            # 检查是否已存在，支持断点续传
            save_path = os.path.join(save_root, f"{str_idx}.pth")
            if os.path.exists(save_path):
                continue

            try:
                # 获取原始数据
                # NuScenesMultiResDataset[str_idx] 会根据字符串解析 h, w, fps, t
                item = dataset[str_idx] 
                
                # --- 视频编码 (VAE) ---
                x = item.pop("pixel_values").to(device, dtype).unsqueeze(0) # [1, T, NC, C, H, W]
                B, T, NC, C, H, W = x.shape
                x = x.view(B * NC, C, T, H, W)
                latents = vae.encode(x) # 结果已包含多卡/多分辨率信息

                # --- 文本编码 (T5) ---
                captions = item.pop("captions") # 已经是 list
                text_ret = text_encoder.encode(captions)
                
                # --- 合并保存 ---
                # 包含训练所需的全部 tensor
                save_data = {
                    "latents": latents ,#.cpu(),
                    "text_hidden_states": text_ret['y'],  #.cpu(),
                    "text_mask": text_ret['mask'],#.cpu(),
                    "maps": item.pop("bev_map_with_aux"),#.cpu(),
                    "cams": item.pop("camera_param"),#.cpu(),
                    "bbox": item.pop("bboxes_3d_data"),#.cpu(), # 需要确保这是 tensor
                    "frame_emb": item.pop("frame_emb"),#.cpu(),
                    "meta": {k: v for k, v in item.items() if isinstance(v, (int, float, str, torch.Tensor))}
                }
                torch.save(save_data, save_path)
            except Exception as e:
                print(f"Error processing {str_idx}: {e}")

    dist.destroy_process_group()

# 运行命令:
# torchrun --nproc_per_node=2 extract_distributed.py
if __name__ == "__main__":
    main()