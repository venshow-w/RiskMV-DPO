#!/bin/bash

# 1. 设置显存优化环境变量，防止 H800 在高分辨率推理时显存碎片化
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 执行分布式推理命令
# 使用 2 个 GPU (nproc_per_node 2) 运行
#checkpoint = /media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/magicdrive/MagicDriveDiT-stage3-40k-ft \
# checkpoint = /mnt/projects/MagicDrive-V2/outputs/MagicDriveSTDiT3-XL-2_stage4_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_gen3c_20251230-0753/epoch0-global_step2000
torchrun --standalone --nproc_per_node 1 \
    inference_pipeline.py \
    configs/magicdrive/inference/test2_fullx424x800_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
    --cfg-options \
    vae_tiling=384 \
    model.from_pretrained='/media/omnisky/12dd907f-8a2c-4a49-954c-a33edc979c06/pretrained/magicdrive/MagicDriveDiT-stage3-40k-ft', 
    num_frames=9 \
    num_workers=0 \
    model.enable_flash_attn=True \
    model.enable_xformers=False \
    global_flash_attn=True \
    global_xformers=False
    