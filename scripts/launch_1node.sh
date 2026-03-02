#!/bin/bash

GPU=2
CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage4_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_gen3c.py
#ARGS=--cfg-options debug=true
# ARGS=${@:3}

torchrun --standalone --nproc_per_node ${GPU} train_magicdrive.py \
     ${CONFIG} #${ARGS}







