#!/bin/bash
# RiskMV-DPO training entry point (Stage 6: LocalDPO fine-tuning)
#
# Usage:
#   bash train.sh
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash train.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/magicdrive/train/stage6_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_localdpo.py}"
NPROC="${NPROC:-1}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"

torchrun --standalone --nproc_per_node "${NPROC}" \
    "${ROOT}/scripts/train_localdpo.py" \
    "${CONFIG}" \
    --cfg-options num_workers=0 prefetch_factor=None sp_size=1 plugin='zero2'
