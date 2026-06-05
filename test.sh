#!/bin/bash
# RiskMV-DPO inference entry point
#
# Usage:
#   bash test.sh
#   CHECKPOINT=/path/to/ckpt bash test.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/magicdrive/inference/localdpo_fullx424x800_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py}"
CHECKPOINT="${CHECKPOINT:-/path/to/your/localdpo_checkpoint}"
NPROC="${NPROC:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node "${NPROC}" \
    "${ROOT}/scripts/inference.py" \
    "${CONFIG}" \
    --cfg-options \
    vae_tiling=384 \
    model.from_pretrained="${CHECKPOINT}" \
    num_frames=9 \
    num_workers=0 \
    model.enable_flash_attn=True \
    model.enable_xformers=False \
    global_flash_attn=True \
    global_xformers=False
