#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/magicdrive/train/stage3_448x840-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py}"
NPROC="${NPROC:-8}"

torchrun --standalone --nproc_per_node "${NPROC}" \
    "${ROOT}/experiments/baseline/train_magicdrive.py" \
    "${CONFIG}"
