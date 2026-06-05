# DesDriveWorld / RiskMV-DPO

Official PyTorch implementation of **Risk-Controllable Multi-View Diffusion for Driving Scenario Generation** (CVPR 2026).

This repository releases **RiskMV-DPO** — a parameter-efficient fine-tuning framework that improves multi-view driving video generation with **region-aware LocalDPO** and **geometry-guided adaptive modulation (GAM)**, built on top of [MagicDrive-V2](https://github.com/flymin/MagicDrive-V2).

---

## Highlights

| Component | Description |
|-----------|-------------|
| **Motion-Aware Masking** | Local corruption targets dynamic regions (estimated via frame differencing) instead of random patches, yielding semantically meaningful preference pairs. |
| **Multi-View LocalDPO** | Extends LocalDPO to 6-camera nuScenes videos with region-weighted DPO loss and frozen EMA reference policy. |
| **GAM (Geometry-Guided Adaptive Modulation)** | Injects VGGT geometry latents via zero-init scale/shift gates — no RGB feature overwrite, first-frame protected. |
| **VGGT Geometry Adapter** | Compresses frozen VGGT tokens into compact geometry latents for alignment supervision. |
| **Efficient Fine-Tuning** | Only GAM + geometry adapter + DPO heads are trained; backbone stays frozen after stage-3 pretraining. |

---

## Method Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RiskMV-DPO Training Loop                      │
├─────────────────────────────────────────────────────────────────┤
│  Real 6-view video                                               │
│       │                                                          │
│       ▼                                                          │
│  MotionAwareMaskGenerator ──► local mask M (dynamic regions)    │
│       │                                                          │
│       ▼                                                          │
│  MultiViewLocalCorrupter ──► corrupted video x̃ (RFlow noise)    │
│       │                    └──► restored x̂ (frozen model denoise)│
│       ▼                                                          │
│  MagicDriveSTDiT3-localdpo ──► prediction + VGGT geometry       │
│       │         (GAM modulation at selected layers)              │
│       ▼                                                          │
│  MultiViewRegionAwareDPOLoss                                     │
│    L = λ_ra·L_DPO + λ_sft·L_SFT + λ_align·L_geo               │
└─────────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module-level details and call graphs.

---

## Repository Structure

```
RiskMV-DPO/
├── train.sh / test.sh          # Main entry points
├── scripts/
│   ├── train_localdpo.py       # Stage-6 LocalDPO fine-tuning
│   └── inference.py            # Multi-view video generation
├── localdpo/                   # Core contribution modules
│   ├── motion_aware_mask.py    # Motion-aware local mask
│   ├── corrupter.py            # Multi-view corrupt-and-restore
│   ├── localdpo_loss.py        # Region-aware DPO + alignment loss
│   └── vggt_scorer.py          # VGGT geometry adapter
├── magicdrivedit/              # MagicDrive backbone (STDiT3 + VAE + data)
│   └── models/magicdrive/
│       └── magicdrive_stdit3_localdpo.py  # GAM + geometry injection
├── dggt/                       # VGGT geometry backbone (vendored)
├── configs/
│   ├── magicdrive/train/       # Training configs (stage6 = LocalDPO)
│   └── magicdrive/inference/   # Inference configs
├── tools/prepare_data/         # nuScenes preprocessing
├── experiments/                # Legacy / ablation scripts (not main path)
└── requirements/
```

---

## Installation

```bash
conda create -n riskmv-dpo python=3.10 -y
conda activate riskmv-dpo

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/requirements.txt

# Flash attention (recommended)
pip install flash-attn --no-build-isolation

# ColossalAI for distributed training
pip install colossalai>=0.4.3
```

Download pretrained weights and place them under `pretrained/`:

| Model | Path |
|-------|------|
| MagicDrive stage-3 checkpoint | `pretrained/magicdrive/MagicDriveDiT-stage3-40k-ft/` |
| CogVideoX VAE | `pretrained/CogVideoX-2b/vae/` |
| T5-XXL | `pretrained/google/t5-v1_1-xxl/` |
| VGGT (DGGT) | `pretrained/dggt/model_latest_waymo.pt` |

Prepare nuScenes following [MagicDrive-V2 data guide](https://github.com/flymin/MagicDrive-V2) and update paths in `configs/magicdrive/train/stage6_*.py`.

---

## Training

**Stage 6 — LocalDPO fine-tuning** (main method):

```bash
# Single GPU
bash train.sh

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash train.sh

# Custom config / overrides
CONFIG=configs/magicdrive/train/stage6_...localdpo.py \
  bash train.sh
```

Key config: `configs/magicdrive/train/stage6_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_localdpo.py`

Override dataset and checkpoint paths:

```bash
torchrun --standalone --nproc_per_node 1 scripts/train_localdpo.py \
  configs/magicdrive/train/stage6_...localdpo.py \
  --cfg-options \
    partial_load=/path/to/stage3_ckpt \
    dataset.data.train.ann_file=/path/to/train.pkl
```

---

## Inference

```bash
CHECKPOINT=/path/to/localdpo_ckpt bash test.sh

# Or directly:
torchrun --standalone --nproc_per_node 1 scripts/inference.py \
  configs/magicdrive/inference/localdpo_fullx424x800_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
  --cfg-options model.from_pretrained=/path/to/ckpt num_frames=9
```

Outputs are saved under `outputs/inference/localdpo/<timestamp>/`.

---

## Key Hyperparameters (Stage 6)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lambda_ra` | 1.0 | Region-aware DPO loss weight |
| `lambda_sft` | 0.1 | Supervised fine-tuning weight |
| `lambda_align` | 0.01 | VGGT geometry alignment weight |
| `motion_threshold` | 0.7 | Optical-flow threshold for dynamic regions |
| `mask_size_range` | (0.1, 0.25) | Local mask area ratio |
| `lr` | 2e-5 | Learning rate (new modules only) |
| `num_geo_tokens` | 16 | Geometry latent token count |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{desdriveworld2026,
  title     = {DesDriveWorld: A Description-Enhanced World Model with 3D Structural Consistency for Autonomous Driving},
  author    = {TODO},
  booktitle = {CVPR},
  year      = {2026}
}
```

---

## Acknowledgements

This project builds upon [MagicDrive-V2](https://github.com/flymin/MagicDrive-V2), [LocalDPO](https://github.com/localdpo/localdpo), [CogVideoX](https://github.com/THUDM/CogVideo), [ColossalAI](https://github.com/hpcaitech/ColossalAI), and VGGT/DGGT. We thank the authors for open-sourcing their code.

---

## License

This repository is released under the Apache 2.0 License. Third-party components (`dggt/`, `magicdrivedit/`) retain their original licenses.
