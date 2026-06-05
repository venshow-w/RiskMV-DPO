# Experiments (Not on Main Release Path)

Legacy and ablation scripts kept for internal reproduction. **Use `train.sh` / `test.sh` at repo root for the CVPR release pipeline.**

## baseline/

| Script | Purpose |
|--------|---------|
| `train_magicdrive.py` | MagicDrive stage 1–3 baseline training |
| `train_magicdrive_with_dggt.py` | DGGT/VGGT fusion ablation (stage 4–5) |
| `train_magicdrive_from_latent.py` | Latent-cache training variant |
| `prep_latent.py` | Precompute VAE/T5 latents |

Launch baseline stage-3: `bash scripts/launch_1node.sh`

## magicsora/

OpenSora-based MagicSora experiments (not required for RiskMV-DPO).
