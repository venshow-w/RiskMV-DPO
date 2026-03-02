# CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage3_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py
# CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage3_424x800-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py
CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage3_448x840-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py

# CUDA_VISIBLE_DEVICES=1 \
# torchrun --standalone --nproc_per_node 1 train_magicdrive.py \
#       ${CONFIG} --cfg-options num_workers=0 prefetch_factor=None sp_size=1 plugin='zero2'

#  torchrun --standalone --nproc_per_node 2 train_magicdrive.py \
#      ${CONFIG} --cfg-options num_workers=2 prefetch_factor=2

# CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage5_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_gen3c_depth.py
# CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage4_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_only_dggtproj_dggtfusion.py
# CONFIG=/mnst/projects/MagicDrive-V2/configs/magicdrive/train/stage5_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_gen3c_dggtfusion.py
# torchrun --standalone --nproc_per_node 1 train_magicdrive_from_latent.py \
#     ${CONFIG} --cfg-options num_workers=0 prefetch_factor=None
# CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage5_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_gen3c_dggtfusion.py

CONFIG=/mnt/projects/MagicDrive-V2/configs/magicdrive/train/stage6_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8_localdpo.py


# torchrun --standalone --nproc_per_node 2 train_magicdrive_with_dggt.py \
#      ${CONFIG} --cfg-options num_workers=2 prefetch_factor=2

# CUDA_VISIBLE_DEVICES=0 \
# torchrun --standalone --nproc_per_node 1 train_magicdrive_with_dggt.py \
#       ${CONFIG} --cfg-options num_workers=0 prefetch_factor=None sp_size=1 plugin='zero2'


CUDA_VISIBLE_DEVICES=1 \
torchrun --standalone --nproc_per_node 1 train_magicdrive_localdpo.py \
      ${CONFIG} --cfg-options num_workers=0 prefetch_factor=None sp_size=1 plugin='zero2'


# torchrun --standalone --nproc_per_node 2 train_magicdrive_localdpo.py \
#      ${CONFIG} --cfg-options num_workers=2 prefetch_factor=2
