# CONFIG=configs/magicdrive/magicsora/stage1_448x840-12Hz-mmdit_opensora2_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py

# # CUDA_VISIBLE_DEVICES=0 \
# # torchrun --standalone --nproc_per_node 1 train_magicsora2.py \
# #       ${CONFIG} --cfg-options num_workers=0 prefetch_factor=None sp_size=1 plugin='zero2'

#  torchrun --standalone --nproc_per_node 2 train_magicsora2.py \
#      ${CONFIG} --cfg-options num_workers=2 prefetch_factor=2 sp_size=2

CONFIG=configs/magicdrive/magicsora/stage1_448X840-12Hz-stdit3_opensora1.3_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py

CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node 1 train_magicsora1_3.py \
      ${CONFIG} --cfg-options num_workers=0 prefetch_factor=None sp_size=1 plugin='zero2'

#  torchrun --standalone --nproc_per_node 2 train_magicsora1_3.py \
#      ${CONFIG} --cfg-options num_workers=2 prefetch_factor=2 sp_size=2