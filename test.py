import torch
from flash_attn import flash_attn_func
print(torch.cuda.get_device_name()) # 确认是 H800/A100