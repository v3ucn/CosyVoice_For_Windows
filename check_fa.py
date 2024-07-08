import torch
print("是否可以开启flash-attn",torch.backends.cuda.flash_sdp_enabled())