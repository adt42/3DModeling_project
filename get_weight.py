#This little code is useful to modify directly the model's weight
#It is not really relevant in general case but very cool for this specific model

import torch


pth_file = r'C:\Users\benja\mast3r\checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

checkpoint = torch.load(pth_file, map_location='cpu')

print(checkpoint.keys())

state_dict = checkpoint['model']

qkv_weight = state_dict['enc_blocks.0.attn.qkv.weight']

state_dict['enc_blocks.0.attn.qkv.weight'] = qkv_weight * 0.001

torch.save(checkpoint, r'C:\Users\benja\mast3r\checkpoints\modified_mast3r.pth')

 
