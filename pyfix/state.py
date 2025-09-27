import torch
ckpt = torch.load(r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\checkpoints\ssl_ssv2_vits\encoder_only_final.pth")
print(ckpt.keys())