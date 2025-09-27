import torch

ckpt = torch.load(r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\Model\Pre_trainedModel\112112vit-s-140epoch.pt", map_location="cpu")
new_ckpt = {"epoch": ckpt.get("epoch", 0), "state": ckpt["module"]}
torch.save(new_ckpt, "converted_checkpoint.tar")