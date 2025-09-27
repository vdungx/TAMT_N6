import os

root = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\20bn-something-something-v2"
with open("datasets/SSV2/ssv2_unlabeled.txt", "w") as f:
    for name in os.listdir(root):
        if name.endswith(".webm"):
            f.write(os.path.join(root, name) + "\n")
