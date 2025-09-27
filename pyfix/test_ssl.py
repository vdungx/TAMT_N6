import random

infile = "datasets/SSV2/ssv2_unlabeled.txt"
outfile = "datasets/SSV2/ssv2_unlabeled_small.txt"

with open(infile) as f:
    lines = [l.strip() for l in f if l.strip()]

subset = random.sample(lines, 5000)  # chọn ngẫu nhiên 50 video

with open(outfile, "w") as f:
    f.write("\n".join(subset))

print(f"Lưu {len(subset)} video vào {outfile}")
