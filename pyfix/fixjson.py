import os
import json

f = open(r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\filelist\SSv2Full\val_few_shot.txt")
names_list = []
labels_list = []

for line in f:
    iamge_labels, iamge_names = line.split('/')
    iamge_labels = int(iamge_labels[3:])  # VD: class00001 → 1
    iamge_names = iamge_names.strip()

    # đường dẫn frames
    frame_dir = f"C:/Users/dungs/Documents/Study/ManageBigData/TAMT-main/datasets/SSV2/frames/{iamge_names}"

    # ✅ kiểm tra thư mục frame có tồn tại chưa
    if not os.path.isdir(frame_dir):
        print(f"⚠️ Bỏ qua {iamge_names}, chưa có frames.")
        continue

    # ✅ chỉ thêm nếu có đủ frames
    for k in range(1, 6):  # lấy 5 frame mỗi video
        frame_path = os.path.join(frame_dir, f"frame_{k:02d}.jpg")
        if os.path.exists(frame_path):
            names_list.append(frame_path)
            labels_list.append(iamge_labels)
        else:
            print(f"⚠️ Thiếu {frame_path}, bỏ qua.")

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('val.json', 'w') as F:
    json.dump(dic, F)

print("✅ Done! Saved base.json")
