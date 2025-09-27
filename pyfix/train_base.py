import json
import os
import re

def normalize_template(template: str) -> str:
    """
    Chuẩn hóa template từ train.json để khớp với labels.json.
    - Bỏ hết dấu [] nhưng giữ nguyên nội dung bên trong.
    - Ví dụ:
        "[part] of [something]" -> "part of something"
        "Putting [something similar to other things]" -> "Putting something similar to other things"
    """
    # Bỏ dấu [ ] nhưng giữ nội dung bên trong
    template = re.sub(r"\[|\]", "", template)
    return template.strip()

# ---- input files ----
train_json = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\labels\train.json"      # file bạn có
labels_json = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\labels\labels.json"    # file mapping template -> class id
output_json   = "base.json"       # file output tác giả cần

# ---- load labels map ----
with open(labels_json, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Load train.json
with open(train_json, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# Convert
base_data = []
for item in train_data:
    template = normalize_template(item["template"])
    if template not in label_map:
        print("⚠️ Không tìm thấy:", item["template"], "->", template)
        continue
    label_id = int(label_map[template])  # lấy id từ labels.json
    label_id = int(label_map[template])  # lấy id từ labels.json
    base_data.append({
        "id": item["id"] + ".webm",
        "label": label_id
    })

# Save
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(base_data, f, indent=2)

print(f"✅ Saved {len(base_data)} samples to {output_json}")