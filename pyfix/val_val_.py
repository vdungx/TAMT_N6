import json
import re

def normalize_template(template: str) -> str:
    """
    Chuẩn hóa template từ validation.json để khớp với labels.json.
    - Bỏ hết dấu [] nhưng giữ nguyên nội dung bên trong.
    """
    return re.sub(r"\[|\]", "", template).strip()

# ---- input files ----
val_json    = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\labels\validation.json"  # file validation gốc
labels_json = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\labels\labels.json"      # mapping template -> class id
output_json = "val.json"   # file output tác giả cần

# ---- load labels map ----
with open(labels_json, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Load validation.json
with open(val_json, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# Convert
val_out = []
miss = 0
for item in val_data:
    template = normalize_template(item["template"])
    if template not in label_map:
        print("⚠️ Không tìm thấy:", item["template"], "->", template)
        miss += 1
        continue
    label_id = int(label_map[template])
    val_out.append({
        "id": item["id"] + ".webm",
        "label": label_id
    })

# Save
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(val_out, f, indent=2)

print(f"✅ Saved {len(val_out)} samples to {output_json}, missed {miss}")
