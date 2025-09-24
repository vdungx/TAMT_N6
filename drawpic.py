import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================
# 1. Load video và lấy frame
# ==========================
def load_frames(video_path, num_frames=8, resize=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return frames

# ==========================
# 2. Chuẩn bị model
# ==========================
# Ví dụ dùng ResNet18 pretrained
from torchvision.models import resnet18
model = resnet18(pretrained=True)
model.eval()

# Layer cuối cùng để visualize
target_layers = [model.layer4[-1]]

# Grad-CAM object
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

# Transform frame → tensor
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ==========================
# 3. Chạy Grad-CAM
# ==========================
video_path = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\HMDB\hmdb51_org\brush_hair\April_09_brush_hair_u_nm_np1_ba_goo_0.avi"   # đổi thành đường dẫn video của bạn
frames = load_frames(video_path, num_frames=6)

input_tensor = torch.stack([transform(Image.fromarray(f)) for f in frames])  # [T, C, H, W]
input_tensor = input_tensor.unsqueeze(0)  # [B, T, C, H, W] -> nhưng ResNet nhận [B,C,H,W], ta flatten theo batch
input_tensor = input_tensor.view(-1, 3, 224, 224)

# Tính cam
grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # [N,H,W]

# ==========================
# 4. Overlay lên ảnh gốc
# ==========================
results = []
for i, frame in enumerate(frames):
    img = frame.astype(np.float32) / 255.0
    cam_img = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
    results.append(cam_img)

# ==========================
# 5. Vẽ figure ngang
# ==========================
fig, axs = plt.subplots(1, len(results), figsize=(20, 5))
for i, r in enumerate(results):
    axs[i].imshow(r)
    axs[i].axis("off")
plt.suptitle("Feature Visualization with Grad-CAM", fontsize=16)
plt.show()
