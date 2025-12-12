import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project path
sys.path.append("/content/drive/MyDrive/deepfake-detection")

from utils.gradcam import GradCAM
from train import DeepfakeDetector


# -------------------------------
# SETTINGS
# -------------------------------
video_folder = "/content/drive/MyDrive/deepfake-detection/data/frames/fake/047_862"  
# <-- CHANGE THIS to any folder name you want

save_root = "/content/drive/MyDrive/deepfake-detection/results/gradcam"
os.makedirs(save_root, exist_ok=True)


# -------------------------------
# LOAD MODEL
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/deepfake-detection/models/best_model.pt"
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

cnn = model.cnn
target_layer = cnn.features[-1]  # last conv block
gradcam = GradCAM(cnn, target_layer)


# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# PROCESS ALL FRAMES
# -------------------------------
frames = sorted([
    f for f in os.listdir(video_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

video_name = os.path.basename(video_folder)
save_path = os.path.join(save_root, video_name)
os.makedirs(save_path, exist_ok=True)

print(f"Processing video: {video_name}")
print(f"Total frames: {len(frames)}")

for idx, frame_file in enumerate(frames):
    frame_path = os.path.join(video_folder, frame_file)
    img_bgr = cv2.imread(frame_path)

    if img_bgr is None:
        print(f"Skipping unreadable frame: {frame_file}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    tensor = transform(pil_img).unsqueeze(0).to(device)

    # Generate CAM
    cam = gradcam.generate(tensor)
    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

    # Save results
    heatmap_path = os.path.join(save_path, f"heatmap_{idx:04d}.jpg")
    overlay_path = os.path.join(save_path, f"overlay_{idx:04d}.jpg")

    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, overlay)

    print(f"Processed frame {idx+1}/{len(frames)}")

print("\nALL FRAMES PROCESSED!")
print(f"Saved results in: {save_path}")
