import sys
sys.path.append("/content/drive/MyDrive/deepfake-detection")


import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from utils.gradcam import GradCAM
from train import DeepfakeDetector   # import your model class


# -----------------------------------------------------
# MODIFY THIS: PATH TO YOUR BEST MODEL
# -----------------------------------------------------
model_path = "/content/drive/MyDrive/deepfake-detection/models/best_model.pt"

# -----------------------------------------------------
# MODIFY THIS: PICK ANY REAL FRAME FROM YOUR DATASET
# -----------------------------------------------------
sample_frame = sample_frame = "/content/drive/MyDrive/deepfake-detection/data/frames/fake/047_862/0001.jpg"
# change VIDEO_FOLDER to an actual folder name


# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepfakeDetector().to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# The CNN part inside your model
cnn_model = model.cnn

# Target layer for Grad-CAM: Last Conv Layer
target_layer = cnn_model.features[-1]   # MobileNetV2 last conv block


# -----------------------------------------------------
# INITIALIZE GRAD-CAM
# -----------------------------------------------------
gradcam = GradCAM(cnn_model, target_layer)


# -----------------------------------------------------
# PREPROCESS IMAGE
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img_bgr = cv2.imread(sample_frame)

if img_bgr is None:
    raise FileNotFoundError(f"Image not found at: {sample_frame}")


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

tensor = transform(pil_img).unsqueeze(0).to(device)


# -----------------------------------------------------
# GENERATE GRAD-CAM MAP
# -----------------------------------------------------
cam = gradcam.generate(tensor)

# Resize to original image size
cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

# Convert CAM to heatmap
heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)

# Overlay heatmap onto original image
overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)


# -----------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------
save_dir = "/content/drive/MyDrive/deepfake-detection/"
cv2.imwrite(save_dir + "gradcam_heatmap.jpg", heatmap)
cv2.imwrite(save_dir + "gradcam_overlay.jpg", overlay)

print("\nSaved:")
print(" - gradcam_heatmap.jpg")
print(" - gradcam_overlay.jpg")
