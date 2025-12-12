import os
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------------------
# FIX: Add project root to path
# ----------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

# ----------------------------------------------------------------------
# IMPORT UTILS
# ----------------------------------------------------------------------
from utils.blink import extract_blink_timeline_from_folder, detect_blinks
from utils.micro import extract_landmark_motion_from_folder
from utils.fusion import fuse_scores
from utils.gradcam import GradCAM
from models.cnn_lstm import CNNLSTMDetector


# ----------------------------------------------------------------------
# Extract Frames
# ----------------------------------------------------------------------
def extract_frames(video_path, output_folder="temp_frames", max_frames=150):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    saved = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break

        fp = os.path.join(output_folder, f"{saved:04d}.jpg")
        cv2.imwrite(fp, frame)
        saved += 1

    cap.release()
    return output_folder


# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
def load_model(weight_path, device):
    model = CNNLSTMDetector(pretrained=False)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Safe GradCAM for single frame
# ----------------------------------------------------------------------
def generate_gradcam_for_frame(model, gradcam, image_path, device, transform):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    # Forward only CNN
    feats = model.cnn(tensor)
    pooled = F.adaptive_avg_pool2d(feats, 1).squeeze()
    fake_score = pooled.mean()

    cam = gradcam.generate(fake_score)
    cam = np.array(cam, dtype=np.float32)

    # collapse channels if needed
    if cam.ndim == 3:
        cam = np.mean(cam, axis=-1)

    # normalize
    cam = np.nan_to_num(cam)
    cam_min, cam_max = cam.min(), cam.max()
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    cam_resized = cv2.resize(cam_norm, (img.shape[1], img.shape[0]))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return cam_resized, overlay


# ----------------------------------------------------------------------
# Main Analysis Pipeline
# ----------------------------------------------------------------------
def analyze_full_video(video_path, model, device):

    # 1. Extract frames
    frame_dir = extract_frames(video_path, "temp_frames")

    # 2. BLINK
    EAR_list, valid_ear = extract_blink_timeline_from_folder(frame_dir)
    blink_count, blink_frames = detect_blinks(EAR_list)

    # 3. MICRO EXPRESSION
    motion_curve, valid_motion = extract_landmark_motion_from_folder(frame_dir)
    diffs = np.diff([x for x in motion_curve if x is not None])
    micro_score = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0

    # 4. MODEL PREDICTION
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    files = sorted(os.listdir(frame_dir))[:30]
    frames = []

    for f in files:
        img = Image.open(os.path.join(frame_dir, f)).convert("RGB")
        frames.append(transform(img))

    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)
    logits = model(frames_tensor)
    model_score = torch.sigmoid(logits).item()

    # 5. FIND LAST CONV LAYER
    import torch.nn as nn

    def find_last_conv(module):
        last = None
        for child in module.children():
            found = find_last_conv(child)
            if found is not None:
                last = found
            if isinstance(child, nn.Conv2d):
                last = child
        return last

    last_conv = find_last_conv(model.cnn)
    if last_conv is None:
        raise RuntimeError("No Conv2D layer found in model.cnn for GradCAM!")

    gradcam = GradCAM(model, last_conv)

    # pick 3 frames for visualization
    idxs = np.linspace(0, len(files) - 1, 3, dtype=int)
    gradcam_maps = []
    overlay_files = []

    os.makedirs("results", exist_ok=True)

    for i in idxs:
        path = os.path.join(frame_dir, files[i])
        cam_map, overlay = generate_gradcam_for_frame(model, gradcam, path, device, transform)

        gradcam_maps.append(cam_map)

        out_path = f"results/overlay_{i}.jpg"
        cv2.imwrite(out_path, overlay)
        overlay_files.append(out_path)

    # 6. Fuse signals
    final_score, final_label, explanation_text = fuse_scores(
        model_score,
        blink_count,
        micro_score,
        gradcam_maps
    )

    # ------------------------------------------------------------------
    # RETURN DICTIONARY (fixes Gradio errors)
    # ------------------------------------------------------------------
    return (
        final_label,
        final_score,
        {
            "explanation": explanation_text,
            "gradcam_overlay": overlay_files,      # <-- list of valid file paths
            "blink_count": blink_count,
            "micro_score": micro_score
        }
    )


# ----------------------------------------------------------------------
# CLI ENTRY
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--weights", type=str, default="models/cnn_lstm_detector.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = load_model(args.weights, device)

    print("Analyzing video...")
    label, score, explanation = analyze_full_video(args.video, model, device)

    print("\n===== FINAL RESULT =====")
    print("LABEL:", label)
    print("CONFIDENCE:", round(score, 3))
    print(explanation)
