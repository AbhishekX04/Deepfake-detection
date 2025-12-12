import cv2
import os
import sys
import numpy as np

# Add project root to path
sys.path.append("/content/drive/MyDrive/deepfake-detection")

# -------------------------------
# SETTINGS
# -------------------------------
video_name = "047_862"

overlay_frames_path = f"/content/drive/MyDrive/deepfake-detection/results/gradcam/{video_name}"
original_frames_path = f"/content/drive/MyDrive/deepfake-detection/data/frames/fake/{video_name}"
save_video_path = f"/content/drive/MyDrive/deepfake-detection/results/gradcam/{video_name}/overlay_video.mp4"
save_side_by_side_path = f"/content/drive/MyDrive/deepfake-detection/results/gradcam/{video_name}/side_by_side_video.mp4"

# -------------------------------
# LOAD FRAME LIST
# -------------------------------
overlay_frames = sorted([
    f for f in os.listdir(overlay_frames_path)
    if f.startswith("overlay_") and f.endswith(".jpg")
])

if len(overlay_frames) == 0:
    raise ValueError("No overlay frames found! Run gradcam_video.py first.")

print(f"Found {len(overlay_frames)} overlay frames.")

# Get size from first frame
first_frame_path = os.path.join(overlay_frames_path, overlay_frames[0])
frame = cv2.imread(first_frame_path)
height, width, _ = frame.shape

fps = 25  # Default video FPS (can modify if needed)

# -------------------------------
# CREATE VIDEO WRITERS
# -------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

overlay_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
side_by_side_writer = cv2.VideoWriter(save_side_by_side_path, fourcc, fps, (width*2, height))

# -------------------------------
# BUILD VIDEO FRAME BY FRAME
# -------------------------------
for fname in overlay_frames:
    overlay_img = cv2.imread(os.path.join(overlay_frames_path, fname))

    # Extract original frame number
    frame_id = fname.replace("overlay_", "").replace(".jpg", "")
    orig_frame_path = os.path.join(original_frames_path, f"{frame_id}.jpg")

    # Some datasets use .png — fallback check
    if not os.path.exists(orig_frame_path):
        orig_frame_path = orig_frame_path.replace(".jpg", ".png")

    original_img = cv2.imread(orig_frame_path)

    if original_img is None:
        print(f"WARNING: Could not read original frame {orig_frame_path}. Skipping.")
        continue

    # Write overlay video
    overlay_writer.write(overlay_img)

    # Create side-by-side
    combined = np.hstack((original_img, overlay_img))
    side_by_side_writer.write(combined)

    print(f"Processed frame: {fname}")

# Release writers
overlay_writer.release()
side_by_side_writer.release()

print("\nVideos saved:")
print(f" - Overlay Only:        {save_video_path}")
print(f" - Side-by-Side Video:  {save_side_by_side_path}")
