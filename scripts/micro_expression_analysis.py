import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append("/content/drive/MyDrive/deepfake-detection")

# -----------------------------
# SETTINGS
# -----------------------------
video_folder = "/content/drive/MyDrive/deepfake-detection/data/frames/fake/047_862"
save_plot = "/content/drive/MyDrive/deepfake-detection/results/micro_expression_motion_047_862.png"
save_report = "/content/drive/MyDrive/deepfake-detection/results/micro_expression_report_047_862.txt"

os.makedirs("/content/drive/MyDrive/deepfake-detection/results", exist_ok=True)

# -----------------------------
# LOAD FRAMES
# -----------------------------
frames = sorted([
    f for f in os.listdir(video_folder)
    if f.lower().endswith((".jpg", ".png"))
])

if len(frames) < 2:
    raise ValueError("Need at least 2 frames for motion analysis.")

motion_values = []

# -----------------------------
# PROCESS EACH FRAME PAIR
# -----------------------------
prev_frame = None

for fname in frames:
    img_path = os.path.join(video_folder, fname)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_energy = np.sum(diff)  # total motion intensity
        motion_values.append(motion_energy)

    prev_frame = gray


# -----------------------------
# PLOT MOTION GRAPH
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(motion_values, label="Facial Motion Energy", color="blue")
plt.xlabel("Frame Index")
plt.ylabel("Motion Intensity")
plt.title("Micro-Expression Motion Pattern Analysis")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(save_plot)
plt.close()

# -----------------------------
# SAVE REPORT
# -----------------------------
mean_motion = float(np.mean(motion_values))
std_motion = float(np.std(motion_values))
max_motion = float(np.max(motion_values))
min_motion = float(np.min(motion_values))

with open(save_report, "w") as f:
    f.write("Micro-Expression Motion Analysis Report\n")
    f.write("--------------------------------------\n")
    f.write(f"Frames analyzed: {len(frames)}\n")
    f.write(f"Mean Motion: {mean_motion:.2f}\n")
    f.write(f"Std Dev: {std_motion:.2f}\n")
    f.write(f"Max Motion Peak: {max_motion:.2f}\n")
    f.write(f"Min Motion: {min_motion:.2f}\n")

print("\nMicro-Expression Motion Analysis Completed!")
print("Saved:")
print(" - Motion Graph:", save_plot)
print(" - Report:", save_report)
