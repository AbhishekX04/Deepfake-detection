import cv2
import mediapipe as mp
import numpy as np
import os
import sys

sys.path.append("/content/drive/MyDrive/deepfake-detection")

# -------------------------------------
# SETTINGS
# -------------------------------------
video_folder = "/content/drive/MyDrive/deepfake-detection/data/frames/fake/047_862"
save_report_path = "/content/drive/MyDrive/deepfake-detection/results/blink_report_047_862.txt"

# -------------------------------------
# EAR CALCULATION
# -------------------------------------
def eye_aspect_ratio(pts):
    # pts: 6 landmark points around eye (2D)
    # EAR formula
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear


# -------------------------------------
# INIT MEDIAPIPE
# -------------------------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]

EAR_THRESHOLD = 0.21    # below = eye is closed
EAR_CONSEC_FRAMES = 2   # frames to confirm a blink

# -------------------------------------
# PROCESS ALL FRAMES
# -------------------------------------
frames = sorted([
    f for f in os.listdir(video_folder)
    if f.lower().endswith((".jpg", ".png"))
])

blink_count = 0
closed_frames = 0
ear_values = []

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:

    for fname in frames:
        img_path = os.path.join(video_folder, fname)
        image = cv2.imread(img_path)

        if image is None:
            print("Skipping:", fname)
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        face = results.multi_face_landmarks[0]

        h, w, _ = image.shape

        # extract eye landmark coordinates
        left_eye_pts = []
        right_eye_pts = []

        for idx in LEFT_EYE_LANDMARKS:
            lm = face.landmark[idx]
            left_eye_pts.append([lm.x * w, lm.y * h])

        for idx in RIGHT_EYE_LANDMARKS:
            lm = face.landmark[idx]
            right_eye_pts.append([lm.x * w, lm.y * h])

        left_eye_pts = np.array(left_eye_pts)
        right_eye_pts = np.array(right_eye_pts)

        ear_left = eye_aspect_ratio(left_eye_pts)
        ear_right = eye_aspect_ratio(right_eye_pts)

        ear = (ear_left + ear_right) / 2.0
        ear_values.append(ear)

        # blink detection
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if closed_frames >= EAR_CONSEC_FRAMES:
                blink_count += 1
            closed_frames = 0


# -------------------------------------
# SAVE REPORT
# -------------------------------------
avg_ear = np.mean(ear_values)

with open(save_report_path, "w") as f:
    f.write("Blink Analysis Report\n")
    f.write("---------------------\n")
    f.write(f"Frames analyzed: {len(frames)}\n")
    f.write(f"Blink count: {blink_count}\n")
    f.write(f"Average EAR: {avg_ear:.3f}\n")
    f.write(f"EAR threshold: {EAR_THRESHOLD}\n")

print("\nBlink Analysis Completed!")
print("Report saved at:", save_report_path)
