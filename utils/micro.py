import os
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def extract_landmark_motion_from_folder(video_folder):
    files = sorted(os.listdir(video_folder))

    motion_curve = []
    valid = 0

    with mp_face_mesh.FaceMesh(static_image_mode=True) as fm:
        prev_landmarks = None

        for file in files:
            path = os.path.join(video_folder, file)
            img = cv2.imread(path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if not res.multi_face_landmarks:
                motion_curve.append(None)
                continue

            valid += 1
            h, w, _ = img.shape
            lm = res.multi_face_landmarks[0].landmark

            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in range(468)])

            if prev_landmarks is None:
                motion_curve.append(0)
            else:
                diff = np.linalg.norm(pts - prev_landmarks, axis=1)
                motion_curve.append(np.mean(diff))

            prev_landmarks = pts

    return motion_curve, valid
