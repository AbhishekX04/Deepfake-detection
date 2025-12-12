import os
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def compute_EAR(landmarks):
    V1 = np.linalg.norm(landmarks[1] - landmarks[5])
    V2 = np.linalg.norm(landmarks[2] - landmarks[4])
    H  = np.linalg.norm(landmarks[0] - landmarks[3])
    return (V1 + V2) / (2.0 * H)

def extract_blink_timeline_from_folder(video_folder):
    files = sorted(os.listdir(video_folder))
    EAR_list = []
    valid_frames = 0

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        for file in files:
            path = os.path.join(video_folder, file)
            img = cv2.imread(path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if not res.multi_face_landmarks:
                EAR_list.append(None)
                continue

            valid_frames += 1
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = img.shape

            left_eye = np.array([[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE])
            right_eye = np.array([[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE])

            EAR = (compute_EAR(left_eye) + compute_EAR(right_eye)) / 2
            EAR_list.append(EAR)

    return EAR_list, valid_frames

def detect_blinks(EAR_list, thresh=0.20, min_frames=3):
    blinks = 0
    blink_frames = []
    c = 0

    for idx, ear in enumerate(EAR_list):
        if ear is None:
            continue
        if ear < thresh:
            c += 1
        else:
            if c >= min_frames:
                blinks += 1
                blink_frames.append(idx)
            c = 0

    return blinks, blink_frames
