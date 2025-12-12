import numpy as np

def fuse_scores(model_pred, blink_count, motion_score, gradcam_heatmaps):
    model_score = float(model_pred)

    if blink_count < 2:
        blink_score = 0.85
    elif blink_count < 5:
        blink_score = 0.65
    else:
        blink_score = 0.25

    motion_score_norm = min(1.0, motion_score / 3.0)

    if gradcam_heatmaps is not None:
        strengths = [np.mean(cam) for cam in gradcam_heatmaps]
        gradcam_score = float(np.mean(strengths))
    else:
        gradcam_score = 0.5

    final_score = (
        0.50 * model_score +
        0.20 * blink_score +
        0.20 * motion_score_norm +
        0.10 * gradcam_score
    )

    label = "FAKE" if final_score >= 0.55 else "REAL"

    explanation = f"""
Model predicted fake probability: {model_score:.2f}
Blink count: {blink_count}
Micro-expression irregularity: {motion_score:.2f}
Grad-CAM intensity: {gradcam_score:.2f}

Final Decision: {label} ({final_score:.2f})
"""

    return final_score, label, explanation
