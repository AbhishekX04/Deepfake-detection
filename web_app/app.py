import gradio as gr
import os
import sys
import torch

# --------------------------------------------------------
# FIX: Add project root path
# --------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from scripts.run_inference import load_model, analyze_full_video


# --------------------------------------------------------
# Load model ONCE
# --------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(PROJECT_DIR, "models/cnn_lstm_detector.pt")

print("Loading model for Gradio UI...")
model = load_model(MODEL_PATH, device)
print("Model loaded.")


# --------------------------------------------------------
# Wrapper for Gradio
# --------------------------------------------------------
def detect(video_file):

    if video_file is None:
        return "No video uploaded!", None, None, None

    print("Running inference on:", video_file)

    label, score, result = analyze_full_video(video_file, model, device)

    explanation_text = result["explanation"]
    blink_count = result["blink_count"]
    micro_score = result["micro_score"]
    gradcam_imgs = result["gradcam_overlay"]   # list of file paths

    # Handle GradCAM visualizations
    gradcam_output = None
    if isinstance(gradcam_imgs, list) and len(gradcam_imgs) > 0:
        gradcam_output = gradcam_imgs  # Gradio supports list of images

    summary = f"""
### FINAL PREDICTION
**Label:** {label}  
**Confidence:** {round(score, 3)}

### SIGNALS USED
- Blink Count: {blink_count}  
- Micro-Expression Score: {micro_score}  
- Model Probability: {round(score, 3)}

### Explanation
{explanation_text}
"""

    return summary, gradcam_output


# --------------------------------------------------------
# Gradio UI Layout
# --------------------------------------------------------
with gr.Blocks(title="Deepfake Detection System") as demo:

    gr.Markdown("## Upload a video to detect if it is REAL or FAKE using Model + Blink + Micro-expression analysis.")

    video_input = gr.Video(label="Upload Video (MP4 / MOV / AVI)")

    submit_btn = gr.Button("Analyze Video")

    summary_box = gr.Markdown("Results will appear here...")
    gradcam_gallery = gr.Gallery(label="Grad-CAM Visualizations", columns=3, rows=1)

    submit_btn.click(
        detect,
        inputs=[video_input],
        outputs=[summary_box, gradcam_gallery]
    )

demo.launch(share=True)
