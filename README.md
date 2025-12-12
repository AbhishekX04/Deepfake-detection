# Deepfake Video Detection Using Eye Blink, Micro-Expression & CNN-LSTM

A hybrid deepfake detection system combining **physiological cues** (eye blink patterns, facial micro-expressions) with **deep learning (CNN + LSTM)** and **Grad-CAM visual explanation**.  
Works at **frame-level** and detects fake regions with explainability.

---

## ⭐ Key Features

### 1. Frame-Level Deepfake Detection  
- Extracts sequence of frames  
- CNN captures spatial features  
- LSTM captures temporal motion  
- Output: Probability (Real vs Fake)

### 2. Physiological Signal Detection  
#### Eye Blink  
- Extract EAR (Eye Aspect Ratio) via Mediapipe  
- Detect abnormal blink frequency  

#### Micro-Expression Motion  
- Landmark motion curves  
- Detect unnatural facial muscle movement  

### 3. Grad-CAM Explainability  
- Highlights manipulated facial regions  
- Generates heatmaps + overlays  
- Transparent model decisions

### 4. Gradio Web App  
Simple UI for:
- Video upload  
- Real/Fake prediction  
- Confidence  
- Blink count  
- Micro-expression irregularity  
- Grad-CAM previews  

---

## 📁 Folder Structure

```
deepfake-detection/
│
├── models/
│     ├── cnn_lstm.py
│     └── (weights downloaded separately)
│
├── scripts/
│     ├── train.py
│     ├── run_inference.py
│     ├── blink_analysis.py
│     ├── micro_expression_analysis.py
│     ├── gradcam_video.py
│     └── gradcam_test.py
│
├── utils/
│     ├── blink.py
│     ├── micro.py
│     ├── fusion.py
│     ├── gradcam.py
│     └── __init__.py
│
├── web_app/
│     └── app.py
│
├── data/
│     ├── raw/            (empty)
│     ├── frames/         (empty)
│     └── preprocessed/   (empty)
│
├── results/              (empty)
│
├── run_inference.py
├── README.md
└── requirements.txt
```

---

## 📥 Download Model Weights  
Place the downloaded files into:

```
models/
```

| Model Weight File | Size | Download Link |
|-------------------|-------|------------------------------------------------------------|
| **best_model.pt** | 25 MB | https://drive.google.com/file/d/1NroeFEm3OYvaPFjdTWQCU8nLsgW80ffC/view |
| **cnn_lstm_detector.pt** | 21 MB | https://drive.google.com/file/d/1LA-9EJNK6n2nyO6uRVf3YZNpp-B5p9RV/view |
| **deepfake_detector_full.pth** | 25 MB | https://drive.google.com/file/d/1fb5vsGwsvWCwVe19uy6Vk7FeSTUu9KBo/view |
| **deepfake_detector_weights.pth** | 15 MB | https://drive.google.com/file/d/1ZD0VDUaKN_Qh2n7OQ_B-M-yMkpneKSZ9/view |

---

## 📦 Dataset Setup (FaceForensics++ Mini Subset)

Download the dataset and place it into:

```
data/raw/
```

Run preprocessing to generate:
- frames  
- resized images  
- dataloaders  

---

## 🧠 Model Architecture

### CNN (MobileNetV2)
Extracts spatial facial features per frame.

### LSTM  
Processes temporal sequence of CNN features.

### Classifier  
Outputs deepfake probability from temporal features.

---

## 🏋️ Training

Run training (Colab recommended):

```bash
python scripts/train.py --epochs 5 --batch_size 2
```

Automatically saves:
```
best_model.pt
```

---

## 🔍 Inference (CLI)

```bash
python scripts/run_inference.py --video your_video.mp4 --weights models/cnn_lstm_detector.pt
```

Outputs:
- FAKE / REAL  
- Confidence score  
- Blink count  
- Micro-expression irregularity  
- Grad-CAM heatmaps saved to disk  

---

## 🌐 Gradio Web App

Run:

```bash
python web_app/app.py
```

Features:
- Upload video  
- Model predicts real/fake  
- Explainability output  
- Visual Grad-CAM highlights  

---

## 🔥 Features Demonstrated

- Hybrid physiological + deep learning pipeline  
- EAR blink detection  
- Landmark micro-expression tracking  
- CNN + LSTM temporal model  
- Grad-CAM explainability  
- End-to-end video inference  
- Clean modular Python package structure  

---

## 🚀 Future Improvements

- EfficientNet-Lite or ViT backbones  
- 3D CNN temporal modeling  
- Larger datasets (DFDC, CDF-v2)  
- Face alignment + segmentation  
- On-device mobile deployment  

---

## 🤝 Credits

- Mediapipe for blink + facial landmark extraction  
- PyTorch for CNN-LSTM model  
- FaceForensics++ for dataset  
- Gradio for UI  
- Guided development with Deepfake Mentor (ChatGPT)

---

## 📌 Notes

This repository includes **all source code**, but excludes:
- Large trained weights (download separately)
- Large datasets  
- Temp frames & Grad-CAM videos  

This keeps the repo clean, lightweight, and professional.

