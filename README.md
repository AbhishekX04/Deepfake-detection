# Deepfake Detection through Eye Blink Behaviour and Micro-Expression Motion Patterns

## 📌 Overview
This project presents a **behavioural deepfake detection system** that analyzes:

- **Eye blink behaviour abnormalities**
- **Micro-expression motion inconsistencies**
- **Manipulated facial regions using Grad-CAM**
- **CNN + LSTM deep learning architecture for video-based detection**

Unlike traditional deepfake detectors that rely only on pixel features, this project focuses on **human physiological cues** that deepfake models often fail to replicate correctly.

---

## 🚀 Key Features

### ✔ Deepfake Classification (CNN + LSTM)
Extracts spatial features using **MobileNetV2** and models temporal dynamics with **LSTM**.

### ✔ Eye Blink Behaviour Analysis (EAR Technique)
Deepfakes often blink:
- too little,  
- too slowly, or  
- in unnatural patterns.

We calculate EAR (Eye Aspect Ratio) and estimate:
- Blink count  
- Average EAR  
- Abnormal blink patterns  

### ✔ Micro-Expression Motion Pattern Analysis
Deepfakes show irregularities in:
- eyebrow motion  
- cheek tension  
- lip corners  
- eye wrinkles  

We compute a **Facial Motion Energy Map (FMEM)** to detect anomalous motion peaks.

### ✔ Grad-CAM Visualization
Highlights **frame-level manipulated regions** such as:
- mouth edges  
- eyes  
- cheeks  
- forehead  

This provides visual explainability for the classification model.

### ✔ Full Video Heatmap Rendering
Generates a **side-by-side video**:  
Original frames ↔ Grad-CAM overlay frames.

---

## 🧠 Model Architecture
```
MobileNetV2 → LSTM → Fully Connected Layer → Sigmoid → Real/Fake
```

- MobileNetV2 extracts spatial features from frames  
- LSTM models temporal continuity  
- Output is binary classification (Real/Fake)

---

## 🎯 Dataset
Dataset Used: **FaceForensics++ (Mini Subset)**

Videos were converted into frames:

```
data/frames/real/<video_id>/
data/frames/fake/<video_id>/
```

Each folder contains 50–150 frames per video.

---

## 🔥 Sample Outputs

### 🟦 Micro-Expression Motion Graph
![Micro Expression Graph](results/micro_expression_motion_047_862.png)

### 🔥 Grad-CAM Heatmap Example
![GradCAM](gradcam_heatmap.jpg)

### 🔥 Grad-CAM Overlay Example
![Overlay](gradcam_overlay.jpg)

---

## 📈 Training Metrics
The training pipeline outputs:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC (Area Under ROC Curve)  
- Confusion Matrix  

This ensures proper evaluation of both classification and detection performance.

---

## 🏗 Project Structure

```
deepfake-detection/
│
├── data/
│   ├── frames/
│   └── raw/
│
├── models/
│   └── best_model.pt
│
├── scripts/
│   ├── train.py
│   ├── gradcam_test.py
│   ├── gradcam_video.py
│   ├── blink_analysis.py
│   └── micro_expression_analysis.py
│
├── utils/
│   └── gradcam.py
│
└── README.md
```

---

## ▶️ Running the Project

### 🔹 Train Your Deepfake Detection Model
```bash
python scripts/train.py
```

### 🔹 Run Single-Frame Grad-CAM
```bash
python scripts/gradcam_test.py
```

### 🔹 Generate Full Grad-CAM Video
```bash
python scripts/gradcam_video.py
```

### 🔹 Blink Behaviour Analysis
```bash
python scripts/blink_analysis.py
```

### 🔹 Micro-Expression Motion Analysis
```bash
python scripts/micro_expression_analysis.py
```

---

## 👥 Team Members

### **Abhishek B. — Team Lead**
- Model Architecture  
- Training Pipeline  
- Grad-CAM Visualization  
- System Design  

### **Deeksha — Research & Testing**
- Behavioural Pattern Analysis  
- Blink/Micro-expression Studies  
- Documentation Support  

### **Khushi Agarwal — Research & Testing**
- Dataset Processing  
- Evaluation Metrics  
- Testing & Verification  

### 🎓 **College**
**Lovely Professional University**

---

## 📝 Conclusion
This project successfully demonstrates:

- Behaviour-based deepfake detection  
- Frame-level manipulation localization  
- Micro-expression and blink anomaly detection  
- A complete deep learning pipeline for research & deployment  

It provides a **robust and explainable AI solution** for modern deepfake detection challenges.

---

## ⭐ Future Enhancements

- Real-time webcam-based deepfake detection  
- Multi-modal deepfake analysis (audio + video)  
- Transformer-based temporal modelling  
- Higher-resolution facial landmark tracking  

---

## 📎 Contact
For queries, collaboration, or improvements:  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abhishek%20Bathnotra-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/abhishek-bathnotra-b18075374/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Deeksha-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/deeksha-%E2%80%8E-23a320297/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Khushi%20Agarwal-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/khushi-agarwal-683a49287/)


