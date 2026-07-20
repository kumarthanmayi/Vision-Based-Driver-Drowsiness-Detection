# Vision-Based Driver Drowsiness & Distraction Detection Using Lightweight CNN on Edge Devices

### Real-Time Monitoring using Transfer Learning and Edge-Optimized Inference

## 🔍 Motivation & Core Idea

Driver drowsiness and distraction are among the leading causes of road accidents, particularly during long-duration and night-time driving. Many existing solutions rely on intrusive sensors or expensive hardware, limiting their practicality for large-scale deployment.

The core idea of this project is to develop a vision-based driver monitoring system that leverages computer vision and deep learning to detect driver alertness in real time while being lightweight enough to run on edge devices such as Raspberry Pi.

This project emphasizes practical deployment, real-time performance, and efficient edge-device inference rather than focusing solely on model accuracy.

---

## 🎯 Project Objective

- Monitor driver alertness in real time using facial cues.
- Detect and classify driver states as **Active**, **Distracted**, or **Drowsy**.
- Utilize Transfer Learning for efficient model training.
- Optimize inference for edge-device deployment using TensorFlow Lite.
- Integrate visual and audio feedback mechanisms for timely alerts.

---

## ✨ Key Features

- Real-time facial landmark detection using MediaPipe.
- Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) based behavioural analysis.
- Lightweight CNN based on MobileNetV2.
- Pretrained and optimized TensorFlow Lite (TFLite) model for fast inference.
- Audio alert system to warn the driver during drowsy states.
- Modular architecture enabling easy future enhancements.
- Successfully deployed on Raspberry Pi for edge-based inference.

---

## 🧠 Model & Technical Approach

**Model Backbone:** MobileNetV2 (Pretrained on ImageNet)

**Learning Strategy:** Transfer Learning

- Utilizes pretrained feature extraction for edges, textures, and facial patterns.
- Reduces training time and improves generalization on limited data.

**Optimization**

- Trained Keras model converted to TensorFlow Lite (TFLite).
- Designed for low-latency inference on edge devices.

**Decision Logic**

- CNN-based classification combined with EAR and MAR thresholds for robust driver-state detection.

---

## 🛠️ Technologies Used

### Programming & Frameworks

- Python 3.10
- TensorFlow / Keras
- TensorFlow Lite

### Computer Vision & ML

- OpenCV
- MediaPipe
- Transfer Learning (MobileNetV2)

### Deployment

- Laptop Webcam (Development & Testing)
- Raspberry Pi (Edge Deployment)

---

## ▶️ Running the Project

### 1️⃣ Clone the repository

### 2️⃣ Install dependencies

```bash
pip install opencv-python mediapipe tensorflow numpy pygame
```

### 3️⃣ Run the real-time detection

```bash
python fusion_code.py
```

The system will activate the webcam and display the driver's current state in real time.

> **Note:** The project can be executed on a laptop using a webcam for development and testing. The edge deployment demonstrated in this repository was implemented on a Raspberry Pi using TensorFlow Lite.

---

## ⛔ How to Stop Execution

Press **Q** in the OpenCV display window.

---

## 🎓 Key Learnings

- End-to-end development of a real-time vision-based machine learning system.
- Practical application of Transfer Learning in constrained environments.
- Model optimization and conversion for edge devices using TensorFlow Lite.
- Understanding performance trade-offs in real-world deployments.
- Integrating deep learning with classical computer vision techniques.
- Deploying AI models on Raspberry Pi for efficient edge inference.

---

## 🚀 Future Enhancements

- Model quantization and frame-skipping to further reduce latency.
- Multi-threaded inference pipeline for improved throughput.
- Adaptive driver-specific alert thresholds.
- Integration with vehicle systems for automated safety responses.
- Improved robustness under varying lighting conditions.

---

## 🎥 Demonstration

### Laptop

Live webcam-based real-time detection for development and testing.

### Raspberry Pi

The demonstration video below showcases the complete deployment and real-time execution of the system on a Raspberry Pi using TensorFlow Lite.

**Demo:**

https://github.com/user-attachments/assets/46bbc914-c244-420a-952c-45322cff0f44

---

## 📝 Final Note

This project includes a pretrained, edge-optimized TensorFlow Lite model, enabling immediate real-time inference without retraining.

The source code can be executed on a laptop using a webcam for testing purposes, while the complete edge deployment demonstrated in this repository has been successfully implemented on a Raspberry Pi.

The project focuses on practical applicability, system integration, and deployment readiness, making it suitable for real-world intelligent transportation systems.
