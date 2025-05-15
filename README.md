# 🧠 SentiAid – Speech to Animated Sign Language Interpreter

SentiAid is an AI-powered solution designed to bridge the communication gap between the Deaf and Hard-of-Hearing community and the rest of the world. Our system converts **spoken or textual input** into **real-time animated Indian Sign Language (ISL)** using deep learning, keypoint detection, and pose-based animation.

> 🌐 Built with: TensorFlow, Mediapipe, VideoMAE, LangGraph, LangChain, Transformers, IndicTrans2, LLM Agentic Frameworks

---

## 🚀 Project Objective

To develop a robust and scalable pipeline that:
- Converts **speech/text to ISL**
- Animates the signs using pose/keypoint-based avatars
- Supports **multilingual inputs** (22 scheduled Indian languages)
- Ensures **accessibility** for education, workplaces, and public communication

---

## ✨ Key Features

- 🎙️ **Speech to Text** using Whisper / IndicTTS
- 📝 **Text to Gloss Conversion** with Agentic LLMs (LangGraph)
- 🔁 **Gloss to ISL Keypoints** using masked motion templates
- 🧍‍♂️ **ISL Animation Renderer** via Mediapipe pose keypoints and Blender/Three.js
- 🌐 **Multilingual Support** using IndicTrans2
- 📱 **Mobile-friendly multilingual web app**

---

## 🧠 Architecture

```text
Speech Input
   ⬇️
[Whisper/IndicTTS] ➡️ [Text]
   ⬇️
[LangGraph Agentic Flow] ➡️ [Gloss Conversion]
   ⬇️
[Gloss ➡️ Pose Sequence] via Transformer / LSTM
   ⬇️
[Masked Pose Templates + ISL Sign Mapping]
   ⬇️
🎥 Animated Avatar Output (Sign Language)

## Dataset Details 📊
Link to the Dataset: [INCLUDE Dataset](https://zenodo.org/records/4010759)

The INCLUDE dataset, sourced from AI4Bharat, forms the foundation of our project. It consists of 4,292 videos, with 3,475 videos used for training and 817 videos for testing. Each video captures a single Indian Sign Language (ISL) sign performed by deaf students from St. Louis School for the Deaf, Adyar, Chennai.


## 🧠 Model Architecture

SentiAid leverages two powerful AI models for robust real-time Sign Language Detection:

### 1. LSTM-based Model 📈
Utilizes Mediapipe-extracted keypoints to model dynamic hand and body movements over time.

- **TimeDistributed Layers**: Extract spatial relationships from each frame.
- **LSTM Layers**: Capture temporal patterns across frames for gesture sequence classification.

### 2. Transformer-based Model 🔄
Delivers high performance through extensive hyperparameter tuning and modern training strategies.

- **Training Techniques**:
  - **Warmup Scheduler**: Gradual learning rate increase to stabilize training.
  - **AdamW Optimizer**: Improved version of Adam for better generalization.
  - **ReduceLROnPlateau**: Dynamic learning rate adjustment.
  - **Finetuned VideoMAE**: Adapted from [VideoMAE](https://arxiv.org/abs/2203.12602) with head-only fine-tuning for optimal efficiency.

---

## 🎯 Solution Approach

SentiAid tackles communication barriers through a dual-mode pipeline:

### 1. Sign Language ➡️ Text
- **Input**: Real-time sign language video.
- **Pipeline**:
  - Mediapipe → Keypoint extraction.
  - LSTM/Transformer → Sign Classification.
  - Agentic LangChain Flow → Text generation.

### 2. Text ➡️ Sign Language
- **Input**: Typed or spoken natural language.
- **Pipeline**:
  - Text → Gloss conversion.
  - Generate masked keypoint-based animations.
  - Output: Realistic ISL sign language videos.

---

## 📋 Action Plans

1. **Pose-to-Text Model**:
   - Build using Mediapipe keypoints + LangGraph for gloss prediction.
2. **Custom Transformer Evaluation**:
   - Compare accuracy, speed, and adaptability on Indian Sign Language datasets.
3. **Multilingual App**:
   - Launch a cross-platform app for real-time ISL translation with accessibility-first UX.

---

## ✅ Progress So Far

- [x] LSTM model for ISL recognition
- [x] Transformer-based custom encoder for dynamic signs
- [x] Full Indian Sign Language dataset tested on Transformer model
- [x] Agentic LangChain implementation for Pose-to-Text
- [x] Multilingual App built with UI optimized for accessibility
- [x] https://www.sentiaid.co.in/

---
## 🚀 Future Work

While SentiAid has made significant progress in both Sign-to-Text and Text-to-Sign pipelines, the next phase will focus on enhancing performance, scalability, and usability. Here’s what lies ahead:

### 🧠 Model Improvements
- **Benchmark and optimize LSTM model**:
  - Finalize evaluation metrics (precision, recall, F1-score) for LSTM results.
  - Compare with Transformer results on unseen dynamic signs.
- **Hybrid Model Fusion**:
  - Explore combining LSTM (for local temporal features) with Transformers (for global context) for improved accuracy.
- **Real-time Latency Optimization**:
  - Reduce inference time using quantization, pruning, and TensorRT/ONNX conversion.

### 🧾 Dataset Expansion & Diversity
- **Fine-tune on domain-specific ISL phrases** (e.g., banking, healthcare).
- **Augment dataset with regional sign variations** for better generalization.
- **Add multimodal data** (facial expressions, eye gaze) to improve emotion/context recognition.

### 📱 App Features & Accessibility
- **Integrate Speech-to-Text + Text-to-Sign pipeline in mobile app**.
- **Offline Mode**:
  - Add support for core sign recognition features without internet dependency.
- **Real-time Avatar Animation**:
  - Implement animated 3D avatar using Blender/WebGL to visualize generated sign language.

### 🕸️ API & Platform Integration
- **Develop RESTful API for easy third-party integration**.
- **Plug-and-play SDK** for EdTech and HR tools to use sign translation modules.

### 🧑‍🏫 User Feedback & Community Testing
- **Beta testing with DHH users** to collect real-world feedback.
- **Accessibility audit** to meet WCAG 2.1 AA standards.

### 📊 Research & Publication
- **Publish comparative study** between LSTM and Transformer models.
- **Submit paper** to conferences like ACL, COLING, or EMNLP under AI for Accessibility category.

> The journey ahead is to move from working prototypes to production-ready, scalable tools that redefine accessibility for millions.


## 🔗 Key Resources

- 🔬 [ISL Dataset](https://zenodo.org/records/4010759)
- 📘 [VideoMAE on HuggingFace](https://huggingface.co/MCG-NJU/videomae-base)
- 🔗 [AI4Bharat NLP & Speech Models](https://huggingface.co/ai4bharat)
- 🔧 [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- 🧬 [LangGraph for Agentic Workflows](https://python.langchain.com/docs/langgraph/)
- 🧏 [History of Indian Sign Language](https://islrtc.nic.in/history-0)

---

> **SentiAid** is dedicated to creating inclusive technology — empowering the DHH community with intelligent, real-time communication tools.

