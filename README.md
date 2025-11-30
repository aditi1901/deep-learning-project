# deep-learning-project
# 🧠 Confused Student Prediction using EEG Brainwave Data

This project explores **multimodal deep learning** to detect whether a student is **confused or not confused** while watching educational content. The model uses:

* **EEG Brainwave Signals**
* **Video Frame Embeddings**
* **Demographic Information**

The objective is to build a system that can help adaptive learning platforms understand student cognitive states in real-time.

---

## 📂 Dataset

Dataset used: **Confused Student EEG Brainwave Dataset**
Source: [Kaggle (Wang et al.)](https://www.kaggle.com/datasets/wanghaohan/confused-eeg)

The dataset includes:

| Component      | Description                              |
| -------------- | ---------------------------------------- |
| EEG recordings | 10 students × multiple videos            |
| Labels         | Confused (1) / Not Confused (0)          |
| Demographics   | Age, gender, ethnicity                   |
| Videos         | Corresponding video clips for embeddings |

---

##  Tech Stack

| Category                | Tools Used          |
| ----------------------- | ------------------- |
| Language                | Python              |
| Deep Learning Framework | TensorFlow / Keras  |
| Classical ML            | XGBoost             |
| Video Processing        | OpenCV + ResNet50   |
| Visualization           | Matplotlib, Seaborn |

---

## 🛠 Preprocessing Pipeline

✔ EEG resampled to fixed length (120 timesteps)
✔ Z-score normalization per channel
✔ Data augmentation (noise, channel dropout, masking)
✔ Video embeddings extracted using **ResNet50**
✔ Demographics encoded using **one-hot encoding**
✔ Final dataset converted to TensorFlow pipelines

---

## 🧠 Models Implemented

| Model                       | Description                                                             | Fusion Type              |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------ |
| **Base Model**              | Conv1D for EEG + Dense for video + demographic embeddings               | Late Fusion              |
| **Hybrid Model** ⭐          | Conv1D + Inception blocks + Transformer video encoder + Cross-attention | Attention-based Fusion   |
| **Transfer Learning Model** | EEG converted into images and passed through ResNet50                   | Image & Embedding Fusion |

Classical model: **XGBoost Ensemble** used to improve final reliability.

---

## 📊 Evaluation Method

We used **Leave-One-Subject-Out (LOSO) cross-validation** to ensure models generalize across individuals.

---

## 🚀 Results

| Model             | Result Summary                             |
| ----------------- | ------------------------------------------ |
| Base Model        | Works but unstable across subjects         |
| Transfer Learning | Better stability but not highest accuracy  |
| **Hybrid Model**  | Best overall F1-score and generalization   |

> The Hybrid model consistently outperformed others across folds.

