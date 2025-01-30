# 🎵 Music Genre Classification & Recommendation System

## 📌 Overview
This project focuses on **music genre classification** using various signal processing and machine learning techniques. The goal is to extract meaningful audio features and apply **classical machine learning models, artificial neural networks (ANNs), and convolutional neural networks (CNNs)** to classify different music genres.

Additionally, a **music recommendation system** has been implemented using **cosine similarity and Dynamic Time Warping (DTW)**.

---

## 📂 Project Workflow

### 🔹 1️⃣ Feature Extraction & Signal Processing
- Extracted key **audio features** such as:
  - **Mel-Frequency Cepstral Coefficients (MFCCs)**
  - **Log Mel-Spectrograms**
  - **Chroma Features**
  - **Spectral Centroid, Bandwidth, Roll-off, and Zero Crossing Rate**
- Used **Librosa** to preprocess and analyze the audio signals.

### 🔹 2️⃣ Classical Machine Learning Approaches
- Applied traditional **ML classifiers** including:
  - **Support Vector Machines (SVM)**
  - **Random Forest**
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
  - **Decision Trees**
- Compared model performances using confusion matrices.

### 🔹 3️⃣ Artificial Neural Networks (ANN)
- Designed and trained an **ANN-based classifier** to improve genre prediction.
- **Input:** Extracted features (MFCCs, Log Mel-Spectrograms).
- **Layers:** Fully connected dense layers with activation functions.

### 🔹 4️⃣ Music Recommendation System
- Built a **music recommender system** using:
  - **Cosine Similarity**: Finding the closest songs based on extracted features.
  - **Dynamic Time Warping (DTW)**: Improved similarity matching by handling time variations in audio signals.

### 🔹 5️⃣ Convolutional Neural Network (CNN) - Best Performance
- Implemented a **CNN model** for **spectrogram-based classification**.
- **CNN Architecture:**
  - **Conv2D layers** with **ReLU activation**
  - **MaxPooling layers** for feature reduction
  - **Batch Normalization** for stable training
  - **Fully Connected (Dense) Layers**
  - **Softmax activation** for classification



---

## 🛠️ Technologies Used
- **Google Colab (Jupyter Notebooks)**
- **Python**
- **Librosa** (Audio Processing)
- **Scikit-learn** (Classical ML Models)
- **TensorFlow/Keras** (Deep Learning)
- **Matplotlib & Seaborn** (Visualization)
- **NumPy & Pandas** (Data Handling)

---

## 📜 License
This project is open-source. Feel free to use and modify it.

---

## 💡 Acknowledgments
- **Librosa** for audio feature extraction.
- **Scikit-learn & TensorFlow** for machine learning and deep learning.
- **GTZAN Dataset** for music genre classification.

---

### 🚀 Connect with Me
If you have any questions or suggestions, feel free to reach out!
---

