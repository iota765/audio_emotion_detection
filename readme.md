# 🎧 Speech Emotion Recognition using CNN and MFCC

This project implements a robust deep learning pipeline to classify human emotions from speech audio. Leveraging **MFCC**, **delta**, and **delta-delta** features combined with **2D CNNs**, the model achieves an impressive **81% accuracy**.    

---

## 🔍 Evaluation Results

### 📉 Confusion Matrix

<p align="center">
  <img src="ba9482ce-a79c-4433-baaf-1ed977f18cc3.png" alt="Confusion Matrix" width="500"/>
</p>

### 🧾 Classification Report

          precision    recall  f1-score   support

   angry       0.87      0.91      0.89        75
    calm       0.90      0.81      0.85        75
 disgust       0.94      0.77      0.85        39
 fearful       0.73      0.81      0.77        75
   happy       0.79      0.79      0.79        75
 neutral       0.80      0.87      0.84        38
     sad       0.78      0.71      0.74        75


---

## 🧠 Overview

Given a raw audio file (WAV/MP3), this system:

1. Preprocesses the signal to a fixed length
2. Extracts MFCC and its derivatives
3. Predicts one of the following 8 emotions:
   - Neutral
   - Calm
   - Happy
   - Sad
   - Angry
   - Fearful
   - Disgust
   - Surprised

---

## 📊 Dataset

The model is trained on the [RAVDESS](https://zenodo.org/record/1188976) emotional speech dataset.


---
- 📉 **Limited Data**: Achieving high performance was challenging due to the relatively small size of the dataset. This was mitigated using targeted data augmentation techniques like noise addition, pitch shifting, and time stretching to improve generalization.


## 🎛️ Data Augmentation Strategy

To improve generalization and increase training data 6×, we applied the following augmentations on each sample:

```python
# 1. Add Gaussian noise
noisy = add_noise(signal)

# 2. Pitch up
pitch_up = shift_pitch(signal, sr, n_steps=+2)

# 3. Pitch down
pitch_down = shift_pitch(signal, sr, n_steps=-2)

# 4. Time stretch (faster)
stretch_fast = time_stretch(signal, rate=1.2)

# 5. Time stretch (slower)
stretch_slow = time_stretch(signal, rate=0.8)


## 🤔 Why Simple CNN Outperformed Complex Models

Although various advanced architectures were explored—such as **CNN + LSTM**, **CNN + GRU**, and **CNN + BiLSTM**—they **failed to outperform** the simpler CNN model, even after aggressive data augmentation.

### 🔑 Key Reasons:

1. **📊 Dataset Still Limited**
   - Despite 6x augmentation, the core dataset lacked **diversity in real emotional variation** (speaker, language, tone).
   - Larger and deeper models like LSTMs tend to **overfit quickly** when real-world diversity is low.

2. **🎭 Augmentation ≠ Real Emotion Diversity**
   - Pitch shifting, noise, and time-stretching introduce variability but **don’t truly replicate emotional nuance**.
   - These synthetic transformations may cause RNNs to **memorize augmentation patterns** instead of general emotion cues.

3. **🧠 MFCCs Already Summarize Audio**
   - MFCCs are short-time spectral features; applying LSTMs or GRUs on them doesn’t add much because **temporal dynamics are already compressed**.
   - CNNs are **more efficient** at learning local acoustic patterns in these fixed-size feature maps.

4. **⏱️ Limited Sequence Complexity**
   - LSTMs are designed for long-term dependencies (e.g., text, long audio).
   - Here, each clip is **cropped/padded to 5 seconds**, and emotions tend to manifest in **short-term spectral cues** (which CNNs can detect well).

5. **⚙️ Simpler Model = Better Regularization**
   - The CNN architecture, with **batch normalization**, **dropout**, and **early stopping**, generalized better.
   - Complex models required **more tuning and regularization** and still showed instability or poor convergence.

### ✅ Conclusion:
The CNN architecture struck the **perfect balance** between complexity and generalizability for this dataset. It learned robust representations without overfitting — making it the best-performing model for this emotion classification task.

## 🚀 Running the App

To launch the audio emotion recognition web app locally using Streamlit:

```bash
streamlit run app.py
