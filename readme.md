
---

## 🧠 Model Overview

- **Architecture**: Simple but effective CNN
- **Input shape**: ` (11766, 157, 40, 3)` representing MFCC + delta + delta-delta + augumentation
- **Final Accuracy on Validation Set**: **85%**
- **Callbacks Used**:
  - `EarlyStopping` (patience=10)
  - `ReduceLROnPlateau` (factor=0.5, patience=3)

---
So each training example is a 3-channel spectrogram of shape (157, 40, 3) representing:

157 time steps (rows)

40 MFCC-related features (columns)

3 channels:

Channel 1 → MFCC

Channel 2 → First-order delta (rate of change)

Channel 3 → Second-order delta (acceleration)

This makes each input comparable to a colored image (like RGB channels), which is why a CNN can process it effectively.


## 📈 Model Evaluation

### Confusion Matrix
<img src="https://github.com/iota765/audio_emotion_detection/blob/main/download.png?raw=true" alt="Confusion Matrix" width="500"/>

### Classification Report

                  precision    recall  f1-score   support

       angry       0.90      0.92      0.91        75
        calm       0.88      0.92      0.90        75
     disgust       0.84      0.79      0.82        39
     fearful       0.82      0.79      0.80        75
       happy       0.89      0.87      0.88        75
     neutral       0.85      0.89      0.87        38
         sad       0.83      0.80      0.82        75
   surprised       0.76      0.82      0.79        39

    accuracy                           0.85       491
   macro avg       0.85      0.85      0.85       491
weighted avg       0.85      0.85      0.85       491


---

## 🔍 Data Challenges

- Audio samples were short and variable in length.
- Extracting robust features from noisy environments was difficult.
- Limited sample size affected complex model generalization.

---

## 🔁 Data Augmentation

Each audio sample was **augmented 6 times** to improve generalization:

1. **Add Noise**
2. **Pitch Up**
3. **Pitch Down**
4. **Time Stretch Fast (1.2x)**
5. **Time Stretch Slow (0.8x)**
6. **Original**

---

🔍 Why Simple CNN Worked Better than Advanced Models
Although combinations like CNN+LSTM, CNN+BiLSTM, and CNN+GRU were experimented with, they didn't yield better accuracy. Here's why:

Limited Data Size: These hybrid models have a higher number of parameters and tend to overfit when trained on relatively small datasets.

Temporal Complexity Not Needed: Since MFCC, delta, and delta-delta already capture temporal dynamics effectively, adding LSTM/GRU layers may have been redundant or even counterproductive.

Padding & Alignment: The time-distributed structure required by LSTM-based architectures is sensitive to sequence length and alignment, which can lead to inconsistencies.

Augmentation Noise: Some augmentations (like pitch-shifting and time-stretching) may have introduced variability that affected sequence modeling more than CNN's local pattern detection.



## 🚀 Streamlit App

You can test your audio directly using our web app.

### Run the App:
```bash
streamlit run app.py
