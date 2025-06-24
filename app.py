import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile

# Emotion labels
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Constants
SR = 16000
TARGET_DURATION = 5
MAX_LEN = SR * TARGET_DURATION
EXPECTED_SHAPE = (157, 40, 3)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_recognition_model(new).h5")
    return model

def pad_or_crop(signal, sr=SR, target_duration=TARGET_DURATION):
    max_len = sr * target_duration
    if len(signal) < max_len:
        return np.pad(signal, (0, max_len - len(signal)))
    else:
        return signal[:max_len]

def extract_mfcc_full(y, sr, n_mfcc=40):
    y = pad_or_crop(y, sr, TARGET_DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.stack([mfcc.T, mfcc_delta.T, mfcc_delta2.T], axis=-1)  # Shape: (T, 40, 3)

    # Pad or crop time dimension (T) to 157
    if features.shape[0] < 157:
        pad_len = 157 - features.shape[0]
        features = np.pad(features, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
    elif features.shape[0] > 157:
        features = features[:157, :, :]

    return features

# --- Streamlit UI ---
st.set_page_config(page_title="🎤 Emotion Predictor", layout="centered")
st.title("🎧 Audio Emotion Recognition")
st.markdown("Upload a `.wav` or `.mp3` file and predict the emotion using a CNN model.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SR)
        st.audio(uploaded_file, format="audio/wav")

        # Show waveform
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Audio Waveform")
        st.pyplot(fig)

        # Extract features
        features = extract_mfcc_full(y, sr)

        if features.shape != EXPECTED_SHAPE:
            st.error(f"Feature shape mismatch: got {features.shape}, expected {EXPECTED_SHAPE}")
        else:
            model = load_model()
            input_tensor = np.expand_dims(features, axis=0)  # (1, 157, 40, 3)
            prediction = model.predict(input_tensor)[0]  # shape: (8,)

            # Show top prediction
            top_label = EMOTION_LABELS[np.argmax(prediction)]
            st.subheader("🎯 Predicted Emotion")
            st.success(f"**{top_label.upper()}**")

            # Show probability chart
            st.subheader("📊 Emotion Probabilities")
            prob_chart = {label: float(prediction[i]) for i, label in enumerate(EMOTION_LABELS)}
            st.bar_chart(prob_chart)

    except Exception as e:
        st.error(f"Error: {e}")
