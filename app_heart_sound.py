import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Assuming you have a trained LSTM model saved at 'model/lstm_model.h5'
MODEL_PATH = 'model/lstm_model.h5'
DURATION = 10  # Duration to which audio files will be padded or trimmed
SR = 22050  # Sampling rate to be used for audio files

def main():
    st.title("Predict Heart Disease by Heart Sounds")
    st.markdown("Upload a .wav file of your heart sound to predict the condition.")

    # Load the LSTM model
    model = load_model(MODEL_PATH)

    audio_file = st.file_uploader("Upload Heart Sound (.wav)", type=['wav'])
    if audio_file is not None:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_file, sr=SR, duration=DURATION)
        if librosa.get_duration(y=audio, sr=sr) < DURATION:
            audio = librosa.util.fix_length(audio, size=SR * DURATION)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=52, n_fft=512, hop_length=2048)
        mfcc_mean = np.mean(mfccs.T, axis=0)

        # Reshape for the model
        mfcc_reshaped = mfcc_mean.reshape(1, 52, 1)

        # Predict the condition
        prediction = model.predict(mfcc_reshaped)

        # Map prediction to labels
        classes = ["artifact", "extrahls", "extrastole", "murmur", "normal"]
        predicted_label = classes[np.argmax(prediction)]
        confidence = np.max(prediction)  # Get the confidence of the prediction

        st.success(f"Predicted Heart Condition: {predicted_label}")
        st.write(f"Prediction Probability: {confidence}")

        # Plotting the waveform using matplotlib
        st.subheader("Heart Sound Waveform")
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, DURATION, len(audio)), audio)
        plt.title("Waveform of the Heart Sound")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        st.pyplot(plt)

if __name__ == "__main__":
    main()
