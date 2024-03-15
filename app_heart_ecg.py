import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load mô hình
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model/ECG.h5')
    return model

model = load_model()

# Danh sách các lớp được cập nhật
classes = {
    0: 'Myocardial Infarction',
    1: 'Have a History of Myocardial Infraction',
    2: 'Abnormal Heart Beat',
    3: 'Normal Heart'
}

st.title("ECG Image Classifier")

st.write("This application classifies ECG images into four categories based on the condition of the heart.")

uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded ECG Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    def preprocess_image(image, target_size=(224, 224)):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)

        return image

    image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(tf.nn.softmax(predictions[0])) * 100

    # Display the prediction
    st.write("Prediction:")
    st.write(f"{classes[predicted_class]} with confidence {confidence:.2f}%")
