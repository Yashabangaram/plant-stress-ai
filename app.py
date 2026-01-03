import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("AI Plant Water Stress Detector")

model = tf.keras.models.load_model("plant_stress_model.keras")

class_names = ["Healthy", "Mild Stress", "Severe Stress"]

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)

    preds = model.predict(img_array)
    label = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
