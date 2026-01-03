import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Plant Water Stress Detector")

# Load model
model = tf.keras.models.load_model("FINAL_plant_stress_model_3class.h5")

class_names = ["Healthy", "Mild Stress", "Severe Stress"]

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")
