import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("DIY Farmer’s Eye 🌱")
st.write("Upload a plant image to detect water stress.")

model = tf.keras.models.load_model("practice_plant_model.h5", compile=False)

class_names = ["Healthy", "Mild Stress", "Severe Stress"]

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")
