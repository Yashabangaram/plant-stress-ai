import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="DIY Farmer's Eye", layout="centered")
st.title("🌱 The DIY Farmer’s Eye")
st.write("Upload a plant image to detect water stress using AI.")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("FINAL_plant_stress_model_3class.h5")
    return model

model = load_model()

# ---------- CLASS NAMES ----------
class_names = ["Healthy", "Mild Stress", "Severe Stress"]

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload a plant image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

# ---------- IMAGE PROCESSING + PREDICTION ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    st.subheader("🧠 AI Prediction")
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
