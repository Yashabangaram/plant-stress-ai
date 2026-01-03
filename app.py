import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.cache_resource.clear()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("FINAL_plant_stress_model_3class.h5")

model = load_model()

class_names = ["Healthy", "Mild Stress", "Severe Stress"]
