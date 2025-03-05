import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import base64

# Ensure correct model path
MODEL_PATH = os.getenv("MODEL_PATH", "models/trained.h5")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()

model = load_model(MODEL_PATH)

# Class Labels
classes = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

# Function to preprocess image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, [255, 255])
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to create a downloadable report
def download_report(pred_class, confidence):
    report_text = f"""
    Brain Tumor Classification Report
    -------------------------------
    Prediction : {pred_class}
    Confidence : {confidence:.2f}%
    """
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">Download Report</a>'
    return href

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

st.sidebar.title("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)
