import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ===============================
# 1. Load the trained model
# ===============================
MODEL_PATH = "covid_cnn_model.h5"
model = load_model(MODEL_PATH)

# Define the class labels (adjust according to your dataset)
class_labels = ["Normal", "Pneumonia"]

# ===============================
# 2. Streamlit UI
# ===============================
st.title("🩻 COVID-19 X-ray Classifier")
st.write("Upload a chest X-ray image and the model will predict whether it is Normal or Pneumonia.")

# ===============================
# 3. Upload image
# ===============================
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    # ===============================
    # 4. Preprocess the image
    # ===============================
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # ===============================
    # 5. Prediction
    # ===============================
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # ===============================
    # 6. Show result
    # ===============================
    st.subheader("🔍 Prediction Result:")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
