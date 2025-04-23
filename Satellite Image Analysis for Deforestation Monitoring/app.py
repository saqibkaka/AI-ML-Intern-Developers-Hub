import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# --- Step 1: Load and Preprocess Images ---
data_dir = "D:/Internships/New folder/archive (3)"  # Path to your dataset folder
img_size = 128  # Resize all images to 128x128
categories = ["deforestation", "no_deforestation"]

data = []
labels = []

# Load the images and preprocess them
for label, category in enumerate(categories):
    path = os.path.join(data_dir, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
        except:
            print(f"Failed to load image: {img_path}")

data = np.array(data) / 255.0  # Normalize images
labels = np.array(labels)

# --- Step 2: Load the Pre-trained CNN Model ---
model = load_model("deforestation_cnn_model.h5")  # Load your trained CNN model

# --- Step 3: Web App UI ---
st.set_page_config(page_title="Deforestation Detector")
st.title("🌍 Deforestation Detection App")
st.write("Upload a satellite image to detect deforestation.")

uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img_array = np.array(image)

    # Check if the image has 4 channels (RGBA) and convert it to RGB
    if img_array.shape[-1] == 4:  # if the image has 4 channels (RGBA)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)  # Convert to RGB

    # Resize and normalize the image to match the model input size
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # --- Step 4: Make Prediction ---
    prediction = model.predict(img_array)[0][0]
    label = "No Deforestation" if prediction > 0.5 else "Deforestation"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display result
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    if label == "Deforestation":
        st.error("⚠️ Deforestation detected!")
    else:
        st.success("✅ No deforestation detected.")
