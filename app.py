import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import pandas as pd


# Loading the model
model = load_model('cat_vs_dog_model.h5')

# Define the labels for binary classification
labels = ['Cat', 'Dog']

st.title("Cat vs. Dog Classifier")

# Image Uploader
image = st.file_uploader("Upload an image of a cat or dog", type=[
                         "jpg", "png", "jpeg"])

if image is not None:
    try:
        img = Image.open(image)
        img = img.convert("RGB")  # Convert to RGB format
        img = img.resize((256, 256))
        img = np.array(img)
        img = img / 255.0  # Normalizing the pixels value

        # Prediction Part
        prediction = model.predict(np.expand_dims(img, axis=0))
        class_index = int(np.round(prediction[0][0]))

        # Result part

        st.image(img, caption=f"Uploaded Image", use_column_width=True)
        st.balloons()
        st.subheader(f"The image is of {labels[class_index]}")
    except Exception as e:
        st.write("Error processing the uploaded image.")
        st.write(e)
