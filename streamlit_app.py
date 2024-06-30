import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np

# Assuming your pre-trained model is saved in HDF5 format
try:
    model = tf.keras.models.load_model('my_40_model.h5')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Set model to None to prevent prediction attempts

st.title("Image Classification App")
st.write("Upload an image for classification:")
def preprocess_image(image, target_size=(224, 224)):
  image = image.resize(target_size)
  image = np.array(image)  # Normalize between 0 and 1
  image = image.astype(np.float32)  # Convert to float32 if needed
  image = image.reshape((1, 224, 224, 3))  # Add batch dimension
  return image

uploaded_file = st.file_uploader("Choose an Image...", type="")

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded Image', use_column_width=True)
  
  if model is not None:
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)
        # Interpret the prediction
    st.write("Prediction:", prediction)
    if prediction > 0.0500 :  # Adjust threshold based on your model's output
        st.success("Predicted class: Positive")
    else:
        st.warning("Predicted class: Negative")

else:
    st.info("Please upload an image to start classification.")