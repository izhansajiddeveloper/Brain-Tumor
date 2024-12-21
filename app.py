import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cotton_disease_inception_model.h5')

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title('Cotton Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    image = preprocess_image(image)
    prediction = model.predict(image)
    
    if prediction[0] > 0.5:
        st.write("The model predicts: Disease Detected")
    else:
        st.write("The model predicts: No Disease Detected")
