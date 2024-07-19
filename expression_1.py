import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

# Load the model
model = load_model('C:/Users/dhill/VS-Code/dataset/CNN_exopression_Q3.h5')

# Define categories
categories = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to preprocess image
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit app
st.title("Emotion Recognition")

st.write("Upload an image to get the emotion prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the image
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)
    
    # Predict the class
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = categories[predicted_class]
    
    # Display the result
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {predicted_label}")

