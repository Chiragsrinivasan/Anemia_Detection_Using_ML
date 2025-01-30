import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras.models

def anemia_detection(upload_image):
    model = load_model("anemia_detection_model_eye.h5")

    if upload_image is not None:
        img = Image.open(upload_image).resize((224,224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        threshold = 0.5
        prediction_classes = 1 if prediction[0][0] > threshold else 0

        return prediction_classes

st.title("Eye-Anemia Detection")
upload_image = st.file_uploader(label='Upload image for detecting Eye-Anemia', type=['png', 'jpg', 'jpeg'])

if upload_image is not None:
    img = Image.open(upload_image)
    st.image(img, width=400)

    prediction = anemia_detection(upload_image)

    if prediction == 0:
        st.markdown("<h1 style='text-align: left; color: green; font: Times New Roman;'>No Anemia Detected</h1>", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("<h1 style='text-align: left; color: red; font: Times New Roman;'>Anemia Detected</h1>", unsafe_allow_html=True)
