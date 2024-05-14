import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def anemia_detection_eye(img_path):
    model = load_model("eye_anemia_detection.h5")

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    threshold = 0.5
    prediction_classes = 1 if prediction[0][0] > threshold else 0

    return "Anemia Detected" if prediction_classes == 1 else "No Anemia Detected"


def main():
    st.title("Eye-Anemia Detection")

    input_image_path = "C:\\Users\\Chiraag\\Desktop\\Final year\\Data\\CP-AnemiC dataset\\Non-anemic\\Image_067.png"

    prediction = anemia_detection_eye(input_image_path)

    if prediction == "No Anemia Detected":
        st.write("No Anemia Detected")
    elif prediction == "Anemia Detected":
        st.write("Anemia Detected")

if __name__ == "__main__":
    main()
