import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib


def predict_anemia_status(image_path):
    # Load the trained model from file
    model = joblib.load('palm_anemia_model.pkl')

    # Load the image
    new_image = cv2.imread(image_path)

    if new_image is not None:
        # Extract features from the image
        new_image_features = extract_features(new_image)

        # Predict anemia status using the loaded model
        prediction = model.predict([new_image_features])

        return "Anemia Detected" if prediction == 1 else "Healthy"
    else:
        return "Failed to load the new image."


def extract_features(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color features
    features = []
    for channel in cv2.split(hsv_image):
        features.extend([np.mean(channel), np.std(channel), np.percentile(channel, 25), np.percentile(channel, 75)])

    # Texture features (using Local Binary Pattern)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, 8, 3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    features.extend(hist)

    return features


# Sample usage
new_image_path = input("Enter the path to the image: ")
anemia_status = predict_anemia_status(new_image_path)
print("Anemia Status:", anemia_status)
