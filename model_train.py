import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

anemiapath = "C:\\Users\\Chiraag\\Desktop\\Final year\\Data\\CP-AnemiC dataset\\Anemic"
nonanemiapath = "C:\\Users\\Chiraag\\Desktop\\Final year\\Data\\CP-AnemiC dataset\\Non-anemic"

anemia_image_paths = [os.path.join(anemiapath, filename) for filename in tqdm(os.listdir(anemiapath))]
anemia_labels = [1] * len(anemia_image_paths)

non_anemia_image_paths = [os.path.join(nonanemiapath, filename) for filename in tqdm(os.listdir(nonanemiapath))]
non_anemia_labels = [0] * len(non_anemia_image_paths)

image_paths = anemia_image_paths + non_anemia_image_paths
labels = anemia_labels + non_anemia_labels

df = pd.DataFrame({"Image": image_paths, "Label": labels})

x_train, x_test = train_test_split(df, test_size=0.2, random_state=42)

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image)
        img = img.resize((224,224), Image.LANCZOS)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),224,224,3)
    return features

x_train_features = extract_features(x_train['Image'])
x_train_features = x_train_features / 255.0  # normalization
y_train = np.array(x_train['Label'])

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_features, y_train, batch_size=32, epochs=40, validation_split=0.2)

# Save the model
model.save("anemia_detection_model_eye.h5")
