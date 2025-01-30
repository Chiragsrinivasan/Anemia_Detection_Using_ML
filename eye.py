import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from keras.preprocessing.image import load_img
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Function to extract features from images
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image)
        img = img.resize((224, 224), Image.LANCZOS)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 224, 224, 3)
    return features

# Path to the dataset folders
anemia_path = 'Dataset/Anemic'  # Replace 'path_to_anemia_folder' with the actual path
non_anemia_path = 'Dataset/Non-Anemic'  # Replace 'path_to_non_anemia_folder' with the actual path

# Collect image paths and labels
anemia_image_paths = [os.path.join(anemia_path, filename) for filename in os.listdir(anemia_path)]
non_anemia_image_paths = [os.path.join(non_anemia_path, filename) for filename in os.listdir(non_anemia_path)]

# Create labels
anemia_labels = [1] * len(anemia_image_paths)
non_anemia_labels = [0] * len(non_anemia_image_paths)

# Combine paths and labels
image_paths = anemia_image_paths + non_anemia_image_paths
labels = anemia_labels + non_anemia_labels

# Create a DataFrame
df = pd.DataFrame({"Image": image_paths, "Label": labels})

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract features and normalize data
x_train = extract_features(train_df['Image'])
x_train = x_train / 255.0
y_train = np.array(train_df['Label'])

# Define CNN architecture
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=40, validation_split=0.2)

# Save the trained model
model.save('anemia_detection_model_eye.h5')


# Evaluate model on test data
x_test = extract_features(test_df['Image'])
x_test = x_test / 255.0
y_test = np.array(test_df['Label'])

# Calculate accuracy
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
