import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Step 1: Data Loading and Preprocessing
def load_data(folder_path):
    X = []
    y = []
    for label in ['Anemic', 'Non-Anemic']:
        label_folder = os.path.join(folder_path, label)
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            img = image.load_img(img_path, target_size=(224, 224))  # Resize images to 224x224
            img_array = image.img_to_array(img)
            X.append(img_array)
            y.append(1 if label == 'Anemic' else 0)  # Assign 1 for Anemic and 0 for Non-Anemic
    return np.array(X), np.array(y)

X, y = load_data("C:\\Users\\Chiraag\\Desktop\\Final year\\Data\\Fingernails")

# Step 2: Feature Extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
features = Model(inputs=base_model.input, outputs=x).predict(X)

# Step 3: Model Creation
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

input_tensor = Input(shape=(features.shape[1],))
top_model = Dense(256, activation='relu')(input_tensor)
output_tensor = Dense(1, activation='sigmoid')(top_model)

model = Model(inputs=input_tensor, outputs=output_tensor)

# Step 4: Model Training
# Compile the model with Adam optimizer and specify the learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Convert test_accuracy to percentage
test_accuracy_percentage = test_accuracy * 100
print("Test Accuracy (%):", test_accuracy_percentage)

# Step 6: Saving the Model
model.save('anemia_detection_model.h5')

import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
