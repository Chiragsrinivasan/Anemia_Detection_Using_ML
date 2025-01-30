import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load the saved model
model = load_model('anemia_detection_model.h5')

# Load the VGG16 model without the fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the VGG16 model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding a global average pooling layer after the base model
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(1, activation='sigmoid')
model = Model(inputs=base_model.input, outputs=prediction_layer(global_average_layer(base_model.output)))

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to make prediction
def predict_anemia(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction[0][0]  # Return the predicted probability of anemia

# Path to the image file
img_path = "C:\\Users\\Chiraag\\Desktop\\images.jpeg"

# Make prediction
prediction = predict_anemia(img_path)

# Print the prediction result
if prediction > 0.5:
    print(f"Anemia Probability: {prediction:.4f} (Anemic)")
else:
    print(f"Anemia Probability: {prediction:.4f} (Non-Anemic)")
