import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import local_binary_pattern
import joblib
import matplotlib.pyplot as plt

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

# Directory paths
anemic_dir = "C:\\Users\\Chiraag\\Desktop\\Palm\\anemic"
healthy_dir = "C:\\Users\\Chiraag\\Desktop\\Palm\\non-anemic"

X = []
y = []

# Extract features from anemic images
for img_name in os.listdir(anemic_dir):
    img_path = os.path.join(anemic_dir, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        features = extract_features(image)
        X.append(features)
        y.append(1)  # Anemic class label

# Extract features from healthy images
for img_name in os.listdir(healthy_dir):
    img_path = os.path.join(healthy_dir, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        features = extract_features(image)
        X.append(features)
        y.append(0)  # Healthy class label

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Varying number of trees in the Random Forest
n_estimators_list = [10, 50, 100, 200, 300]
train_accuracy = []
cv_accuracy = []

for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_accuracy.append(np.mean(cv_scores))

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Training accuracy
    train_pred = model.predict(X_train)
    train_accuracy.append(accuracy_score(y_train, train_pred))

# Plotting the accuracy graph
plt.plot(n_estimators_list, train_accuracy, label='Train Accuracy')
plt.plot(n_estimators_list, cv_accuracy, label='Cross-Validation Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees in Random Forest')
plt.legend()
plt.show()
