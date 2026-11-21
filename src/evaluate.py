import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import os
import numpy as np

# Paths
MODEL_PATH = r"D:\pneumonia_detection\outputs\models\pneumonia_model.h5"
TEST_DIR = r"D:\pneumonia_detection\data\raw\chest_xray\test"  # Adjust if different

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Test data generator (no augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important for correct label ordering
)

# Predict for all test images
preds = model.predict(test_generator)
pred_labels = (preds > 0.5).astype(int).flatten()

# True labels
true_labels = test_generator.classes

# Class names
class_names = list(test_generator.class_indices.keys())

# Classification report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("\n📝 Classification Report:\n")
print(report)
