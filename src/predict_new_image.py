import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = r"D:\pneumonia_detection\models\pneumonia_model.h5"  # path to your trained model
NEW_IMAGE_PATH = r"D:\pneumonia_detection\data\new_images\myxray.jpeg"  # path to the new X-ray

# -------------------------------
# Check if files exist
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

if not os.path.exists(NEW_IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found: {NEW_IMAGE_PATH}")

# -------------------------------
# Load model
# -------------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# Load and preprocess image
# -------------------------------
# Assuming your model expects 224x224 RGB images
img = Image.open(NEW_IMAGE_PATH).convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0  # normalize to [0,1]
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# -------------------------------
# Make prediction
# -------------------------------
pred = model.predict(img_array)
# Assuming binary classification: 0 = Normal, 1 = Pneumonia
pred_class = "Pneumonia" if pred[0][0] > 0.5 else "Normal"

print(f"✅ Prediction: {pred_class}")
