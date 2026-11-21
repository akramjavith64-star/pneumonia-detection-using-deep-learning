import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------------------
# Paths
# ------------------------------
MODEL_PATH = r"D:\pneumonia_detection\outputs\models\pneumonia_model.h5"
IMG_FOLDER = r"D:\pneumonia_detection\data\new_images"

# ------------------------------
# Load trained model
# ------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------------------
# Get all images in the folder
# ------------------------------
allowed_exts = ['.jpg', '.jpeg', '.png']
img_files = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(tuple(allowed_exts))]

if not img_files:
    raise FileNotFoundError(f"❌ No images found in folder: {IMG_FOLDER}")

# ------------------------------
# Prediction loop
# ------------------------------
IMG_SIZE = (224, 224)  # Must match training size

for img_name in img_files:
    IMG_PATH = os.path.join(IMG_FOLDER, img_name)
    print(f"\n🖼️ Predicting for image: {IMG_PATH}")

    # Load and preprocess image
    img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if label == "PNEUMONIA" else 1 - prediction

    print(f"Prediction: {label} | Confidence: {confidence*100:.2f}%")

    # Display image
    plt.imshow(Image.open(IMG_PATH))
    plt.title(f"{label} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()
