import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ------------------------------
# Paths
# ------------------------------
TRAIN_DIR = r"D:\pneumonia_detection\data\train"
VAL_DIR = r"D:\pneumonia_detection\data\val"
MODEL_SAVE_PATH = r"D:\pneumonia_detection\outputs\models\pneumonia_model.h5"

# ------------------------------
# Image parameters
# ------------------------------
IMG_SIZE = (224, 224)  # match this with predict.py
BATCH_SIZE = 32

# ------------------------------
# Data generators with augmentation
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ------------------------------
# Build CNN model
# ------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# Train the model
# ------------------------------
EPOCHS = 10

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ------------------------------
# Save the model
# ------------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at {MODEL_SAVE_PATH}")
