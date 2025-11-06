import pandas as pd
import os
import shutil
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
dataset_path = "dataset"
image_dir1 = os.path.join(dataset_path, "HAM10000_images_part_1")
image_dir2 = os.path.join(dataset_path, "HAM10000_images_part_2")
metadata_path = os.path.join(dataset_path, "HAM10000_metadata.csv")
combined_dir = os.path.join(dataset_path, "all_images")

# Combine images into one folder
os.makedirs(combined_dir, exist_ok=True)

for folder in [image_dir1, image_dir2]:
    for img_file in os.listdir(folder):
        shutil.copy(os.path.join(folder, img_file), combined_dir)

print("✅ Images combined!")

# Load metadata
df = pd.read_csv(metadata_path)
print("✅ Metadata loaded. Shape:", df.shape)

# Image processing
img_size = 64
X = []
y = []

label_encoder = LabelEncoder()
df['dx'] = label_encoder.fit_transform(df['dx'])

for index, row in df.iterrows():
    img_id = row['image_id'] + '.jpg'
    img_path = os.path.join(combined_dir, img_id)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(row['dx'])

X = np.array(X)
y = np.array(y)

# Normalize
X = X / 255.0

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data ready! Training samples:", len(X_train))
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("skin_disease_model.h5")
print("✅ Model saved.")