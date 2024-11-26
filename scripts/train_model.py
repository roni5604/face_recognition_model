# scripts/train_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

def train_model():
    # Update the directories to point to new_train and new_validation
    train_dir = "data/face-expression-recognition-dataset/images/images/new_train"
    val_dir = "data/face-expression-recognition-dataset/images/images/new_validation"

    # Verify that the directories exist
    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} does not exist.")
        return
    if not os.path.exists(val_dir):
        print(f"Validation directory {val_dir} does not exist.")
        return

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    # Check if classes are found
    if train_generator.num_classes == 0 or train_generator.samples == 0:
        print("No training data found. Please check the dataset.")
        return
    if val_generator.num_classes == 0 or val_generator.samples == 0:
        print("No validation data found. Please check the dataset.")
        return

    steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
    validation_steps = max(1, val_generator.samples // val_generator.batch_size)

    model = build_model((48, 48, 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    os.makedirs("models", exist_ok=True)
    model.save("models/expression_model.h5")

if __name__ == "__main__":
    train_model()
