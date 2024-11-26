# scripts/evaluate_model.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def evaluate_model():
    test_dir = "data/face-expression-recognition-dataset/images/images/test"
    model_path = "models/expression_model.h5"

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist.")
        return
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )

    if test_generator.num_classes == 0 or test_generator.samples == 0:
        print("No test data found. Please check the dataset.")
        return

    model = load_model(model_path)
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    class_labels = list(test_generator.class_indices.keys())
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
