

# **Facial Expression Recognition Project**

## **Author:** Roni Michaeli

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation Instructions](#installation-instructions)
5. [Running the Application](#running-the-application)
6. [How to Use the Application](#how-to-use-the-application)
7. [Project Components and Code Explanation](#project-components-and-code-explanation)
   - [Dataset Download and Preparation](#dataset-download-and-preparation)
     - [`dataset_download.py`](#dataset_downloadpy)
     - [`prepare_dataset.py`](#prepare_datasetpy)
   - [Model Training and Evaluation](#model-training-and-evaluation)
     - [`train_model.py`](#train_modelpy)
     - [`evaluate_model.py`](#evaluate_modelpy)
   - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
     - [`main_gui.py`](#main_guipy)
     - [GUI Modules](#gui-modules)
8. [Understanding the Model](#understanding-the-model)
9. [Conclusion](#conclusion)
10. [Additional Resources](#additional-resources)

---

## **Project Overview**

This project utilizes deep learning techniques to recognize facial expressions in real-time. It employs a Convolutional Neural Network (CNN) model built with TensorFlow and Keras, integrated into a user-friendly graphical interface using PySide6 (Qt for Python).

### **Project Goals**

- **Accurate Emotion Detection**: Detect and classify human emotions from facial expressions captured via webcam.
- **Real-Time Analysis**: Provide immediate feedback to the user with live emotion detection.
- **User-Friendly Interface**: Offer an intuitive GUI for users to interact with the model without needing technical expertise.
- **Educational Value**: Serve as a comprehensive example of applying deep learning to a practical problem, with detailed explanations of each component.

---

## **Project Structure**

```plaintext
face_recognition_model/
├── assets/
│   ├── icons/
│   └── images/
├── data/
│   └── face-expression-recognition-dataset/
├── models/
│   └── expression_model.h5
├── saved_images/
│   ├── session_YYYYMMDD_HHMMSS/
│   └── correct_classifications/
├── scripts/
│   ├── dataset_download.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── main_gui.py
│   ├── windows/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── info_window.py
│   │   ├── live_detection_window.py
│   │   ├── results_window.py
│   │   ├── gallery_window.py
│   │   └── emotion_gallery_window.py
│   └── utils/
│       ├── __init__.py
│       └── utils.py
├── requirements.txt
└── README.md
```

- **`assets/`**: Contains icons and images used in the GUI.
- **`data/`**: Stores the dataset downloaded from Kaggle.
- **`models/`**: Holds the trained model file.
- **`saved_images/`**: Contains images captured during live detection sessions.
- **`scripts/`**: All Python scripts for dataset handling, model training, evaluation, and GUI.
- **`requirements.txt`**: Lists all Python dependencies.
- **`README.md`**: Detailed documentation (this file).

---

## **Prerequisites**

- **Python 3.7 or later**
- **Virtual Environment** (recommended)
- **Kaggle Account** with API credentials
- **Internet Connection**

---

## **Installation Instructions**

### **1. Clone the Repository or Download the Project**

Navigate to your desired directory and clone the repository or download the project files.

```bash
git clone https://github.com/yourusername/face_recognition_model.git
cd face_recognition_model
```

### **2. Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Required Packages**

Install all necessary Python packages using `pip`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note for macOS with M1 Chip:**

If you're using an M1 Mac, install TensorFlow using:

```bash
pip install tensorflow-macos tensorflow-metal
```

### **4. Set Up Kaggle API Credentials**

1. **Create a Kaggle Account**: If you don't have one, sign up at [kaggle.com](https://www.kaggle.com/).
2. **Generate API Credentials**:
   - Go to your Kaggle account settings: `https://www.kaggle.com/<your-username>/account`
   - Scroll to the **API** section and click **Create New API Token**.
   - This downloads a `kaggle.json` file containing your API credentials.
3. **Place `kaggle.json` in the Correct Directory**:
   - On macOS/Linux:

     ```bash
     mkdir ~/.kaggle
     mv /path/to/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

   - On Windows:

     ```bash
     mkdir %USERPROFILE%\.kaggle
     move \path\to\kaggle.json %USERPROFILE%\.kaggle\
     ```

---

## **Running the Application**

### **1. Download the Dataset**

Run the `dataset_download.py` script to download the dataset from Kaggle.

```bash
python scripts/dataset_download.py
```

### **2. Prepare the Dataset**

Run the `prepare_dataset.py` script to preprocess and split the dataset.

```bash
python scripts/prepare_dataset.py
```

### **3. Train the Model**

Train the CNN model using the `train_model.py` script.

```bash
python scripts/train_model.py
```

- **Note**: Training may take some time depending on your hardware.

### **4. Evaluate the Model (Optional)**

Evaluate the trained model using the `evaluate_model.py` script.

```bash
python scripts/evaluate_model.py
```

### **5. Run the GUI Application**

Launch the graphical user interface.

```bash
python scripts/main_gui.py
```

---

## **How to Use the Application**

### **Main Menu**

Upon launching the application, you'll see the main window with three options:

1. **Information About the Project**: View detailed information and explanations about the project.
2. **Use the Model**: Start live emotion detection using your webcam.
3. **Images the Model Got Right**: Browse images that the model correctly classified during live sessions.

### **1. Information About the Project**

- **Description**: Provides an in-depth explanation of the project, including goals, dataset details, model architecture, and more.
- **Navigation**: Use the **Back to Main Menu** button to return to the main screen.

### **2. Use the Model**

- **Instructions**: Before starting, you'll see instructions and examples of emotions.
- **Start Live Detection**: Click the **Start Live Detection** button to begin.
- **Live Emotion Detection**:
  - The application accesses your webcam.
  - Detected faces are processed, and emotions are displayed in real-time.
  - The application saves images when a new emotion is detected.
- **End Live Session**: Click the **End Live** button to finish.
- **Results Review**:
  - After ending the session, you can review the captured images.
  - For each image, indicate whether the model's classification was correct or incorrect.
  - This feedback is used to calculate the model's accuracy.

### **3. Images the Model Got Right**

- **Browse Emotions**: View a list of emotions with correctly classified images.
- **View Images**: Select an emotion to see the images.
- **Navigation**: Use **Previous**, **Next**, and **Back** buttons to navigate through images and return to the main menu.

---

## **Project Components and Code Explanation**

### **Dataset Download and Preparation**

#### **`dataset_download.py`**

**Purpose**: Automates the download of the Face Expression Recognition Dataset from Kaggle.

**Key Functions:**

- **`download_dataset()`**:
  - Authenticates with the Kaggle API.
  - Downloads and unzips the dataset into the `data/` directory.

**Code Explanation:**

```python
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset = "jonathanoheix/face-expression-recognition-dataset"
    output_dir = "data/face-expression-recognition-dataset/"
    os.makedirs(output_dir, exist_ok=True)
    api.dataset_download_files(dataset, path=output_dir, unzip=True)

if __name__ == "__main__":
    download_dataset()
```

- **`KaggleApi()`**: Initializes the Kaggle API client.
- **`authenticate()`**: Authenticates using the `kaggle.json` credentials.
- **`dataset_download_files()`**: Downloads and unzips the dataset.

#### **`prepare_dataset.py`**

**Purpose**: Preprocesses the dataset by combining training and validation sets, then splitting into new training, validation, and test sets.

**Key Functions:**

- **`prepare_dataset()`**:
  - Combines original training and validation data.
  - Splits combined data into new sets using a 70/15/15 ratio.
  - Organizes data into `new_train/`, `new_validation/`, and `test/` directories.

**Code Explanation:**

```python
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_path="data/face-expression-recognition-dataset/images/images", split_ratios=(0.7, 0.15, 0.15)):
    # Combine train and validation data
    # Split into new train, validation, and test sets
    # Organize data into new directories
    # ... (code continues)
```

- **`train_test_split()`**: Splits data into training and testing subsets.
- **Directory Operations**: Uses `os` and `shutil` to manipulate directories and files.

### **Model Training and Evaluation**

#### **`train_model.py`**

**Purpose**: Builds and trains the CNN model using the prepared dataset.

**Key Functions:**

- **`build_model(input_shape)`**:
  - Constructs the CNN architecture with convolutional, pooling, dropout, and dense layers.
  - Uses ReLU activation and softmax for the output layer.

- **`train_model()`**:
  - Sets up data generators for training and validation data.
  - Compiles the model with an optimizer, loss function, and metrics.
  - Trains the model over specified epochs.
  - Saves the trained model to `models/expression_model.h5`.

**Code Explanation:**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    # Add convolutional layers
    # Add pooling and dropout layers
    # Flatten and add dense layers
    # ... (code continues)
    return model

def train_model():
    # Set up directories
    # Create data generators
    # Compile and train the model
    # Save the model
    # ... (code continues)
```

- **Model Architecture**:
  - **Convolutional Layers**: Extract features from images.
  - **Pooling Layers**: Reduce spatial dimensions.
  - **Dropout Layers**: Prevent overfitting.
  - **Dense Layers**: Perform classification.

- **Data Augmentation**:
  - Uses `ImageDataGenerator` to rescale images and apply transformations like horizontal flips.

#### **`evaluate_model.py`**

**Purpose**: Evaluates the trained model on the test set and outputs performance metrics.

**Key Functions:**

- **`evaluate_model()`**:
  - Loads the trained model.
  - Generates predictions on the test data.
  - Computes classification report and confusion matrix.

**Code Explanation:**

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def evaluate_model():
    # Load model and test data
    # Predict on test data
    # Generate evaluation metrics
    # ... (code continues)
```

- **Metrics**:
  - **Accuracy**: Overall correctness.
  - **Precision**: Correct positive predictions.
  - **Recall**: Ability to find all positive instances.
  - **Confusion Matrix**: Detailed breakdown of predictions.

### **Graphical User Interface (GUI)**

#### **`main_gui.py`**

**Purpose**: Entry point for the GUI application.

**Key Functions:**

- **`main()`**:
  - Sets up the application and displays the main window.

**Code Explanation:**

```python
import sys
import os
from PySide6.QtWidgets import QApplication

# Adjust sys.path to include project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.windows.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

- **`QApplication`**: Initializes the Qt application.
- **`MainWindow`**: The main GUI window.

#### **GUI Modules**

All GUI components are organized in the `scripts/windows/` directory.

- **`main_window.py`**: The main menu window with navigation to other sections.
- **`info_window.py`**: Displays detailed project information.
- **`live_detection_window.py`**: Handles live emotion detection using the webcam.
- **`results_window.py`**: Allows users to review and validate captured images.
- **`gallery_window.py`**: Provides a gallery of correctly classified images.
- **`emotion_gallery_window.py`**: Displays images for a specific emotion.

**Example Code Explanation for `main_window.py`:**

```python
from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt, QPropertyAnimation, QSize, QEasingCurve

from scripts.windows.info_window import InfoWindow
from scripts.windows.live_detection_window import LiveDetectionWindow
from scripts.windows.gallery_window import GalleryWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up window properties
        # Create layout and widgets
        # Connect buttons to functions
        # Apply animations
        # ... (code continues)
```

- **Widgets Used**:
  - **`QMainWindow`**: Main window class.
  - **`QPushButton`**: Buttons for navigation.
  - **`QLabel`**: Display images and text.
  - **`QVBoxLayout`**: Arrange widgets vertically.

- **GUI Features**:
  - **Icons and Images**: Enhances visual appeal.
  - **Animations**: Provides smooth transitions.
  - **Styling**: Uses stylesheets for consistent design.

---

## **Understanding the Model**

### **Model Architecture**

The CNN model consists of:

- **Input Layer**: Accepts images of shape `(48, 48, 1)` (grayscale images).
- **Convolutional Layers**: Extract features using filters.
- **MaxPooling Layers**: Reduce spatial dimensions.
- **Dropout Layers**: Prevent overfitting by randomly disabling neurons.
- **Flatten Layer**: Converts 2D feature maps to 1D feature vectors.
- **Dense Layers**: Perform classification using fully connected layers.
- **Output Layer**: Uses softmax activation to output probabilities for each emotion class.

### **Training Process**

- **Data Augmentation**: Enhances the dataset by applying transformations.
- **Compilation**: The model is compiled with the Adam optimizer and categorical cross-entropy loss.
- **Epochs**: The model is trained over multiple epochs to improve accuracy.
- **Validation**: Uses a validation set to monitor overfitting.

---

## **Conclusion**

This project demonstrates how deep learning can be applied to real-time facial expression recognition. By integrating a CNN model with a user-friendly GUI, it offers both practical utility and educational value.

---

## **Additional Resources**

- **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **OpenCV Documentation**: [https://opencv.org/](https://opencv.org/)
- **PySide6 Documentation**: [https://doc.qt.io/qtforpython/](https://doc.qt.io/qtforpython/)
- **Kaggle Dataset**: [https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

---

**Author**: Roni Michaeli

If you have any questions or need further assistance, feel free to reach out.

---

## **Images and Assets**

All images and icons used in the GUI are stored in the `assets/` directory. Ensure that this directory is correctly placed in the project root to avoid any issues with missing assets.

---

## **Final Notes**

- **Ensure Correct Paths**: Double-check that all file paths in the code match your project's directory structure.
- **Permissions on macOS**: If you're on macOS and encounter issues accessing the webcam, you may need to grant permissions in `System Preferences` > `Security & Privacy` > `Privacy` > `Camera`.
- **Virtual Environment Activation**: Always activate your virtual environment before running scripts to ensure all dependencies are available.
- **Python Version**: Use Python 3.7 or later for compatibility.
