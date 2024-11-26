# scripts/windows/info_window.py

from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QTextBrowser
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QPropertyAnimation, QSize, QEasingCurve

class InfoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project Information")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon('assets/icons/info.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Info content
        info_browser = QTextBrowser()
        info_browser.setStyleSheet("font-size: 14px; padding: 10px;")
        info_browser.setHtml(self.get_project_info())
        info_browser.setOpenExternalLinks(True)

        back_button = QPushButton("Back to Main Menu")
        back_button.setIcon(QIcon('assets/icons/back.png'))
        back_button.setIconSize(QSize(24, 24))
        back_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        back_button.clicked.connect(self.close)

        layout.addWidget(info_browser)
        layout.addWidget(back_button, alignment=Qt.AlignRight)
        central_widget.setLayout(layout)

        # Slide-in animation
        self.slide_in_animation = QPropertyAnimation(self, b"geometry")
        self.slide_in_animation.setDuration(500)
        self.slide_in_animation.setStartValue(self.geometry().translated(self.width(), 0))
        self.slide_in_animation.setEndValue(self.geometry())
        self.slide_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.slide_in_animation.start()

    def get_project_info(self):
        info = """
        <h2>Facial Expression Recognition Project</h2>
        <p>
            This project utilizes deep learning techniques to recognize facial expressions in real-time.
            It employs a Convolutional Neural Network (CNN) model built with TensorFlow and Keras.
        </p>

        <h3>Project Goals:</h3>
        <p>
            The goal is to accurately detect and classify human emotions from facial expressions captured via webcam.
            The application provides real-time analysis and feedback to the user, enabling both research and practical applications.
        </p>

        <h3>Dataset Description:</h3>
        <p>
            The model is trained on the 
            <a href='https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset'>Face Expression Recognition Dataset</a> 
            from Kaggle. This dataset contains grayscale images categorized into seven emotions:
            Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
            Each image is resized to 48x48 pixels, ensuring optimal input for the CNN.
        </p>

        <h3>Examples of Emotions:</h3>
        <p>Below are examples of the seven emotions from the dataset:</p>
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 20px; margin-top: 20px;">
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/angry_example.jpg" alt="Angry" width="120" />
                <p><b>Angry</b></p>
            </div>
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/disgust_example.jpg" alt="Disgust" width="120" />
                <p><b>Disgust</b></p>
            </div>
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/fear_example.jpg" alt="Fear" width="120" />
                <p><b>Fear</b></p>
            </div>
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/happy_example.jpg" alt="Happy" width="120" />
                <p><b>Happy</b></p>
            </div>
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/neutral_example.jpg" alt="Neutral" width="120" />
                <p><b>Neutral</b></p>
            </div>
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/sad_example.jpg" alt="Sad" width="120" />
                <p><b>Sad</b></p>
            </div>
            <div style="text-align: center;">
                <img src="assets/images/emotion_examples/surprise_example.jpg" alt="Surprise" width="120" />
                <p><b>Surprise</b></p>
            </div>
        </div>

        <h3>Features and Labels:</h3>
        <p>
            - <b>Features:</b> Pixel values of grayscale images (numerical data).<br>
            - <b>Labels:</b> Emotion categories (classification problem).
        </p>

        <h3>Data Splitting:</h3>
        <p>
            The dataset is split into training, validation, and test sets using a 70/15/15 ratio. 
            This ensures that the model can be effectively trained, fine-tuned, and evaluated.
        </p>

        <h3>Model Development:</h3>
        <p>
            - <b>Softmax Regression:</b> Implemented as a baseline but found insufficient for complex image data.<br>
            - <b>Convolutional Neural Network (CNN):</b> Used to capture spatial hierarchies and improve accuracy.
        </p>

        <h3>Model Evaluation:</h3>
        <p>
            The CNN achieved higher accuracy compared to the baseline models. Metrics used include:
            <ul>
                <li><b>Accuracy:</b> Percentage of correctly classified images.</li>
                <li><b>Precision:</b> Measure of true positive results for each emotion category.</li>
                <li><b>Recall:</b> Measure of the model's ability to identify relevant instances.</li>
            </ul>
            Confusion matrix analysis provided insights into misclassifications and areas for improvement.
        </p>

        <h3>How It Works:</h3>
        <p>
            This system follows a structured pipeline:
        </p>
        <ul>
            <li><b>Dataset Download:</b> The <code>dataset_download.py</code> script automates retrieving and organizing the dataset.</li>
            <li><b>Data Preparation:</b> The <code>prepare_dataset.py</code> script cleans, validates, and splits the dataset into train/validation/test sets.</li>
            <li><b>Model Training:</b> The <code>train_model.py</code> script builds a CNN and trains it using augmented data to improve robustness.</li>
            <li><b>Evaluation:</b> The <code>evaluate_model.py</code> script analyzes performance metrics and provides diagnostic tools like confusion matrices.</li>
            <li><b>Real-Time Detection:</b> The GUI integrates the trained model, allowing users to perform live emotion detection through webcam feeds or static images.</li>
        </ul>

        <h3>Code:</h3>
<ul>
    <li>
        <b>dataset_download.py:</b> 
        This script downloads the Face Expression Recognition Dataset directly from Kaggle. 
        It ensures reproducibility by automating the download and extraction process, saving the dataset in a structured directory.
        <ul>
            <li><b>Purpose:</b> Simplifies dataset acquisition and prepares it for further preprocessing.</li>
            <li><b>Output:</b> A dataset folder containing raw images categorized into subdirectories based on emotions.</li>
        </ul>
    </li>
    <li>
        <b>prepare_dataset.py:</b> 
        Handles dataset cleaning and splitting. It processes raw images, applies data validation, and splits the dataset into training, validation, and test sets.
        <ul>
            <li><b>Purpose:</b> Ensures balanced distribution of emotions across splits for effective training and evaluation.</li>
            <li><b>Functionality:</b> Renames files, groups them into new directories, and verifies the presence of required data.</li>
            <li><b>Output:</b> Three folders (<code>new_train</code>, <code>new_validation</code>, and <code>new_test</code>) ready for training and evaluation.</li>
        </ul>
    </li>
    <li>
        <b>train_model.py:</b> 
        Constructs a Convolutional Neural Network (CNN) model and trains it on the preprocessed dataset.
        <ul>
            <li><b>Architecture:</b> The CNN includes multiple convolutional, pooling, and dropout layers for feature extraction and regularization.</li>
            <li><b>Purpose:</b> Captures spatial features in grayscale images to classify emotions accurately.</li>
            <li><b>Training Process:</b> Utilizes data augmentation (horizontal flips, scaling) to enhance generalization and trains over multiple epochs.</li>
            <li><b>Output:</b> A trained model saved as <code>models/expression_model.h5</code> for further use.</li>
        </ul>
    </li>
    <li>
        <b>evaluate_model.py:</b> 
        Evaluates the trained model on the test dataset to assess its performance and identify strengths and weaknesses.
        <ul>
            <li><b>Metrics:</b> Generates accuracy, precision, recall, and F1-score for each emotion category.</li>
            <li><b>Confusion Matrix:</b> Provides a detailed view of misclassifications, enabling fine-tuning.</li>
            <li><b>Purpose:</b> Validates the model's robustness and highlights areas for improvement.</li>
            <li><b>Output:</b> Evaluation reports printed in the console, including the classification report and confusion matrix.</li>
        </ul>
    </li>
    <li>
        <b>main_gui.py:</b> 
        Integrates the trained model into a PySide6-based graphical interface. It allows users to interact with the model and analyze emotions in real time.
        <ul>
            <li><b>Features:</b> Enables live emotion detection through a webcam or image uploads.</li>
            <li><b>Purpose:</b> Provides an intuitive interface for non-technical users to leverage the model.</li>
            <li><b>Navigation:</b> Includes buttons to access different project components like live detection, saved results, and project information.</li>
            <li><b>Output:</b> Real-time emotion predictions displayed in the GUI.</li>
        </ul>
    </li>
</ul>

        <h3>Installation Instructions:</h3>
        <ol>
            <li>Ensure Python 3 is installed.</li>
            <li>Install the required libraries using <code>pip install -r requirements.txt</code>.</li>
            <li>Set up Kaggle API credentials for dataset downloads.</li>
            <li>Run <code>dataset_download.py</code> to fetch the dataset.</li>
            <li>Use <code>prepare_dataset.py</code> to preprocess the data.</li>
            <li>Train the model using <code>train_model.py</code>.</li>
            <li>Launch the GUI with <code>python scripts/main_gui.py</code>.</li>
        </ol>

        <h3>Additional Resources:</h3>
        <p>
            - <a href='https://www.tensorflow.org/'>TensorFlow Documentation</a><br>
            - <a href='https://opencv.org/'>OpenCV Documentation</a><br>
            - <a href='https://doc.qt.io/qtforpython/'>PySide6 Documentation</a>
        </p>
        """
        return info

