# scripts/windows/live_detection_window.py

import os
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QMessageBox
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QSize, QEasingCurve
from tensorflow.keras.models import load_model
from scripts.windows.results_window import ResultsWindow


class LiveDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Emotion Detection")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon('assets/icons/live.png'))

        self.model = load_model("models/expression_model.h5")
        self.emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                             3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        self.color_dict = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 255, 0),
            'Fear': (255, 0, 0),
            'Happy': (255, 255, 0),
            'Neutral': (255, 255, 255),
            'Sad': (0, 0, 128),
            'Surprise': (255, 165, 0)
        }
        self.captured_images = []
        self.previous_emotion = None
        self.session_folder = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()

        # Instructions label
        instructions_label = QLabel("You are about to start live emotion detection. Click the button below to proceed.")
        instructions_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        instructions_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(instructions_label, alignment=Qt.AlignCenter)

        # Emotion examples
        self.emotion_examples = QLabel()
        examples_pixmap = QPixmap('assets/images/emotion_examples/emotion_examples.jpg').scaled(600, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.emotion_examples.setPixmap(examples_pixmap)
        self.emotion_examples.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.emotion_examples)

        # Start button
        self.start_button = QPushButton("Start Live Detection")
        self.start_button.setIcon(QIcon('assets/icons/live.png'))
        self.start_button.setIconSize(QSize(24, 24))
        self.start_button.setFixedHeight(50)
        self.start_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 10px;
                background-color: #FF5722;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        self.start_button.clicked.connect(self.start_live_detection)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # Back button
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
        self.layout.addWidget(back_button, alignment=Qt.AlignRight)

        central_widget.setLayout(self.layout)

        # Apply fade-in animation
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.start()

    def start_live_detection(self):
        # Remove instructions and button
        self.layout.removeWidget(self.start_button)
        self.start_button.deleteLater()
        self.layout.removeWidget(self.emotion_examples)
        self.emotion_examples.deleteLater()

        # Add video frame label
        self.image_label = QLabel()
        self.image_label.setFixedSize(800, 500)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.image_label)

        # End button
        self.end_button = QPushButton("End Live")
        self.end_button.setIcon(QIcon('assets/icons/stop.png'))
        self.end_button.setIconSize(QSize(24, 24))
        self.end_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 8px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.end_button.clicked.connect(self.end_live)
        self.layout.addWidget(self.end_button, alignment=Qt.AlignRight)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not access the webcam.")
            self.close()
            return

        # Create session folder to save images
        self.create_session_folder()

        # Start timer to read frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def create_session_folder(self):
        if not os.path.exists("saved_images"):
            os.makedirs("saved_images")
        self.session_folder = os.path.join("saved_images", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.session_folder, exist_ok=True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        emotion = None

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            prediction = self.model.predict(roi_gray)
            maxindex = int(np.argmax(prediction))
            emotion = self.emotion_dict[maxindex]

            color = self.color_dict.get(emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Save image if emotion changes
            if emotion != self.previous_emotion:
                self.save_captured_image(frame, emotion)
                self.previous_emotion = emotion

        # Convert the image to Qt format
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def save_captured_image(self, frame, emotion):
        emotion_folder = os.path.join(self.session_folder, emotion)
        os.makedirs(emotion_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = os.path.join(emotion_folder, f"{timestamp}.png")
        cv2.imwrite(image_path, frame)
        self.captured_images.append((image_path, emotion))

    def end_live(self):
        self.timer.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()
        self.results_window = ResultsWindow(self.captured_images, self.session_folder)
        self.results_window.show()
