# scripts/windows/results_window.py

import os
import shutil
from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt, QPropertyAnimation, QSize
from scripts.utils.utils import calculate_accuracy


class ResultsWindow(QMainWindow):
    def __init__(self, images, session_folder):
        super().__init__()
        self.setWindowTitle("Live Session Results")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon('assets/icons/results.png'))

        self.images = images
        self.session_folder = session_folder
        self.user_feedback = []
        self.current_index = 0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 400)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.emotion_label = QLabel()
        self.emotion_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.emotion_label, alignment=Qt.AlignCenter)

        # Feedback buttons
        feedback_layout = QHBoxLayout()
        correct_button = QPushButton("Correct")
        correct_button.setIcon(QIcon('assets/icons/correct.png'))
        correct_button.setIconSize(QSize(24, 24))
        incorrect_button = QPushButton("Incorrect")
        incorrect_button.setIcon(QIcon('assets/icons/incorrect.png'))
        incorrect_button.setIconSize(QSize(24, 24))
        exit_button = QPushButton("Exit")
        exit_button.setIcon(QIcon('assets/icons/stop.png'))
        exit_button.setIconSize(QSize(24, 24))

        for btn in [correct_button, incorrect_button, exit_button]:
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    padding: 10px;
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
            """)

        correct_button.clicked.connect(self.mark_correct)
        incorrect_button.clicked.connect(self.mark_incorrect)
        exit_button.clicked.connect(self.exit_review)
        feedback_layout.addWidget(correct_button)
        feedback_layout.addWidget(incorrect_button)
        feedback_layout.addWidget(exit_button)
        self.layout.addLayout(feedback_layout)

        central_widget.setLayout(self.layout)
        self.show_image()

        # Apply fade-in animation
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.start()

    def show_image(self):
        if self.current_index < len(self.images):
            image_path, emotion = self.images[self.current_index]
            pixmap = QPixmap(image_path).scaled(600, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.emotion_label.setText(f"Detected Emotion: {emotion}")
        else:
            self.display_analysis()

    def mark_correct(self):
        self.user_feedback.append(True)
        self.save_correct_image()
        self.current_index += 1
        self.show_image()

    def mark_incorrect(self):
        self.user_feedback.append(False)
        self.current_index += 1
        self.show_image()

    def exit_review(self):
        self.display_analysis()

    def save_correct_image(self):
        # Save the correctly classified image for future access
        image_path, emotion = self.images[self.current_index]
        dest_folder = os.path.join("saved_images", "correct_classifications", emotion)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, os.path.basename(image_path))
        shutil.copy(image_path, dest_path)

    def display_analysis(self):
        total = len(self.user_feedback)
        correct = sum(self.user_feedback)
        accuracy = calculate_accuracy(correct, total)

        QMessageBox.information(
            self,
            "Analysis",
            f"Total Images: {total}\nCorrect Classifications: {correct}\nAccuracy: {accuracy:.2f}%"
        )
        self.close()
