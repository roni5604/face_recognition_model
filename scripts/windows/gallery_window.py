# scripts/windows/gallery_window.py

import os
from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize, Qt, QPropertyAnimation, QEasingCurve
from scripts.windows.emotion_gallery_window import EmotionGalleryWindow


class GalleryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Browse Correctly Classified Images")
        self.setFixedSize(900, 700)
        self.setWindowIcon(QIcon('assets/icons/gallery.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)

        # Title Label
        title_label = QLabel("Model Results - Correctly Classified Images")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin: 20px 0;
            text-align: center;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label)

        # Explanation Text
        explanation_label = QLabel(
            "Below is a list of emotions with images that the trained model identified during live testing. "
            "These results are saved when the model classifies live detections as correct. Use this screen "
            "to review and verify the model's performance for each emotion category."
        )
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet("""
            font-size: 16px;
            color: #555;
            margin: 10px 20px;
        """)
        self.layout.addWidget(explanation_label)

        # Emotion selection list
        self.emotion_list = QListWidget()
        self.emotion_list.setStyleSheet("""
            QListWidget {
                font-size: 18px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: #f0f8ff;
            }
            QListWidget::item {
                margin: 5px;
                border: none;
                padding: 10px;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
            QListWidget::item:selected {
                background-color: #42a5f5;
                color: white;
                border: none;
            }
        """)
        self.refresh_emotion_list()  # Populate the list on initialization
        self.emotion_list.itemClicked.connect(self.open_emotion_gallery)
        self.layout.addWidget(self.emotion_list)

        # Navigation Buttons Layout
        button_layout = QHBoxLayout()

        # Back button
        back_button = QPushButton("Back to Main Menu")
        back_button.setIcon(QIcon('assets/icons/back.png'))
        back_button.setIconSize(QSize(24, 24))
        back_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px;
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
        button_layout.addWidget(back_button)

        # Refresh button
        refresh_button = QPushButton("Refresh List")
        refresh_button.setIcon(QIcon('assets/icons/refresh.png'))
        refresh_button.setIconSize(QSize(24, 24))
        refresh_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px;
                background-color: #FFC107;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #FFA000;
            }
        """)
        refresh_button.clicked.connect(self.refresh_emotion_list)
        button_layout.addWidget(refresh_button)

        self.layout.addLayout(button_layout)

        central_widget.setLayout(self.layout)

        # Slide-in animation
        self.slide_in_animation = QPropertyAnimation(self, b"geometry")
        self.slide_in_animation.setDuration(500)
        self.slide_in_animation.setStartValue(self.geometry().translated(self.width(), 0))
        self.slide_in_animation.setEndValue(self.geometry())
        self.slide_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.slide_in_animation.start()

    def open_emotion_gallery(self, item):
        emotion = item.text()
        self.emotion_gallery = EmotionGalleryWindow(emotion)
        self.emotion_gallery.show()

    def refresh_emotion_list(self):
        # Clear the current list
        self.emotion_list.clear()

        # Path to the directory containing emotion folders
        base_path = os.path.join("saved_images", "correct_classifications")

        # Check if the directory exists
        if not os.path.exists(base_path):
            QMessageBox.warning(self, "Warning", "No saved emotions found!")
            return

        # Fetch available emotions based on folders
        emotions = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

        if not emotions:
            QMessageBox.information(self, "Information", "No emotions to display.")
            return

        # Re-populate the list
        for emotion in emotions:
            icon_path = f'assets/icons/{emotion.lower()}.png'
            if not os.path.exists(icon_path):  # Fallback icon if specific icon is missing
                icon_path = 'assets/icons/gallery.png'
            item = QListWidgetItem(QIcon(icon_path), emotion.capitalize())
            item.setSizeHint(QSize(80, 50))  # Keep item size consistent
            self.emotion_list.addItem(item)

