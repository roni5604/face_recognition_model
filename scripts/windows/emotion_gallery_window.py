# scripts/windows/emotion_gallery_window.py

import os
from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize
from PySide6.QtCore import Qt, QPropertyAnimation

class EmotionGalleryWindow(QMainWindow):
    def __init__(self, emotion):
        super().__init__()
        self.setWindowTitle(f"{emotion} Images")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon(f'assets/icons/{emotion.lower()}.png'))

        self.emotion = emotion
        self.image_paths = self.get_image_paths()
        self.current_index = 0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 400)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        prev_button.setIcon(QIcon('assets/icons/prev.png'))
        prev_button.setIconSize(QSize(24, 24))
        next_button = QPushButton("Next")
        next_button.setIcon(QIcon('assets/icons/next.png'))
        next_button.setIconSize(QSize(24, 24))
        back_button = QPushButton("Back")
        back_button.setIcon(QIcon('assets/icons/back.png'))
        back_button.setIconSize(QSize(24, 24))

        for btn in [prev_button, next_button, back_button]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
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

        prev_button.clicked.connect(self.show_previous_image)
        next_button.clicked.connect(self.show_next_image)
        back_button.clicked.connect(self.close)
        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(next_button)
        nav_layout.addWidget(back_button)
        self.layout.addLayout(nav_layout)

        central_widget.setLayout(self.layout)
        self.show_image()

        # Fade-in animation
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(500)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.start()

    def get_image_paths(self):
        emotion_folder = os.path.join("saved_images", "correct_classifications", self.emotion)
        if os.path.exists(emotion_folder):
            return [os.path.join(emotion_folder, img) for img in os.listdir(emotion_folder) if not img.startswith('.')]
        else:
            return []

    def show_image(self):
        if self.image_paths:
            image_path = self.image_paths[self.current_index]
            pixmap = QPixmap(image_path).scaled(600, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
        else:
            QMessageBox.information(self, "No Images", f"No images found for emotion: {self.emotion}")
            self.close()

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image()
