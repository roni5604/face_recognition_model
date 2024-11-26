# scripts/windows/main_window.py

from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt, QPropertyAnimation, QSize, QEasingCurve

from .info_window import InfoWindow
from .live_detection_window import LiveDetectionWindow
from .gallery_window import GalleryWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Expression Recognition")
        self.setFixedSize(800, 600)
        self.setWindowIcon(QIcon('assets/icons/app_icon.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Title label with image
        title_layout = QVBoxLayout()
        title_image = QLabel()
        title_pixmap = QPixmap('assets/icons/app_icon.png').scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        title_image.setPixmap(title_pixmap)
        title_image.setAlignment(Qt.AlignCenter)

        title_label = QLabel("Facial Expression Recognition")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        """)
        title_layout.addWidget(title_image)
        title_layout.addWidget(title_label)

        layout.addLayout(title_layout)

        # Buttons with icons
        info_button = QPushButton("Information About the Project")
        use_model_button = QPushButton("Use the Model")
        view_images_button = QPushButton("Images the Model Got Right")

        buttons = [info_button, use_model_button, view_images_button]
        icons = ['assets/icons/info.png', 'assets/icons/live.png', 'assets/icons/images.png']
        for btn, icon_path in zip(buttons, icons):
            btn.setFixedHeight(60)
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(40, 40))
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 10px;
                    text-align: left;
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    margin: 10px;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
            """)
            layout.addWidget(btn)

        info_button.clicked.connect(self.open_info_window)
        use_model_button.clicked.connect(self.open_live_detection_window)
        view_images_button.clicked.connect(self.open_gallery_window)

        central_widget.setLayout(layout)

        # Apply fade-in animation
        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(1000)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.start()

    def open_info_window(self):
        self.info_window = InfoWindow()
        self.info_window.show()

    def open_live_detection_window(self):
        self.live_detection_window = LiveDetectionWindow()
        self.live_detection_window.show()

    def open_gallery_window(self):
        self.gallery_window = GalleryWindow()
        self.gallery_window.show()
