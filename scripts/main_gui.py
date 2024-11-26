# scripts/main_gui.py

import sys
import os
from PySide6.QtWidgets import QApplication
from scripts.windows.main_window import MainWindow

def main():
    # Ensure the current working directory is the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
