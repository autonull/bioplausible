import sys

from bioplausible_ui.app.window import AppMainWindow
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    window = AppMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
