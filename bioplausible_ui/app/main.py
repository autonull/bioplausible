import sys

from PyQt6.QtWidgets import QApplication

from bioplausible_ui.app.window import AppMainWindow


def main():
    app = QApplication(sys.argv)
    window = AppMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
