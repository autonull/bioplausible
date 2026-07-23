import sys

from PyQt6.QtWidgets import QApplication

from bioplausible_ui.lab.window import LabMainWindow


def main():
    app = QApplication(sys.argv)
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = LabMainWindow(model_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
