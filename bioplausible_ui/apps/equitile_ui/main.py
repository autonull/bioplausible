import sys
from PyQt6.QtWidgets import QApplication
from bioplausible_ui.apps.equitile_ui.window import EquiTileWindow

def main():
    """Main entry point for EquiTile UI."""
    app = QApplication(sys.argv)

    # Set Application Metadata
    app.setApplicationName("EquiTile Demo")
    app.setApplicationVersion("1.0.0")

    # Create and Show Window
    window = EquiTileWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
