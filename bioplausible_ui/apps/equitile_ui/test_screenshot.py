
import sys
import os
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from bioplausible_ui.apps.equitile_ui.window import EquiTileWindow

def capture_screenshot():
    """Run the EquiTileUI, wait for a bit, take a screenshot, and exit."""
    app = QApplication(sys.argv)

    # Create window
    window = EquiTileWindow()
    window.show()

    # Define screenshot function
    def take_screenshot_and_quit():
        print("Taking screenshot...")
        # Grab window
        pixmap = window.grab()
        # Save screenshot
        screenshot_path = "equitile_ui_screenshot.png"
        pixmap.save(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        # Close and quit
        window.close()
        app.quit()

    # Schedule screenshot after 5 seconds (allow training to start and metrics to populate)
    QTimer.singleShot(5000, take_screenshot_and_quit)

    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    capture_screenshot()
