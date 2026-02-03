import os
from unittest.mock import patch

from bioplausible_ui.app.window import AppMainWindow
from bioplausible_ui.lab.window import LabMainWindow


def test_app_screenshots_all_tabs(qtbot):
    """Takes screenshots of every tab in the main application window."""
    window = AppMainWindow()
    qtbot.addWidget(window)
    window.show()
    # qtbot.waitExposed(window)  # May hang in headless/CI

    os.makedirs("screenshots", exist_ok=True)

    # Capture main overview
    window.grab().save("screenshots/app_00_overview.png")

    # Iterate through tabs
    tab_widget = window.tabs
    for i in range(tab_widget.count()):
        tab_name = tab_widget.tabText(i)
        clean_name = tab_name.replace(" ", "_").lower()

        tab_widget.setCurrentIndex(i)
        qtbot.wait(100)  # Wait for tab switch animation/render

        # Capture tab
        window.grab().save(f"screenshots/app_tab_{i:02d}_{clean_name}.png")

        assert os.path.exists(f"screenshots/app_tab_{i:02d}_{clean_name}.png")


def test_lab_screenshot_populated(qtbot):
    """Takes a screenshot of the lab window with a mock model loaded."""

    # Mock torch.load to return a dummy checkpoint
    mock_checkpoint = {
        "config": {"model_name": "EqProp MLP"},  # A model with many capabilities
        "state_dict": {},
    }

    with patch("torch.load", return_value=mock_checkpoint):
        # We also need to mock get_model_spec because it might read from registry
        # But registry is in memory, so it should be fine if 'EqProp MLP' is in registry.
        # It is in registry (we saw it in previous turns).

        # We pass a dummy path "dummy.pt"
        window = LabMainWindow(model_path="dummy.pt")
        qtbot.addWidget(window)
        window.show()
        # qtbot.waitExposed(window)  # May hang in headless/CI

        os.makedirs("screenshots", exist_ok=True)

        # Capture overview with tools loaded
        window.grab().save("screenshots/lab_00_populated.png")
        assert os.path.exists("screenshots/lab_00_populated.png")

        # Iterate through tool tabs
        tab_widget = window.tabs
        # There should be tools loaded now.
        assert tab_widget.count() > 0, "Lab window should have tools loaded"

        for i in range(tab_widget.count()):
            tab_name = tab_widget.tabText(i)
            # tabText might contain emojis, let's strip them or just use index
            clean_name = (
                "".join([c for c in tab_name if c.isalnum() or c in (" ", "_")])
                .strip()
                .replace(" ", "_")
                .lower()
            )

            tab_widget.setCurrentIndex(i)
            qtbot.wait(100)

            window.grab().save(f"screenshots/lab_tab_{i:02d}_{clean_name}.png")
            assert os.path.exists(f"screenshots/lab_tab_{i:02d}_{clean_name}.png")
