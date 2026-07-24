import json
import pathlib
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bioplausible_ui.app.tabs.console_tab import ConsoleTab
from bioplausible_ui.app.tabs.settings_tab import SettingsTab


def test_console_save_logs(qtbot):
    """Test saving logs to a file."""
    tab = ConsoleTab()
    qtbot.addWidget(tab)

    # Mock log content
    tab.log_output.text_edit.setPlainText("Test Log Content")

    # Mock QFileDialog to return a path
    with patch.object(
        QFileDialog, "getSaveFileName", return_value=("test_log.log", "Log Files")
    ):
        # Mock QMessageBox
        with patch.object(QMessageBox, "information") as mock_info:
            tab._save_logs()

            # Verify file created
            assert pathlib.Path("test_log.log").exists()
            with pathlib.Path("test_log.log").open() as f:
                content = f.read()
            assert content == "Test Log Content"

            # Verify success message
            mock_info.assert_called_once()

    # Cleanup
    if pathlib.Path("test_log.log").exists():
        pathlib.Path("test_log.log").unlink()


def test_settings_persistence(qtbot):
    """Test saving and loading settings."""
    tab = SettingsTab()
    qtbot.addWidget(tab)

    # Set mock values
    new_settings = {"theme": "light", "backend": "numpy"}
    tab.preferences.get_values = MagicMock(return_value=new_settings)

    # Save
    with patch.object(QMessageBox, "information"):
        tab._save_settings()

    assert pathlib.Path(SettingsTab.SETTINGS_FILE).exists()

    # Verify file content
    with pathlib.Path(SettingsTab.SETTINGS_FILE).open() as f:
        saved = json.load(f)
    assert saved == new_settings

    # Test Load
    # Reset widget state mock
    tab.preferences.set_values = MagicMock()
    tab._load_settings()
    tab.preferences.set_values.assert_called_with(new_settings)

    # Cleanup
    if pathlib.Path(SettingsTab.SETTINGS_FILE).exists():
        pathlib.Path(SettingsTab.SETTINGS_FILE).unlink()
