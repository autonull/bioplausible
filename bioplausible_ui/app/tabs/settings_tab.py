import json
import os

from PyQt6.QtWidgets import QMessageBox

from bioplausible_ui.app.schemas.settings import SETTINGS_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab


class SettingsTab(BaseTab):
    """Settings tab - UI auto-generated from schema."""

    SCHEMA = SETTINGS_TAB_SCHEMA
    SETTINGS_FILE = "bioplausible_settings.json"

    def _post_init(self):
        self._load_settings()

    def _load_settings(self):
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                self.preferences.set_values(settings)
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def _save_settings(self):
        settings = self.preferences.get_values()
        try:
            with open(self.SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    def _reset_settings(self):
        defaults = self.SCHEMA.widgets[0].params.get("defaults", {})
        self.preferences.set_values(defaults)
        QMessageBox.information(self, "Settings", "Settings reset to defaults.")
