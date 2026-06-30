from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QWidget)


class PresetSelector(QWidget):
    presetSelected = pyqtSignal(dict)  # Emits configuration dict

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("🚀 Quick Presets:"))

        self.combo = QComboBox()
        self.combo.addItems(
            ["Custom", "Standard Backprop", "Fast EqProp", "Deep EqProp (Accurate)"]
        )
        self.combo.currentTextChanged.connect(self._apply_preset)
        layout.addWidget(self.combo)

        # Add a reset button too
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setToolTip("Reset to defaults")
        self.reset_btn.clicked.connect(
            lambda: self._apply_preset("Custom", force_reset=True)
        )
        layout.addWidget(self.reset_btn)

    def _apply_preset(self, name, force_reset=False):
        if name == "Custom" and not force_reset:
            return

        config = {}

        if name == "Standard Backprop":
            config = {
                "advanced": {"gradient_method": "BPTT (Standard)", "use_kernel": False},
                "hyperparams": {"steps": 5},  # Minimal steps
            }
        elif name == "Fast EqProp":
            config = {
                "advanced": {
                    "gradient_method": "Equilibrium (Implicit Diff)",
                    "use_kernel": True,
                },
                "hyperparams": {"steps": 15},
            }
        elif name == "Deep EqProp (Accurate)":
            config = {
                "advanced": {
                    "gradient_method": "Equilibrium (Implicit Diff)",
                    "use_kernel": True,
                },
                "hyperparams": {"steps": 50},
            }
        elif force_reset:
            # Defaults
            config = {
                "advanced": {
                    "gradient_method": "BPTT (Standard)",
                    "use_kernel": False,
                    "use_compile": True,
                    "track_dynamics": False,
                },
                "hyperparams": {"steps": 30, "learning_rate": 0.001, "hidden_dim": 256},
                "training": {"epochs": 10, "batch_size": 64},
            }

        self.presetSelected.emit(config)
