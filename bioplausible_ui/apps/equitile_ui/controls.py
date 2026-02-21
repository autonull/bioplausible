from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QProgressBar, QGroupBox, QSlider, QSpinBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

class ControlPanel(QWidget):
    """
    Control panel for managing training hyperparameters and visualization settings.
    """
    params_changed = pyqtSignal(dict) # {param_name: value}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Hyperparameters Group ---
        hp_group = QGroupBox("Hyperparameters")
        hp_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ff00ff; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        hp_layout = QVBoxLayout(hp_group)

        # Learning Rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QSpinBox() # Using SpinBox for simplicity, maybe DoubleSpinBox?
        # Actually DoubleSpinBox is better for LR
        from PyQt6.QtWidgets import QDoubleSpinBox
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01) # Default
        self.lr_spin.valueChanged.connect(self.emit_params)
        self.lr_spin.setToolTip("Controls the step size of weight updates.")
        lr_layout.addWidget(self.lr_spin)
        hp_layout.addLayout(lr_layout)

        # Inference Steps (Relaxation)
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Inference Steps:"))
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setRange(1, 20)
        self.steps_slider.setValue(5)
        self.steps_label = QLabel("5")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        self.steps_slider.valueChanged.connect(self.emit_params)
        self.steps_slider.setToolTip("Number of relaxation steps per forward pass. Higher = deeper settling.")
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        hp_layout.addLayout(steps_layout)

        layout.addWidget(hp_group)

        # --- Demo Settings Group ---
        demo_group = QGroupBox("Demo Settings")
        demo_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffff00; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        demo_layout = QVBoxLayout(demo_group)

        # Speedup Factor
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Simulated Speedup:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 100.0)
        self.speed_spin.setValue(17.0)
        self.speed_spin.valueChanged.connect(self.emit_params)
        self.speed_spin.setSuffix("x")
        self.speed_spin.setToolTip("Visual scaling factor for the speed gauge (demonstration only).")
        speed_layout.addWidget(self.speed_spin)
        demo_layout.addLayout(speed_layout)

        # Educational Mode
        self.edu_check = QCheckBox("Educational Mode")
        self.edu_check.setChecked(False) # Default off
        self.edu_check.setToolTip("Slows down visualization to explain steps.")
        self.edu_check.stateChanged.connect(self.emit_params)
        demo_layout.addWidget(self.edu_check)

        layout.addWidget(demo_group)

        # Spacer
        layout.addStretch()

    def emit_params(self):
        params = {
            "learning_rate": self.lr_spin.value(),
            "inference_steps": self.steps_slider.value(),
            "demo_speedup": self.speed_spin.value(),
            "educational_mode": self.edu_check.isChecked()
        }
        self.params_changed.emit(params)
