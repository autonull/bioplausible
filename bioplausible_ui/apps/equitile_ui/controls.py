from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QSlider, QSpinBox, QCheckBox, QPushButton, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal

class ControlPanel(QWidget):
    """
    Control panel for managing training hyperparameters, demo settings, and network architecture.
    """
    params_changed = pyqtSignal(dict) # Real-time updates {param_name: value}
    reconfigure_requested = pyqtSignal(dict) # Architecture changes {param_name: value}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Hyperparameters Group (Real-time) ---
        hp_group = QGroupBox("Training Parameters (Live)")
        hp_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ff00ff; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        hp_layout = QVBoxLayout(hp_group)

        # Learning Rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01)
        self.lr_spin.valueChanged.connect(self.emit_params)
        lr_layout.addWidget(self.lr_spin)
        hp_layout.addLayout(lr_layout)

        # Inference Steps
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Relaxation Steps:"))
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setRange(1, 20)
        self.steps_slider.setValue(5)
        self.steps_label = QLabel("5")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        self.steps_slider.valueChanged.connect(self.emit_params)
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        hp_layout.addLayout(steps_layout)

        layout.addWidget(hp_group)

        # --- Architecture Group (Requires Restart) ---
        arch_group = QGroupBox("Network Architecture (Reset)")
        arch_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00ffcc; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        arch_layout = QVBoxLayout(arch_group)

        # Layers
        layers_layout = QHBoxLayout()
        layers_layout.addWidget(QLabel("Layers:"))
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 12)
        self.layers_spin.setValue(6)
        layers_layout.addWidget(self.layers_spin)
        arch_layout.addLayout(layers_layout)

        # Tiles per Layer
        tiles_layout = QHBoxLayout()
        tiles_layout.addWidget(QLabel("Tiles / Layer:"))
        self.tiles_spin = QSpinBox()
        self.tiles_spin.setRange(4, 256)
        self.tiles_spin.setValue(64)
        self.tiles_spin.setSingleStep(4)
        tiles_layout.addWidget(self.tiles_spin)
        arch_layout.addLayout(tiles_layout)

        # Neurons per Tile
        neurons_layout = QHBoxLayout()
        neurons_layout.addWidget(QLabel("Neurons / Tile:"))
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(16, 256)
        self.neurons_spin.setValue(64)
        self.neurons_spin.setSingleStep(16)
        neurons_layout.addWidget(self.neurons_spin)
        arch_layout.addLayout(neurons_layout)

        # Apply Button
        self.apply_btn = QPushButton("Apply Configuration & Restart")
        self.apply_btn.setStyleSheet("background-color: #333; color: #00ffcc; font-weight: bold;")
        self.apply_btn.clicked.connect(self.emit_reconfigure)
        arch_layout.addWidget(self.apply_btn)

        layout.addWidget(arch_group)

        # --- Demo Settings Group ---
        demo_group = QGroupBox("Visualization")
        demo_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffff00; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        demo_layout = QVBoxLayout(demo_group)

        # Speedup Factor
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed Mul:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 100.0)
        self.speed_spin.setValue(17.0)
        self.speed_spin.valueChanged.connect(self.emit_params)
        speed_layout.addWidget(self.speed_spin)
        demo_layout.addLayout(speed_layout)

        # Educational Mode
        self.edu_check = QCheckBox("Educational Mode (Slow)")
        self.edu_check.setChecked(False)
        self.edu_check.stateChanged.connect(self.emit_params)
        demo_layout.addWidget(self.edu_check)

        layout.addWidget(demo_group)

        layout.addStretch()

    def emit_params(self):
        params = {
            "learning_rate": self.lr_spin.value(),
            "inference_steps": self.steps_slider.value(),
            "demo_speedup": self.speed_spin.value(),
            "educational_mode": self.edu_check.isChecked()
        }
        self.params_changed.emit(params)

    def emit_reconfigure(self):
        config = {
            "num_layers": self.layers_spin.value(),
            "tiles_per_layer": self.tiles_spin.value(),
            "neurons_per_tile": self.neurons_spin.value()
        }
        self.reconfigure_requested.emit(config)
