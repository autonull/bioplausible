from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QSlider, QSpinBox, QCheckBox, QPushButton,
                             QDoubleSpinBox, QComboBox, QTabWidget)
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
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # === Tab 1: Live Controls ===
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)

        # Training Parameters
        hp_group = QGroupBox("Training")
        hp_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ff00ff; border: 1px solid #555; margin-top: 10px; }")
        hp_layout = QVBoxLayout(hp_group)

        # LR
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("LR:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01)
        self.lr_spin.valueChanged.connect(self.emit_params)
        lr_layout.addWidget(self.lr_spin)
        hp_layout.addLayout(lr_layout)

        # Dropout
        drop_layout = QHBoxLayout()
        drop_layout.addWidget(QLabel("Dropout:"))
        self.drop_spin = QDoubleSpinBox()
        self.drop_spin.setRange(0.0, 0.9)
        self.drop_spin.setSingleStep(0.05)
        self.drop_spin.setValue(0.1)
        self.drop_spin.valueChanged.connect(self.emit_params)
        drop_layout.addWidget(self.drop_spin)
        hp_layout.addLayout(drop_layout)

        # Weight Decay
        wd_layout = QHBoxLayout()
        wd_layout.addWidget(QLabel("W. Decay:"))
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 0.1)
        self.wd_spin.setSingleStep(0.001)
        self.wd_spin.setValue(0.01)
        self.wd_spin.valueChanged.connect(self.emit_params)
        wd_layout.addWidget(self.wd_spin)
        hp_layout.addLayout(wd_layout)

        # Inference Steps
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Relax Steps:"))
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setRange(1, 20)
        self.steps_slider.setValue(5)
        self.steps_label = QLabel("5")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        self.steps_slider.valueChanged.connect(self.emit_params)
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        hp_layout.addLayout(steps_layout)

        # MoT k
        mot_layout = QHBoxLayout()
        mot_layout.addWidget(QLabel("MoT k (Top-k):"))
        self.mot_spin = QSpinBox()
        self.mot_spin.setRange(1, 64)
        self.mot_spin.setValue(4)
        self.mot_spin.valueChanged.connect(self.emit_params)
        mot_layout.addWidget(self.mot_spin)
        hp_layout.addLayout(mot_layout)

        live_layout.addWidget(hp_group)

        # Demo Settings
        demo_group = QGroupBox("Visualization")
        demo_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffff00; border: 1px solid #555; margin-top: 10px; }")
        demo_layout = QVBoxLayout(demo_group)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Sim Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 100.0)
        self.speed_spin.setValue(17.0)
        self.speed_spin.valueChanged.connect(self.emit_params)
        speed_layout.addWidget(self.speed_spin)
        demo_layout.addLayout(speed_layout)

        self.edu_check = QCheckBox("Educational Mode")
        self.edu_check.stateChanged.connect(self.emit_params)
        demo_layout.addWidget(self.edu_check)

        live_layout.addWidget(demo_group)
        live_layout.addStretch()

        self.tabs.addTab(live_tab, "Live")

        # === Tab 2: Architecture & Data (Restart) ===
        arch_tab = QWidget()
        arch_layout = QVBoxLayout(arch_tab)

        # Data Config
        data_group = QGroupBox("Data & Context")
        data_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00ffcc; border: 1px solid #555; margin-top: 10px; }")
        data_layout = QVBoxLayout(data_group)

        # Dataset
        data_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Random", "Tiny Shakespeare", "WikiText-2"])
        data_layout.addWidget(self.dataset_combo)

        # Batch Size
        bs_layout = QHBoxLayout()
        bs_layout.addWidget(QLabel("Batch Size:"))
        self.bs_spin = QSpinBox()
        self.bs_spin.setRange(1, 128)
        self.bs_spin.setValue(32)
        bs_layout.addWidget(self.bs_spin)
        data_layout.addLayout(bs_layout)

        # Context Len
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel("Context Len:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(32, 1024)
        self.ctx_spin.setValue(128)
        ctx_layout.addWidget(self.ctx_spin)
        data_layout.addLayout(ctx_layout)

        arch_layout.addWidget(data_group)

        # Network Config
        net_group = QGroupBox("Network")
        net_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00ffcc; border: 1px solid #555; margin-top: 10px; }")
        net_layout = QVBoxLayout(net_group)

        # Layers
        l_layout = QHBoxLayout()
        l_layout.addWidget(QLabel("Layers:"))
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 12)
        self.layers_spin.setValue(6)
        l_layout.addWidget(self.layers_spin)
        net_layout.addLayout(l_layout)

        # Tiles
        t_layout = QHBoxLayout()
        t_layout.addWidget(QLabel("Tiles/Layer:"))
        self.tiles_spin = QSpinBox()
        self.tiles_spin.setRange(4, 256)
        self.tiles_spin.setValue(64)
        t_layout.addWidget(self.tiles_spin)
        net_layout.addLayout(t_layout)

        # Neurons
        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Neurons/Tile:"))
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(16, 256)
        self.neurons_spin.setValue(64)
        n_layout.addWidget(self.neurons_spin)
        net_layout.addLayout(n_layout)

        arch_layout.addWidget(net_group)

        # Apply Button
        self.apply_btn = QPushButton("Apply & Restart")
        self.apply_btn.setStyleSheet("background-color: #333; color: #00ffcc; font-weight: bold; padding: 10px;")
        self.apply_btn.clicked.connect(self.emit_reconfigure)
        arch_layout.addWidget(self.apply_btn)

        arch_layout.addStretch()

        self.tabs.addTab(arch_tab, "Config")

    def emit_params(self):
        params = {
            "learning_rate": self.lr_spin.value(),
            "inference_steps": self.steps_slider.value(),
            "demo_speedup": self.speed_spin.value(),
            "educational_mode": self.edu_check.isChecked(),
            "mot_k": self.mot_spin.value(),
            "dropout": self.drop_spin.value(),
            "weight_decay": self.wd_spin.value()
        }
        self.params_changed.emit(params)

    def emit_reconfigure(self):
        config = {
            "dataset_name": self.dataset_combo.currentText(),
            "batch_size": self.bs_spin.value(),
            "max_seq_len": self.ctx_spin.value(),
            "num_layers": self.layers_spin.value(),
            "tiles_per_layer": self.tiles_spin.value(),
            "neurons_per_tile": self.neurons_spin.value()
        }
        self.reconfigure_requested.emit(config)
