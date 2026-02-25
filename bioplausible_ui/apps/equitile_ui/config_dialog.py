from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QSpinBox, QDoubleSpinBox,
                             QPushButton, QFormLayout, QGroupBox, QLineEdit)
from PyQt6.QtCore import Qt

from bioplausible.models.registry import MODEL_REGISTRY, list_model_names, get_model_spec

class ModelConfigDialog(QDialog):
    """Dialog to configure a new model session."""

    def __init__(self, parent=None, current_config=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Bio-Plausible Model")
        self.resize(500, 600)
        self.result_config = None

        self.current_config = current_config or {}

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Model Selection
        model_group = QGroupBox("Model Architecture")
        model_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(list_model_names())
        # Select current model if applicable
        current_name = self.current_config.get("name", "EquiTile")
        index = self.model_combo.findText(current_name, Qt.MatchFlag.MatchContains)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        self.desc_label = QLabel("Description...")
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #888; font-style: italic;")

        model_layout.addRow("Algorithm:", self.model_combo)
        model_layout.addRow("", self.desc_label)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Task & Data
        data_group = QGroupBox("Task & Data")
        data_layout = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItems(["Language Modeling (LM)", "Vision (Classification)"])

        self.dataset_combo = QComboBox()
        # Will be populated based on task

        self.task_combo.currentIndexChanged.connect(self.on_task_changed)

        data_layout.addRow("Task Type:", self.task_combo)
        data_layout.addRow("Dataset:", self.dataset_combo)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Hyperparameters
        params_group = QGroupBox("Hyperparameters")
        params_layout = QFormLayout()

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 100)
        self.layers_spin.setValue(self.current_config.get("num_layers", 4))

        self.hidden_spin = QSpinBox() # Tiles per layer or Hidden Dim
        self.hidden_spin.setRange(1, 4096)
        self.hidden_spin.setValue(self.current_config.get("tiles_per_layer", 64))
        self.hidden_label = QLabel("Hidden Units / Tiles:")

        self.neurons_spin = QSpinBox() # Neurons per tile
        self.neurons_spin.setRange(1, 1024)
        self.neurons_spin.setValue(self.current_config.get("neurons_per_tile", 32))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(self.current_config.get("learning_rate", 0.001))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(self.current_config.get("batch_size", 32))

        params_layout.addRow("Num Layers:", self.layers_spin)
        params_layout.addRow(self.hidden_label, self.hidden_spin)
        params_layout.addRow("Neurons per Unit:", self.neurons_spin)
        params_layout.addRow("Learning Rate:", self.lr_spin)
        params_layout.addRow("Batch Size:", self.batch_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("Create Model")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept_config)

        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        # Trigger updates
        self.on_model_changed()
        self.on_task_changed()

    def on_model_changed(self):
        name = self.model_combo.currentText()
        try:
            spec = get_model_spec(name)
            self.desc_label.setText(spec.description)
            self.lr_spin.setValue(spec.default_lr)

            # Filter compatible tasks
            compat = spec.task_compat
            self.task_combo.blockSignals(True)
            self.task_combo.clear()

            has_lm = not compat or "lm" in compat
            has_vision = not compat or "vision" in compat or "cifar10" in compat

            if has_lm: self.task_combo.addItem("Language Modeling (LM)", "lm")
            if has_vision: self.task_combo.addItem("Vision (Classification)", "vision")

            self.task_combo.blockSignals(False)
            self.on_task_changed()

        except ValueError:
            pass

    def on_task_changed(self):
        task_data = self.task_combo.currentData()
        self.dataset_combo.clear()

        if task_data == "lm":
            self.dataset_combo.addItems(["Tiny Shakespeare", "WikiText-2"])
            self.hidden_label.setText("Tiles / Hidden Dim:")
        else:
            self.dataset_combo.addItems(["MNIST", "Fashion MNIST", "CIFAR-10", "Digits"])
            self.hidden_label.setText("Hidden Units:")

    def accept_config(self):
        self.result_config = {
            "name": self.model_combo.currentText(),
            "task_type": self.task_combo.currentData(),
            "dataset_name": self.dataset_combo.currentText(),
            "num_layers": self.layers_spin.value(),
            "tiles_per_layer": self.hidden_spin.value(), # Used as hidden dim for non-tiled
            "neurons_per_tile": self.neurons_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
            "max_seq_len": 64 # Default
        }
        self.accept()
