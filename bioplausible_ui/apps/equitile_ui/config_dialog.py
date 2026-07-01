from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bioplausible.models.registry import (
    MODEL_REGISTRY,
    get_model_spec,
    list_model_names,
)


class CustomStackBuilder(QWidget):
    """Widget to build a custom stack of layers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers_config = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # List of layers
        self.layer_list = QListWidget()
        layout.addWidget(self.layer_list)

        # Controls to add layers
        add_layout = QHBoxLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Linear", "Conv2d", "EquiTile", "Activation"])

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 4096)
        self.size_spin.setValue(128)
        self.size_spin.setSuffix(" units")

        # Only for Conv
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 11)
        self.kernel_spin.setValue(3)
        self.kernel_spin.setSuffix(" k")
        self.kernel_spin.hide()

        self.type_combo.currentIndexChanged.connect(self.on_type_changed)

        add_btn = QPushButton("Add Layer")
        add_btn.clicked.connect(self.add_layer)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_layer)

        add_layout.addWidget(self.type_combo)
        add_layout.addWidget(self.size_spin)
        add_layout.addWidget(self.kernel_spin)
        add_layout.addWidget(add_btn)

        layout.addLayout(add_layout)
        layout.addWidget(remove_btn)

        # Add default stack
        self.add_layer_config("linear", 128)
        self.add_layer_config("activation", "relu")
        self.add_layer_config("linear", 64)
        self.add_layer_config("activation", "relu")

    def on_type_changed(self):
        t = self.type_combo.currentText()
        if t == "Conv2d":
            self.kernel_spin.show()
            self.size_spin.setSuffix(" ch")
        elif t == "Activation":
            self.size_spin.hide()
            self.kernel_spin.hide()
        else:
            self.kernel_spin.hide()
            self.size_spin.show()
            self.size_spin.setSuffix(" units")

    def add_layer(self):
        t = self.type_combo.currentText().lower()
        if t == "activation":
            # Just use relu for now, could add combo
            self.add_layer_config("activation", "relu")
        elif t == "conv2d":
            self.add_layer_config(
                "conv2d", self.size_spin.value(), kernel=self.kernel_spin.value()
            )
        else:
            self.add_layer_config(t, self.size_spin.value())

    def add_layer_config(self, type_name, size_or_act, kernel=3):
        cfg = {"type": type_name}

        if type_name == "activation":
            cfg["act"] = size_or_act
            display = f"Activation ({size_or_act})"
        elif type_name == "conv2d":
            cfg["size"] = size_or_act
            cfg["kernel_size"] = kernel
            display = f"Conv2d ({size_or_act} ch, k={kernel})"
        else:
            cfg["size"] = size_or_act
            display = f"{type_name.capitalize()} ({size_or_act})"

        self.layers_config.append(cfg)
        self.layer_list.addItem(display)

    def remove_layer(self):
        row = self.layer_list.currentRow()
        if row >= 0:
            self.layer_list.takeItem(row)
            self.layers_config.pop(row)

    def get_config(self):
        return self.layers_config


class ModelConfigDialog(QDialog):
    """Dialog to configure a new model session."""

    def __init__(self, parent=None, current_config=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Bio-Plausible Model")
        self.resize(600, 700)
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

        # Mode Switcher (Standard Params vs Custom Builder)
        self.param_stack = QStackedWidget()

        # 1. Standard Hyperparameters
        self.params_group = QGroupBox("Hyperparameters")
        params_layout = QFormLayout()

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 100)
        self.layers_spin.setValue(self.current_config.get("num_layers", 4))

        self.hidden_spin = QSpinBox()  # Tiles per layer or Hidden Dim
        self.hidden_spin.setRange(1, 4096)
        self.hidden_spin.setValue(self.current_config.get("tiles_per_layer", 64))
        self.hidden_label = QLabel("Hidden Units / Tiles:")

        self.neurons_spin = QSpinBox()  # Neurons per tile
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

        self.params_group.setLayout(params_layout)
        self.param_stack.addWidget(self.params_group)

        # 2. Custom Builder
        self.custom_group = QGroupBox("Custom Layer Stack")
        custom_layout = QVBoxLayout()
        self.builder = CustomStackBuilder()
        custom_layout.addWidget(self.builder)
        self.custom_group.setLayout(custom_layout)
        self.param_stack.addWidget(self.custom_group)

        layout.addWidget(self.param_stack)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        queue_btn = QPushButton("Add to Queue")
        queue_btn.clicked.connect(self.add_to_queue)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("Run Now")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept_config)

        btn_layout.addWidget(queue_btn)
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

            if has_lm:
                self.task_combo.addItem("Language Modeling (LM)", "lm")
            if has_vision:
                self.task_combo.addItem("Vision (Classification)", "vision")

            self.task_combo.blockSignals(False)
            self.on_task_changed()

            # Switch between Standard and Custom builder
            if spec.model_type == "custom_stacked_model":
                self.param_stack.setCurrentIndex(1)
            else:
                self.param_stack.setCurrentIndex(0)

        except ValueError:
            pass

    def on_task_changed(self):
        task_data = self.task_combo.currentData()
        self.dataset_combo.clear()

        if task_data == "lm":
            self.dataset_combo.addItems(["Tiny Shakespeare", "WikiText-2"])
            self.hidden_label.setText("Tiles / Hidden Dim:")
        else:
            self.dataset_combo.addItems(
                ["MNIST", "Fashion MNIST", "CIFAR-10", "Digits"]
            )
            self.hidden_label.setText("Hidden Units:")

    def accept_config(self):
        self._build_config(queue=False)
        self.accept()

    def add_to_queue(self):
        self._build_config(queue=True)
        self.accept()

    def _build_config(self, queue=False):
        config = {
            "name": self.model_combo.currentText(),
            "task_type": self.task_combo.currentData(),
            "dataset_name": self.dataset_combo.currentText(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
            "max_seq_len": 64,
            "queue_requested": queue,
        }

        # Standard params
        config["num_layers"] = self.layers_spin.value()
        config["tiles_per_layer"] = self.hidden_spin.value()
        config["neurons_per_tile"] = self.neurons_spin.value()

        # Custom params
        if self.param_stack.currentIndex() == 1:
            config["layers_config"] = self.builder.get_config()

        self.result_config = config
