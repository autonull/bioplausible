from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)


class TrainingConfigWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        self.layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(64)
        self.layout.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.layout.addRow("Learning Rate:", self.lr_spin)

        # RL Specific
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setValue(0.99)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_label = QLabel("Gamma (Discount):")
        self.layout.addRow(self.gamma_label, self.gamma_spin)

        # LM Specific
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(8, 2048)
        self.seq_len_spin.setValue(128)
        self.seq_len_label = QLabel("Seq Length:")
        self.layout.addRow(self.seq_len_label, self.seq_len_spin)

        self.grad_combo = QComboBox()
        self.grad_combo.addItems([
            "BPTT (Standard)",
            "Equilibrium (Implicit Diff)",
            "Contrastive (Hebbian)",
        ])
        self.layout.addRow("Gradient:", self.grad_combo)

        self.compile_check = QCheckBox("torch.compile")
        self.compile_check.setChecked(True)
        self.layout.addRow("", self.compile_check)

        self.kernel_check = QCheckBox("O(1) Kernel Mode (GPU)")
        self.layout.addRow("", self.kernel_check)

        self.micro_check = QCheckBox("Live Dynamics Analysis")
        self.layout.addRow("", self.micro_check)

        # Initial state: hide specific controls
        self.set_task("vision")  # Default

    def set_task(self, task):
        # Hide all first
        self.gamma_spin.hide()
        self.gamma_label.hide()
        self.seq_len_spin.hide()
        self.seq_len_label.hide()

        if task == "rl":
            self.gamma_spin.show()
            self.gamma_label.show()
        elif task == "lm":
            self.seq_len_spin.show()
            self.seq_len_label.show()

    def set_values(self, values):
        """Set values from a dictionary."""
        if not values:
            return

        if "epochs" in values:
            self.epochs_spin.setValue(int(values["epochs"]))
        if "batch_size" in values:
            self.batch_spin.setValue(int(values["batch_size"]))
        if "learning_rate" in values:
            self.lr_spin.setValue(float(values["learning_rate"]))
        if "gradient_method" in values:
            self.grad_combo.setCurrentText(values["gradient_method"])
        if "use_compile" in values:
            self.compile_check.setChecked(bool(values["use_compile"]))
        if "use_kernel" in values:
            self.kernel_check.setChecked(bool(values["use_kernel"]))
        if "monitor_dynamics" in values:
            self.micro_check.setChecked(bool(values["monitor_dynamics"]))
        if "gamma" in values:
            self.gamma_spin.setValue(float(values["gamma"]))
        if "seq_len" in values:
            self.seq_len_spin.setValue(int(values["seq_len"]))

    def get_values(self):
        return {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "gradient_method": self.grad_combo.currentText(),
            "use_compile": self.compile_check.isChecked(),
            "use_kernel": self.kernel_check.isChecked(),
            "monitor_dynamics": self.micro_check.isChecked(),
            "gamma": self.gamma_spin.value(),
            "seq_len": self.seq_len_spin.value(),
        }
