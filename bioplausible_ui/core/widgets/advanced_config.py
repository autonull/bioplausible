from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox,
                             QLabel, QVBoxLayout, QWidget)


class AdvancedConfigWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Advanced Configuration")
        grid = QGridLayout(group)

        # Gradient Method
        grid.addWidget(QLabel("Gradient Method:"), 0, 0)
        self.grad_combo = QComboBox()
        self.grad_combo.addItems(
            ["BPTT (Standard)", "Equilibrium (Implicit Diff)", "Contrastive (Hebbian)"]
        )
        self.grad_combo.setToolTip("Select the method for calculating gradients.")
        grid.addWidget(self.grad_combo, 0, 1)

        # Toggles
        self.compile_check = QCheckBox("Enable torch.compile (Speedup)")
        self.compile_check.setChecked(True)
        grid.addWidget(self.compile_check, 1, 0, 1, 2)

        self.kernel_check = QCheckBox("Use O(1) Memory Kernel (GPU)")
        self.kernel_check.setToolTip(
            "Use custom CUDA/Triton kernels for memory efficiency."
        )
        grid.addWidget(self.kernel_check, 2, 0, 1, 2)

        self.dynamics_check = QCheckBox("Live Dynamics Analysis")
        self.dynamics_check.setToolTip(
            "Perform expensive analysis of settling dynamics during training."
        )
        grid.addWidget(self.dynamics_check, 3, 0, 1, 2)

        layout.addWidget(group)

    def get_values(self):
        return {
            "gradient_method": self.grad_combo.currentText(),
            "use_compile": self.compile_check.isChecked(),
            "use_kernel": self.kernel_check.isChecked(),
            "track_dynamics": self.dynamics_check.isChecked(),
        }

    def set_values(self, values):
        if "gradient_method" in values:
            self.grad_combo.setCurrentText(values["gradient_method"])
        if "use_compile" in values:
            self.compile_check.setChecked(values["use_compile"])
        if "use_kernel" in values:
            self.kernel_check.setChecked(values["use_kernel"])
        if "track_dynamics" in values:
            self.dynamics_check.setChecked(values["track_dynamics"])
