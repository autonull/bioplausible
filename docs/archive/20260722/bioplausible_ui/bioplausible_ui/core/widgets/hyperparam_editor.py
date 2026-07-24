from bioplausible.models.registry import get_model_spec
from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout, QSpinBox, QWidget


class HyperparamEditor(QWidget):
    def __init__(self, model=None, defaults=None, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.values = {}
        if model:
            self.update_for_model(model)
        elif defaults:
            self.update_from_dict(defaults)

    def set_model(self, model_name):
        self.update_for_model(model_name)

    def update_from_dict(self, defaults):
        # Clear layout
        self._clear_layout()
        self.values = {}

        for key, value in defaults.items():
            if isinstance(value, bool):
                from PyQt6.QtWidgets import QCheckBox

                widget = QCheckBox()
                widget.setChecked(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-10000.0, 10000.0)
                widget.setValue(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-10000, 10000)
                widget.setValue(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, str):
                from PyQt6.QtWidgets import QLineEdit

                widget = QLineEdit(value)
                self.layout.addRow(key.title() + ":", widget)
                self.values[key] = widget
            elif isinstance(value, tuple) and len(value) == 2:
                # Handle tuple as (default, range) or (min, max)
                # Heuristic: if types match, assume range with default = min
                min_val, max_val = value
                if isinstance(min_val, float):
                    widget = QDoubleSpinBox()
                    widget.setRange(min_val, max_val)
                    widget.setValue(min_val)  # Default to min
                    self.layout.addRow(key.title() + ":", widget)
                    self.values[key] = widget
                elif isinstance(min_val, int):
                    widget = QSpinBox()
                    widget.setRange(min_val, max_val)
                    widget.setValue(min_val)
                    self.layout.addRow(key.title() + ":", widget)
                    self.values[key] = widget

    def _clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def update_for_model(self, model_name):
        # Clear layout
        self._clear_layout()

        self.values = {}
        try:
            spec = get_model_spec(model_name)

            # --- Standard Params ---

            # Hidden Dimension
            self.hidden_spin = QSpinBox()
            self.hidden_spin.setRange(16, 4096)
            self.hidden_spin.setValue(256)
            self.hidden_spin.setSingleStep(32)
            self.layout.addRow("Hidden Dim:", self.hidden_spin)
            self.values["hidden_dim"] = self.hidden_spin

            # LR
            self.lr_spin = QDoubleSpinBox()
            self.lr_spin.setRange(0.00001, 1.0)
            self.lr_spin.setSingleStep(0.0001)
            self.lr_spin.setDecimals(5)
            self.lr_spin.setValue(spec.default_lr)
            self.layout.addRow("Learning Rate:", self.lr_spin)
            self.values["learning_rate"] = self.lr_spin

            # Beta
            self.beta_spin = QDoubleSpinBox()
            self.beta_spin.setRange(0.0, 10.0)
            self.beta_spin.setSingleStep(0.01)
            self.layout.addRow("Beta:", self.beta_spin)
            self.values["beta"] = self.beta_spin

            # Steps
            self.steps_spin = QSpinBox()
            self.steps_spin.setRange(1, 100)
            self.layout.addRow("Steps:", self.steps_spin)
            self.values["steps"] = self.steps_spin

            # --- Custom Params ---
            if spec.custom_hyperparams:
                from PyQt6.QtWidgets import QCheckBox

                for key, default_val in spec.custom_hyperparams.items():
                    label = key.replace("_", " ").title() + ":"

                    if isinstance(default_val, bool):
                        w = QCheckBox()
                        w.setChecked(default_val)
                        self.layout.addRow(label, w)
                        self.values[key] = w
                    elif isinstance(default_val, float):
                        w = QDoubleSpinBox()
                        w.setRange(-1e6, 1e6)
                        w.setValue(default_val)
                        self.layout.addRow(label, w)
                        self.values[key] = w
                    elif isinstance(default_val, int):
                        w = QSpinBox()
                        w.setRange(-1e6, 1e6)
                        w.setValue(default_val)
                        self.layout.addRow(label, w)
                        self.values[key] = w
                    elif isinstance(default_val, tuple) and len(default_val) == 2:
                        # Assumed (min, max) range, use min as default
                        min_v, max_v = default_val
                        if isinstance(min_v, float):
                            w = QDoubleSpinBox()
                            w.setRange(min_v, max_v)
                            w.setValue(min_v)
                            self.layout.addRow(label, w)
                            self.values[key] = w
                        elif isinstance(min_v, int):
                            w = QSpinBox()
                            w.setRange(min_v, max_v)
                            w.setValue(min_v)
                            self.layout.addRow(label, w)
                            self.values[key] = w

        except ValueError:
            pass

    def get_values(self):
        res = {}
        for k, v in self.values.items():
            if hasattr(v, "value"):
                res[k] = v.value()
            elif hasattr(v, "isChecked"):
                res[k] = v.isChecked()
            elif hasattr(v, "text"):
                res[k] = v.text()
        return res

    def set_values(self, values):
        """Set values for existing widgets."""
        for k, v in values.items():
            if k in self.values:
                widget = self.values[k]
                if hasattr(widget, "setValue"):
                    # Handle int vs float mismatch if necessary
                    try:
                        widget.setValue(v)
                    except TypeError:
                        widget.setValue(float(v))
                elif hasattr(widget, "setChecked"):
                    widget.setChecked(bool(v))
                elif hasattr(widget, "setText"):
                    widget.setText(str(v))
