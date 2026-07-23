from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from bioplausible.models.registry import MODEL_REGISTRY


class ModelSelector(QWidget):
    valueChanged = pyqtSignal(str)

    def __init__(self, task="vision", parent=None):
        super().__init__(parent)
        self.task = task
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Model:"))
        self.combo = QComboBox()
        self.update_models(task)
        self.combo.currentTextChanged.connect(self.valueChanged.emit)
        layout.addWidget(self.combo)

    def set_task(self, task):
        self.task = task
        self.update_models(task)

    def update_models(self, task):
        self.combo.clear()
        models = [
            m.name
            for m in MODEL_REGISTRY
            if m.task_compat is None or task in m.task_compat
        ]
        self.combo.addItems(models)

    def get_selected_model(self):
        return self.combo.currentText()
