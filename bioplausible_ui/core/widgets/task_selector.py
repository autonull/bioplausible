from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class TaskSelector(QWidget):
    valueChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Task:"))
        self.combo = QComboBox()
        self.combo.addItems(["vision", "lm", "rl", "diffusion"])
        self.combo.currentTextChanged.connect(self.valueChanged.emit)
        layout.addWidget(self.combo)

    def get_task(self):
        return self.combo.currentText()
