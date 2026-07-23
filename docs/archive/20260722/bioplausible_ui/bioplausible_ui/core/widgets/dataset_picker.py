from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class DatasetPicker(QWidget):
    valueChanged = pyqtSignal(str)

    def __init__(self, task="vision", parent=None):
        super().__init__(parent)
        self.task = task
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Dataset:"))
        self.combo = QComboBox()
        self.update_datasets(task)
        self.combo.currentTextChanged.connect(self.valueChanged.emit)
        layout.addWidget(self.combo)

    def set_task(self, task):
        self.task = task
        self.update_datasets(task)

    def update_datasets(self, task):
        self.combo.clear()
        if task == "vision":
            self.combo.addItems(["mnist", "cifar10"])
        elif task == "lm":
            self.combo.addItems(["tiny_shakespeare", "wikitext"])
        elif task == "rl":
            self.combo.addItems(["cartpole", "lunarlander"])
        elif task == "diffusion":
            self.combo.addItems(["mnist", "cifar10"])

    def get_dataset(self):
        return self.combo.currentText()
