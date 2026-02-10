from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class ExportFormatSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Export Format:"))
        self.combo = QComboBox()
        self.combo.addItems(["ONNX", "TorchScript"])
        layout.addWidget(self.combo)

    def get_format(self):
        return self.combo.currentText()
