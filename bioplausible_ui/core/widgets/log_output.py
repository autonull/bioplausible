from PyQt6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class LogOutput(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("System Logs:"))
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def log(self, message):
        self.text_edit.append(message)
