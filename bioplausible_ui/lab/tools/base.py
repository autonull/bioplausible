from PyQt6.QtWidgets import QVBoxLayout, QWidget


class BaseTool(QWidget):
    ICON = "🛠️"

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)

    def set_model(self, model):
        self.model = model
        self.refresh()

    def refresh(self):
        pass
