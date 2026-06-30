from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QVBoxLayout,
                             QWidget)


class QueueManager:
    """Manages a queue of experiment configurations."""

    def __init__(self):
        self.queue = []

    def add_job(self, config):
        self.queue.append(config)

    def pop_job(self):
        if self.queue:
            return self.queue.pop(0)
        return None

    def clear(self):
        self.queue = []

    def remove_job(self, index):
        if 0 <= index < len(self.queue):
            self.queue.pop(index)

    def get_jobs(self):
        return self.queue


class QueuePanel(QWidget):
    """Widget to display and control the experiment queue."""

    # Signal emitted when queue processing should start
    start_queue_signal = pyqtSignal()

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("Experiment Queue")
        group_layout = QVBoxLayout(group)

        self.list_widget = QListWidget()
        group_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()

        run_btn = QPushButton("Run Queue")
        run_btn.setStyleSheet(
            "background-color: #00aa00; color: white; font-weight: bold;"
        )
        run_btn.clicked.connect(self.on_run_clicked)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_queue)

        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(clear_btn)

        group_layout.addLayout(btn_layout)
        layout.addWidget(group)

        self.update_list()

    def update_list(self):
        self.list_widget.clear()
        for i, job in enumerate(self.manager.get_jobs()):
            name = job.get("name", "Unknown Model")
            task = job.get("task_type", "lm")
            layers = job.get("num_layers", "?")
            item = QListWidgetItem(f"{i+1}. {name} ({task}) - L{layers}")
            self.list_widget.addItem(item)

    def add_job(self, config):
        self.manager.add_job(config)
        self.update_list()

    def remove_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.manager.remove_job(row)
            self.update_list()

    def clear_queue(self):
        self.manager.clear()
        self.update_list()

    def on_run_clicked(self):
        self.start_queue_signal.emit()
