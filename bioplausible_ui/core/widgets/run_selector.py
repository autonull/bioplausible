from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from bioplausible.pipeline.results import ResultsManager


class RunSelector(QWidget):
    """Widget to select a past training run."""

    valueChanged = pyqtSignal(str)  # Emits run_id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Select Run:"))

        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self._on_changed)
        layout.addWidget(self.combo)

        self.manager = ResultsManager()
        self.refresh()

    def refresh(self):
        """Reload runs from disk."""
        self.combo.blockSignals(True)
        self.combo.clear()

        runs = self.manager.list_runs()
        self.run_map = {}  # Maps display text to run_id

        for run in runs:
            run_id = run.get("run_id")
            timestamp = run.get("timestamp", "")[:16].replace("T", " ")
            config = run.get("config", {})
            metrics = run.get("metrics", {})

            if "final_metrics" in metrics:
                metrics = metrics["final_metrics"]

            task = config.get("task", "?")
            model = config.get("model", "?")
            acc = metrics.get("accuracy", 0.0)

            display = f"[{timestamp}] {task.upper()} - {model} (Acc: {acc:.3f})"

            self.combo.addItem(display)
            self.run_map[display] = run_id

        self.combo.blockSignals(False)

        if self.combo.count() > 0:
            self.combo.setCurrentIndex(0)
            # Emit initial
            self._on_changed(self.combo.currentText())

    def _on_changed(self, text):
        run_id = self.run_map.get(text)
        if run_id:
            self.valueChanged.emit(run_id)

    def get_run_id(self):
        return self.run_map.get(self.combo.currentText())
