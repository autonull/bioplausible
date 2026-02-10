from PyQt6.QtCore import QObject, QThread, pyqtSignal

from bioplausible.pipeline.events import CompletedEvent, ProgressEvent
from bioplausible.pipeline.session import TrainingConfig, TrainingSession


class TrainingWorker(QThread):
    progress = pyqtSignal(int, dict)
    completed = pyqtSignal(dict)

    def __init__(self, session: TrainingSession):
        super().__init__()
        self.session = session

    def run(self):
        try:
            for event in self.session.start():
                if isinstance(event, ProgressEvent):
                    self.progress.emit(event.epoch, event.metrics)
                elif isinstance(event, CompletedEvent):
                    self.completed.emit(event.final_metrics)
        except Exception as e:
            # We should probably emit error signal from worker or let it crash
            print(f"Worker Error: {e}")


class SessionBridge(QObject):
    """Adapts TrainingSession to Qt signals."""

    progress_updated = pyqtSignal(int, dict)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.session = TrainingSession(config)
        self.worker = None

    def start(self):
        self.worker = TrainingWorker(self.session)
        self.worker.progress.connect(self.progress_updated)
        self.worker.completed.connect(self.training_completed)
        self.worker.start()

    def stop(self):
        self.session.stop()
        if self.worker:
            self.worker.quit()
            self.worker.wait()
