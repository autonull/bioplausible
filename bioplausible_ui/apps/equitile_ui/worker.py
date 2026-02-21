import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

class TrainingWorker(QThread):
    """
    Worker thread for EquiTile training.

    Signals:
    - update_signal: Emitted on each training step with metrics and state.
    """
    update_signal = pyqtSignal(float, float, float, object, object, str) # loss, tokens_sec, sparsity, importance, activity, generated_text

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = False
        self.paused = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()

    def run(self):
        """Main training loop."""
        self.running = True

        while self.running:
            # Handle Pause
            self._mutex.lock()
            if self.paused:
                self._cond.wait(self._mutex)
            self._mutex.unlock()

            if not self.running:
                break

            # 1. Perform Training Step
            # Returns: (loss, tokens_per_sec, importance, relaxation_snapshots, generated_text)
            loss, tps, importance, snapshots, gen_text = self.model.training_step()

            # 2. Extract Visualization Data
            # Get latest activity from snapshots (last frame) or mean
            if snapshots:
                latest_activity = snapshots[-1]
            else:
                latest_activity = np.zeros_like(importance) # Fallback

            # Calculate Sparsity (percentage of tiles with low importance)
            sparsity = (importance < 0.1).float().mean().item()

            # 3. Emit Update Signal
            self.update_signal.emit(
                loss,
                tps,
                sparsity,
                importance.numpy(),
                latest_activity,
                gen_text
            )

            # Optional: Sleep to control update rate (avoid overwhelming UI)
            self.msleep(50) # ~20 FPS

    def stop(self):
        """Stop the worker."""
        self.running = False
        self.resume() # Wake up if paused
        self.wait()

    def pause(self):
        """Pause training."""
        self._mutex.lock()
        self.paused = True
        self._mutex.unlock()

    def resume(self):
        """Resume training."""
        self._mutex.lock()
        self.paused = False
        self._cond.wakeAll()
        self._mutex.unlock()
