import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

class TrainingWorker(QThread):
    """
    Worker thread for EquiTile training.

    Signals:
    - update_signal: Emitted on each training step with metrics and state.
    - tile_details_signal: Emitted when tile details are requested.
    """
    update_signal = pyqtSignal(float, float, float, object, object, str) # loss, tokens_sec, sparsity, importance, activity, generated_text
    tile_details_signal = pyqtSignal(int, float, float, object) # tile_id, importance, activity, neuron_states

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = False
        self.paused = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._pending_params = {}
        self._requested_tile_id = None

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

            # 0. Apply Parameter Updates
            if self._pending_params:
                self.model.update_params(self._pending_params)
                self._pending_params = {}

            # 1. Perform Training Step
            loss, tps, importance, snapshots, gen_text = self.model.training_step()

            # 2. Extract Visualization Data
            if snapshots:
                latest_activity = snapshots[-1]
            else:
                latest_activity = np.zeros_like(importance)

            sparsity = (importance < 0.1).float().mean().item()

            self.update_signal.emit(
                loss, tps, sparsity, importance.numpy(), latest_activity, gen_text
            )

            # 3. Handle Inspection Request
            if self._requested_tile_id is not None:
                tid = self._requested_tile_id
                imp, act, neurons = self.model.get_tile_details(tid)
                self.tile_details_signal.emit(tid, imp, act, neurons)
                # Reset request to avoid flooding (or keep updating if we want live view)
                # For live view, we don't reset.

            self.msleep(50)

    def update_params(self, params):
        """Queue parameter updates."""
        self._pending_params.update(params)

    def request_tile_details(self, tile_id):
        """Set the tile ID to inspect."""
        self._requested_tile_id = tile_id

    def stop(self):
        self.running = False
        self.resume()
        self.wait()

    def pause(self):
        self._mutex.lock()
        self.paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self.paused = False
        self._cond.wakeAll()
        self._mutex.unlock()
