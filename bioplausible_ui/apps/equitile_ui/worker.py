import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

class TrainingWorker(QThread):
    """
    Worker thread for EquiTile training.

    Signals:
    - update_signal: Emitted on each training step with metrics and state.
    - tile_details_signal: Emitted when tile details are requested.
    """
    # Updated signature: importances/activities are lists of arrays
    update_signal = pyqtSignal(float, float, float, list, list, str) # loss, tokens_sec, sparsity, [importances], [activities], generated_text

    # Updated signature: include layer_id
    tile_details_signal = pyqtSignal(int, int, float, float, object) # layer_id, tile_id, importance, activity, neuron_states

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = False
        self.paused = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._pending_params = {}
        self._requested_tile = None # (layer_id, tile_id)

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
            # Returns: (loss, tokens_per_sec, list[importances], list[activities], generated_text)
            loss, tps, all_importances, all_activities, gen_text = self.model.training_step()

            # Calculate Global Sparsity
            total_tiles = sum(len(imp) for imp in all_importances)
            active_tiles = sum((imp > 0.1).sum() for imp in all_importances)
            sparsity = 1.0 - (active_tiles / max(1, total_tiles))

            self.update_signal.emit(
                loss, tps, sparsity, all_importances, all_activities, gen_text
            )

            # 3. Handle Inspection Request
            if self._requested_tile is not None:
                lid, tid = self._requested_tile
                imp, act, neurons = self.model.get_tile_details(lid, tid)
                self.tile_details_signal.emit(lid, tid, imp, act, neurons)

            self.msleep(50)

    def update_params(self, params):
        """Queue parameter updates."""
        self._pending_params.update(params)

    def request_tile_details(self, layer_id, tile_id):
        """Set the tile ID to inspect."""
        self._requested_tile = (layer_id, tile_id)

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
