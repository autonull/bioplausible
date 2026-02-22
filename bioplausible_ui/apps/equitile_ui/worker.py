import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition


class TrainingWorker(QThread):
    """Worker thread for EquiTile training with immediate feedback."""

    update_signal = pyqtSignal(float, float, float, float, float, float, list, list, str, list, list)
    # loss, tps, sparsity, train_acc, test_acc, perplexity, importances, activities, gen_text, tile_losses, gate_states

    tile_details_signal = pyqtSignal(int, int, float, float, object)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._pending_params = {}
        self._requested_tile = None
        self._stop_flag = False
        self._paused = True
        self._first_step = True

    def run(self):
        """Main training loop - emits on EVERY step initially for immediate feedback."""
        steps_since_update = 0
        update_threshold = 3  # Emit every 3 steps for responsive UI

        while True:
            # Check pause state
            self._mutex.lock()
            while self._paused and not self._stop_flag:
                self._cond.wait(self._mutex)

            if self._stop_flag:
                self._mutex.unlock()
                break
            self._mutex.unlock()

            # Apply Parameter Updates
            self._mutex.lock()
            params_copy = dict(self._pending_params) if self._pending_params else None
            self._pending_params = {}
            self._mutex.unlock()

            if params_copy:
                self.model.update_params(params_copy)

            # Perform Training Step
            loss, tps, train_acc, test_acc, perplexity, all_importances, all_activities, gen_text, tile_losses = self.model.training_step()

            # Extract gate states from model layers
            all_gate_states = []
            for layer in self.model.layers:
                if hasattr(layer, 'get_gate_state'):
                    gate_states, _ = layer.get_gate_state()
                    all_gate_states.append(gate_states)

            # Calculate Global Sparsity using model's threshold
            threshold = getattr(self.model.fast_config, 'sparsity_threshold', 0.1)
            total_tiles = sum(len(imp) for imp in all_importances)
            active_tiles = sum((imp > threshold).sum() for imp in all_importances)
            sparsity = 1.0 - (active_tiles / max(1, total_tiles))

            # Emit immediately on first few steps, then throttle
            if self._first_step:
                self.update_signal.emit(
                    loss, tps, sparsity, train_acc, test_acc, perplexity, all_importances, all_activities, gen_text, tile_losses, all_gate_states
                )
                self._first_step = False
            else:
                steps_since_update += 1
                if steps_since_update >= update_threshold:
                    self.update_signal.emit(
                        loss, tps, sparsity, train_acc, test_acc, perplexity, all_importances, all_activities, gen_text, tile_losses, all_gate_states
                    )
                    steps_since_update = 0

            # Handle Inspection Request
            self._mutex.lock()
            tile_request = self._requested_tile
            if tile_request:
                self._requested_tile = None
            self._mutex.unlock()

            if tile_request:
                lid, tid = tile_request
                imp, act, neurons = self.model.get_tile_details(lid, tid)
                self.tile_details_signal.emit(lid, tid, imp, act, neurons)

            # Minimal sleep - keep responsive
            self.msleep(1)

    def update_params(self, params):
        """Queue parameter updates."""
        self._mutex.lock()
        self._pending_params.update(params)
        self._mutex.unlock()

    def request_tile_details(self, layer_id, tile_id):
        """Set the tile ID to inspect."""
        self._mutex.lock()
        self._requested_tile = (layer_id, tile_id)
        self._mutex.unlock()

    def stop(self):
        """Stop the worker thread."""
        self._mutex.lock()
        self._stop_flag = True
        self._paused = False
        self._cond.wakeAll()
        self._mutex.unlock()
        self.wait(1000)

    def pause(self):
        """Pause training."""
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self):
        """Resume training."""
        self._mutex.lock()
        self._paused = False
        self._cond.wakeAll()
        self._mutex.unlock()

    @property
    def paused(self):
        """Check if paused (thread-safe)."""
        self._mutex.lock()
        val = self._paused
        self._mutex.unlock()
        return val
