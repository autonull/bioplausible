import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition


class TrainingWorker(QThread):
    """
    Worker thread for training.
    Generic: Emits whatever the model wrapper returns.
    """

    # Generic signal: emits a dictionary of metrics/data
    update_signal = pyqtSignal(dict)

    # Generic inspection signal
    tile_details_signal = pyqtSignal(int, int, float, float, object)

    def __init__(self, model_wrapper):
        super().__init__()
        self.wrapper = model_wrapper
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._pending_params = {}
        self._requested_tile = None
        self._stop_flag = False
        self._paused = True
        self._first_step = True

    def run(self):
        """Main training loop."""
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
                self.wrapper.update_params(params_copy)

            # Perform Training Step via Wrapper
            # This returns a dict with 'loss', 'tps', 'activities', etc.
            step_data = self.wrapper.training_step()

            # Emit immediately on first few steps, then throttle
            if self._first_step:
                self.update_signal.emit(step_data)
                self._first_step = False
            else:
                steps_since_update += 1
                if steps_since_update >= update_threshold:
                    self.update_signal.emit(step_data)
                    steps_since_update = 0

            # Handle Inspection Request
            self._mutex.lock()
            tile_request = self._requested_tile
            if tile_request:
                self._requested_tile = None
            self._mutex.unlock()

            if tile_request:
                lid, tid = tile_request
                # Wrapper needs get_tile_details? Or just generic logic?
                # Let's assume wrapper has get_tile_details or we handle it here?
                # The generic wrapper didn't implement get_tile_details yet.
                # I should add it to wrapper or handle it here if possible.
                # But activities are in wrapper.

                # For generic visualization, we might just pass empty details or implement it.
                # Let's add get_tile_details to wrapper later or now.
                # For now, if wrapper has it, call it.
                if hasattr(self.wrapper, 'get_tile_details'):
                    imp, act, neurons, is_active = self.wrapper.get_tile_details(lid, tid)
                    self.tile_details_signal.emit(lid, tid, imp, act, neurons)
                else:
                    # Fallback
                    self.tile_details_signal.emit(lid, tid, 1.0, 0.0, np.zeros(10))

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
