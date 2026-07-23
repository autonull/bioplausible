from PyQt6.QtCore import QThread, pyqtSignal


class BenchmarkWorker(QThread):
    """Worker for running validation tracks (benchmarks)."""

    log_message = pyqtSignal(
        str
    )  # Renamed from progress to match usage in BenchmarksTab
    finished = pyqtSignal()  # Renamed/Simplified
    error = pyqtSignal(str)

    def __init__(self, track_id, parallel=False, parent=None):
        super().__init__(parent)
        self.track_id = track_id
        self.parallel = parallel
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            from bioplausible.validation.core import Verifier

            verifier = Verifier(quick_mode=False, seed=42)  # Use standard mode

            self.log_message.emit(
                f"Starting Benchmark (Track={self.track_id}, Parallel={self.parallel})..."
            )

            track_ids = [self.track_id] if self.track_id is not None else None

            if (
                self.parallel and track_ids is None
            ):  # Parallel only makes sense for all tracks
                self.log_message.emit(
                    "Running in parallel mode. Check console for details."
                )
                verifier.run_tracks(track_ids, parallel=True)
            else:
                # Use sequential signal-based verifier for rich updates
                class SignalVerifier(Verifier):
                    def __init__(self, worker, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.worker = worker

                    def run_tracks(self, track_ids, parallel=False):
                        if parallel:
                            return super().run_tracks(track_ids, parallel=True)
                        if track_ids is None:
                            track_ids = sorted(self.tracks.keys())

                        results = {}
                        for track_id in track_ids:
                            if self.worker._stop_requested:
                                break

                            self.worker.log_message.emit(f"Running Track {track_id}...")

                            try:
                                name, method = self.tracks[track_id]
                                result = method(self)
                                results[track_id] = result

                                status_icon = "✅" if result.status == "pass" else "❌"
                                self.worker.log_message.emit(
                                    f"{status_icon} Track {track_id}: {result.status.upper()} ({result.score}/100)"
                                )

                            except Exception as e:
                                self.worker.log_message.emit(
                                    f"❌ Track {track_id} Failed: {e}"
                                )

                        return results

                # Swap instance
                verifier = SignalVerifier(self, quick_mode=False, seed=42)
                verifier.run_tracks(track_ids, parallel=False)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.log_message.emit(f"Error: {e}")
            import traceback

            traceback.print_exc()
