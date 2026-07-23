from PyQt6.QtWidgets import QMessageBox

from bioplausible_ui.app.schemas.benchmarks import BENCHMARKS_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.core.workers import BenchmarkWorker


class BenchmarksTab(BaseTab):
    SCHEMA = BENCHMARKS_TAB_SCHEMA

    def _run_benchmarks(self):
        track_id = self.track_selector.get_selected_track_id()
        parallel = self.parallel_check.isChecked()

        self.results_list.clear()
        self.results_list.addItem(
            f"Starting benchmark (Track: {track_id}, Parallel: {parallel})..."
        )

        self.worker = BenchmarkWorker(
            track_id if track_id != -1 else None, parallel=parallel
        )
        self.worker.log_message.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

        self._actions["run"].setEnabled(False)

    def _log(self, message):
        self.results_list.addItem(message)
        self.results_list.scrollToBottom()

    def _on_finished(self):
        self.results_list.addItem("Benchmarks Completed.")
        self.results_list.scrollToBottom()
        self._actions["run"].setEnabled(True)

        # Check for success (simple heuristic from logs)
        # Or better, update worker to emit results
        QMessageBox.information(self, "Benchmarks", "Execution finished.")

    def _clear_logs(self):
        self.results_list.clear()
