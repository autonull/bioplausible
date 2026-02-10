from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bioplausible.pipeline.results import ResultsManager
from bioplausible_ui.app.schemas.results import RESULTS_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab


class ResultsTab(BaseTab):
    """Results tab - UI auto-generated from schema."""

    SCHEMA = RESULTS_TAB_SCHEMA

    def _post_init(self):
        self.results_manager = ResultsManager()
        self._refresh_results()

    def _refresh_results(self):
        self.results_table.clear_table()
        runs = self.results_manager.list_runs()

        for run in runs:
            config = run.get("config", {})
            metrics = run.get("metrics", {})

            # Handle nested structure (history support)
            if "final_metrics" in metrics:
                metrics = metrics["final_metrics"]

            # Determine main metric
            metric_val = metrics.get("accuracy", 0.0)
            if "loss" in metrics and metric_val == 0.0:
                metric_val = metrics.get("loss", 0.0)  # Fallback

            self.results_table.add_run(
                run_id=run.get("run_id", "???"),
                timestamp=run.get("timestamp", "")[:19].replace("T", " "),
                task=config.get("task", "unknown"),
                model=config.get("model", "unknown"),
                metric_val=metric_val,
            )

    def _delete_run(self):
        run_id = self.results_table.get_selected_run_id()
        if run_id:
            confirm = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete run {run_id}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if confirm == QMessageBox.StandardButton.Yes:
                self.results_manager.delete_run(run_id)
                self._refresh_results()
        else:
            QMessageBox.warning(self, "Warning", "Please select a run to delete.")

    def _analyze_run(self):
        run_id = self.results_table.get_selected_run_id()
        if not run_id:
            QMessageBox.warning(self, "Warning", "Please select a run to analyze.")
            return

        import os

        from bioplausible_ui.lab.window import LabMainWindow

        # We need the path to model.pt
        model_path = os.path.join(self.results_manager.BASE_DIR, run_id, "model.pt")
        if not os.path.exists(model_path):
            QMessageBox.warning(
                self, "Warning", "Model weights not found for this run."
            )
            return

        self.lab_window = LabMainWindow(model_path)
        self.lab_window.show()

    def _export_run(self):
        run_id = self.results_table.get_selected_run_id()
        if not run_id:
            QMessageBox.warning(self, "Warning", "Please select a run to export.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, "Export Run", f"{run_id}.zip", "Zip Files (*.zip)"
        )
        if fname:
            try:
                self.results_manager.export_run(run_id, fname)
                QMessageBox.information(self, "Success", f"Run exported to {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _import_run(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Import Run", "", "Zip Files (*.zip)"
        )
        if fname:
            try:
                run_id = self.results_manager.import_run(fname)
                QMessageBox.information(self, "Success", f"Imported run {run_id}")
                self._refresh_results()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
