import os

import torch
import uvicorn
from bioplausible_ui.app.schemas.deploy import DEPLOY_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bioplausible.export import export_to_onnx, export_to_torchscript
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec
from bioplausible.pipeline.results import ResultsManager


class ExportWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, format, path, input_sample, parent=None):
        super().__init__(parent)
        self.model = model
        self.format = format
        self.path = path
        self.input_sample = input_sample

    def run(self):
        try:
            if self.format == "onnx":
                export_to_onnx(self.model, self.input_sample, self.path)
            elif self.format == "torchscript":
                export_to_torchscript(self.model, self.input_sample, self.path)
            self.finished.emit(f"Successfully exported to {self.path}")
        except Exception as e:
            self.error.emit(str(e))


class ServerWorker(QThread):
    def __init__(self, model, host="0.0.0.0", port=8000):
        super().__init__()
        self.model = model
        self.host = host
        self.port = port
        self.server = None

    def run(self):
        import bioplausible.export as export

        export.model_instance = self.model
        if self.model:
            export.model_instance.eval()
            # Move to CPU for serving typically
            export.model_instance.cpu()

        config = uvicorn.Config(
            export.app, host=self.host, port=self.port, log_level="info"
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        if self.server:
            self.server.should_exit = True
            self.wait()


class DeployTab(BaseTab):
    """Deploy tab - UI auto-generated from schema."""

    SCHEMA = DEPLOY_TAB_SCHEMA

    def _post_init(self):
        self.server_worker = None
        self.results_manager = ResultsManager()

    def _refresh_runs(self):
        self.run_selector.refresh()

    def _load_model_from_run(self):
        run_id = self.run_selector.get_run_id()
        if not run_id:
            QMessageBox.warning(self, "Warning", "Please select a run.")
            return None

        # Load metadata
        run_data = self.results_manager.load_run(run_id)
        if not run_data:
            QMessageBox.critical(self, "Error", "Could not load run data.")
            return None

        config = run_data.get("config", {})
        model_name = config.get("model")
        config.get("task", "vision")
        hyperparams = config.get("hyperparams", {})

        # Recreate model
        try:
            spec = get_model_spec(model_name)
            # We need input/output dim.
            # In headless load, we might not have dataset loaded.
            # We can infer from task defaults or save dims in config?
            # Creating task to get dims is safest.
            from bioplausible.hyperopt.tasks import create_task

            dataset = config.get("dataset", "mnist")
            task_obj = create_task(dataset, device="cpu", quick_mode=True)
            task_obj.setup()

            model = create_model(
                spec=spec,
                input_dim=task_obj.input_dim,
                output_dim=task_obj.output_dim,
                device="cpu",  # Export on CPU usually
                task_type=task_obj.task_type,
                **hyperparams,
            )

            # Load weights
            weights_path = os.path.join(
                self.results_manager.BASE_DIR, run_id, "model.pt"
            )
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"Loaded weights from {weights_path}")
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No weights found for this run. Exporting initialized model.",
                )

            return model, task_obj.input_dim

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _export_model(self):
        loaded = self._load_model_from_run()
        if not loaded:
            return
        model, input_dim = loaded

        fmt = self.format_selector.get_format()  # e.g. "onnx"
        ext = "onnx" if "onnx" in fmt else "pt"

        fname, _ = QFileDialog.getSaveFileName(
            self, "Export Model", f"model.{ext}", f"{fmt.upper()} (*.{ext})"
        )
        if not fname:
            return

        # Prepare dummy input
        # If input_dim is int -> (1, dim). If None (LM) -> (1, 64) long
        try:
            if input_dim is None:  # Likely LM
                shape = (1, 64)
                input_sample = torch.randint(0, 100, shape)
            elif isinstance(input_dim, int):
                # Use flat input shape (1, input_dim).
                # Models that require spatial dimensions (e.g., ConvNets) typically handle reshaping internally
                # or accept flattened input if specified.
                shape = (1, input_dim)
                input_sample = torch.randn(shape)

            self.log_output.append("Exporting...")
            self.worker = ExportWorker(model, fmt, fname, input_sample)
            self.worker.finished.connect(
                lambda msg: QMessageBox.information(self, "Success", msg)
            )
            self.worker.error.connect(
                lambda err: QMessageBox.critical(self, "Error", err)
            )
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Input preparation failed: {e}")

    def _serve_model(self):
        loaded = self._load_model_from_run()
        if not loaded:
            return
        model, _ = loaded

        if self.server_worker and self.server_worker.isRunning():
            self.server_worker.stop()
            self.log_output.append("Server stopped.")
            self._actions["serve"].setText("🚀")  # Reset icon?
            # Schema actions don't update icon easily, but text changes might persist if we access widget
            # Actually ActionDef creates a QAction or button.
            # We can find button?
            return

        self.log_output.append("Starting server on port 8000...")
        self.server_worker = ServerWorker(model)
        self.server_worker.start()

        # Open browser
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("http://localhost:8000/docs"))
