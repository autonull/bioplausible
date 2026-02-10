import torch
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QTabWidget

import bioplausible_ui.lab.tools  # Register all tools
from bioplausible.models.registry import get_model_spec
from bioplausible_ui.core.themes import Theme
from bioplausible_ui.lab.registry import ToolRegistry


class LabMainWindow(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.setWindowTitle("Bioplausible Lab (biopl-lab)")
        self.resize(1200, 800)
        self.setStyleSheet(Theme.get_stylesheet())

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Menu
        menu = self.menuBar().addMenu("File")
        menu.addAction("Load Model", self.load_model_dialog)

        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Model Checkpoint", "", "PyTorch Checkpoints (*.pt)"
        )
        if fname:
            self.load_model(fname)

    def load_model(self, path):
        try:
            checkpoint = torch.load(path)
            # Support both structure types if necessary, but typically saved as state_dict with metadata separate or bundled
            # ResultsManager saves metadata.json and model.pt (state dict).
            # If path points to model.pt, we might need metadata from json.

            import json
            import os

            from bioplausible.hyperopt.tasks import create_task
            from bioplausible.models.factory import create_model

            dir_path = os.path.dirname(path)
            meta_path = os.path.join(dir_path, "metadata.json")

            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                config = meta.get("config", {})
            else:
                # Fallback: maybe embedded in checkpoint?
                config = checkpoint.get("config", {})

            model_name = config.get("model")
            if not model_name:
                raise ValueError("Model name not found in configuration.")

            spec = get_model_spec(model_name)

            # Recreate Task to get dims
            task_name = config.get("task", "vision")  # Default
            dataset = config.get("dataset", "mnist")

            # Create dummy task for dims
            task_obj = create_task(dataset, device="cpu", quick_mode=True)
            task_obj.setup()

            hyperparams = config.get("hyperparams", {})

            model = create_model(
                spec=spec,
                input_dim=task_obj.input_dim,
                output_dim=task_obj.output_dim,
                device="cpu",
                task_type=task_obj.task_type,
                **hyperparams,
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                # Might be direct state dict
                try:
                    model.load_state_dict(checkpoint)
                except:
                    pass  # best effort

            self.load_model_instance(model, spec)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            import traceback

            traceback.print_exc()

    def load_model_instance(self, model, spec):
        self.model = model
        self.tabs.clear()

        tools = ToolRegistry.get_compatible_tools(spec)
        if not tools:
            QMessageBox.information(
                self, "Info", "No compatible tools found for this model."
            )

        for tool_name in tools:
            try:
                ToolClass = ToolRegistry.get_tool_class(tool_name)
                # Some tools might need specific init args?
                # BaseTool takes (model, parent)
                tool = ToolClass(self.model)
                self.tabs.addTab(tool, f"{tool.ICON} {tool_name.title()}")
            except Exception as e:
                print(f"Failed to load tool {tool_name}: {e}")
