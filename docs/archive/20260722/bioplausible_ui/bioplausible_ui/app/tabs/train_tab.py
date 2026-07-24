import numpy as np
import torch
from bioplausible.pipeline.config import TrainingConfig
from bioplausible.pipeline.session import SessionState
from bioplausible_ui.app.schemas.train import TRAIN_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.core.bridge import SessionBridge
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout


class InferenceDialog(QDialog):
    def __init__(self, task_type, input_data, prediction, truth=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Inference Check")
        self.resize(400, 500)
        layout = QVBoxLayout(self)

        # Header
        lbl = QLabel("Model Prediction")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(lbl)

        # Content based on task
        if task_type == "vision":
            # Input data is (1, C, H, W) or (1, H*W) tensor
            # Convert to QImage
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(300, 300)

            # Normalize and convert
            img = input_data.squeeze()
            if img.ndim == 1:  # Flattened
                d = int(np.sqrt(img.shape[0]))
                img = img.view(d, d)

            # img is now (H, W) or (C, H, W)
            if img.ndim == 3:
                img = img.permute(1, 2, 0)  # CHW -> HWC

            img_np = img.cpu().numpy()
            # Rescale 0-1 to 0-255
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
            img_np = (img_np * 255).astype(np.uint8)

            h, w = img_np.shape[:2]
            if img_np.ndim == 2:
                qimg = QImage(img_np.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                qimg = QImage(img_np.data, w, h, 3 * w, QImage.Format.Format_RGB888)

            pix = QPixmap.fromImage(qimg).scaled(
                300, 300, Qt.AspectRatioMode.KeepAspectRatio
            )
            img_lbl.setPixmap(pix)
            layout.addWidget(img_lbl)

            # Result
            res_layout = QVBoxLayout()
            pred_lbl = QLabel(f"Prediction: {prediction}")
            pred_lbl.setFont(QFont("Arial", 14))
            pred_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            res_layout.addWidget(pred_lbl)

            if truth is not None:
                truth_lbl = QLabel(f"Truth: {truth}")
                truth_lbl.setFont(QFont("Arial", 12))
                truth_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                res_layout.addWidget(truth_lbl)

                # Feedback
                correct = int(prediction) == int(truth)
                feed_lbl = QLabel("✅ CORRECT" if correct else "❌ INCORRECT")
                feed_lbl.setStyleSheet(
                    f"color: {'green' if correct else 'red'}; font-size: 18px; font-weight: bold;"
                )
                feed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                res_layout.addWidget(feed_lbl)

            layout.addLayout(res_layout)

        elif task_type == "lm":
            # Text display
            layout.addWidget(QLabel("Prompt & Completion:"))
            text_box = QLabel(prediction)  # Prediction here is full text
            text_box.setWordWrap(True)
            text_box.setStyleSheet(
                "background-color: #222; color: #0f0; padding: 10px; font-family: monospace;"
            )
            layout.addWidget(text_box)

        else:
            layout.addWidget(QLabel(f"Output: {prediction}"))

        # Close
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class TrainTab(BaseTab):
    """Training tab - UI auto-generated from schema."""

    SCHEMA = TRAIN_TAB_SCHEMA

    def _post_init(self):
        if "stop" in self._actions:
            self._actions["stop"].setEnabled(False)

    def _start_training(self):
        try:
            train_config = self.training_config.get_values()
            config = TrainingConfig(
                task=self.task_selector.get_task(),
                dataset=self.dataset_picker.get_dataset(),
                model=self.model_selector.get_selected_model(),
                epochs=train_config.get("epochs", 10),
                batch_size=train_config.get("batch_size", 64),
                gradient_method=train_config.get("gradient_method", "BPTT (Standard)"),
                use_compile=train_config.get("use_compile", True),
                use_kernel=train_config.get("use_kernel", False),
                monitor_dynamics=train_config.get("monitor_dynamics", False),
                hyperparams=self.hyperparam_editor.get_values(),
            )
            self.bridge = SessionBridge(config)
            self.bridge.progress_updated.connect(self._on_progress)
            self.bridge.training_completed.connect(self._on_completed)
            self.bridge.start()

            # Disable start, enable stop
            self._actions["start"].setEnabled(False)
            self._actions["stop"].setEnabled(True)

            # Clear plots
            self.plot_loss.clear()
            self.plot_accuracy.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            import traceback

            traceback.print_exc()

    def _stop_training(self):
        if hasattr(self, "bridge"):
            self.bridge.stop()
            self._actions["start"].setEnabled(True)
            self._actions["stop"].setEnabled(False)

    def _on_progress(self, epoch, metrics):
        self.plot_loss.add_point(epoch, metrics.get("loss", 0))
        self.plot_accuracy.add_point(epoch, metrics.get("accuracy", 0))

        # Log rich metrics
        metric_str = " | ".join([
            f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        ])
        self.log_output.append(f"Epoch {epoch}: {metric_str}")

    def _on_completed(self, final_metrics):
        QMessageBox.information(
            self,
            "Training Complete",
            f"Final Accuracy: {final_metrics.get('accuracy', 0):.4f}",
        )
        self._actions["start"].setEnabled(True)
        self._actions["stop"].setEnabled(False)

    def _test_model(self):
        if not hasattr(self, "bridge") or not self.bridge.session.model:
            QMessageBox.warning(
                self, "Warning", "No model available. Train a model first."
            )
            return

        # Ensure not running
        if self.bridge.session.state == SessionState.RUNNING:
            QMessageBox.warning(self, "Warning", "Please stop training before testing.")
            return

        try:
            session = self.bridge.session
            model = session.model
            task = session.task

            model.eval()

            # Get sample
            x, y = task.get_batch("val")  # Returns batch

            # Take first item
            x_sample = x[0:1]
            y_sample = y[0:1] if y is not None else None

            # Inference
            with torch.no_grad():
                # Prepare input logic similar to Trainer
                h = x_sample
                # Simple prep for common cases
                if x_sample.dim() == 4 and "MLP" in model.__class__.__name__:
                    h = x_sample.view(1, -1)

                h = h.to(session.device)
                logits = model(h)

                if session.config.task == "vision":
                    pred = logits.argmax(1).item()
                    truth = y_sample.item()
                    dialog = InferenceDialog("vision", x_sample, pred, truth, self)
                    dialog.exec()

                elif session.config.task == "lm":
                    # Generate text if supported
                    if hasattr(model, "generate"):
                        # Use simple generation
                        start_tokens = x_sample[0, :5]  # Seed with 5 tokens
                        gen_text = model.generate(
                            start_tokens.unsqueeze(0), max_new_tokens=20
                        )
                        # Decode
                        # Task implementations should provide decoding logic.
                        # For now, we display raw tokens.
                        text = f"Raw Tokens: {gen_text.tolist()}"
                        dialog = InferenceDialog("lm", None, text, None, self)
                        dialog.exec()
                    else:
                        QMessageBox.information(
                            self, "Result", f"Logits shape: {logits.shape}"
                        )

                elif session.config.task == "rl":
                    # RL typically returns Q-values
                    q_vals = logits
                    action = q_vals.argmax().item()
                    QMessageBox.information(
                        self,
                        "RL Action",
                        f"State: {x_sample.shape}\nPredicted Action: {action}\nQ-Values: {q_vals.cpu().numpy()}",
                    )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Inference failed: {e}")
            import traceback

            traceback.print_exc()

    def _analyze_model(self):
        if not hasattr(self, "bridge") or not self.bridge.session.model:
            QMessageBox.warning(
                self, "Warning", "No model available. Train a model first."
            )
            return

        from bioplausible.models.registry import get_model_spec
        from bioplausible_ui.lab.window import LabMainWindow

        session = self.bridge.session
        spec = get_model_spec(session.config.model)

        self.lab_window = LabMainWindow()
        self.lab_window.load_model_instance(session.model, spec)
        self.lab_window.show()

    def set_config(self, config):
        """Populate UI from config dict."""
        if "task" in config:
            self.task_selector.combo.setCurrentText(config["task"])
        if "dataset" in config:
            self.dataset_picker.combo.setCurrentText(config["dataset"])
        if "model" in config:
            self.model_selector.combo.setCurrentText(config["model"])

        # Hyperparams need model to be set first to initialize fields
        if "hyperparams" in config and hasattr(self, "hyperparam_editor"):
            # Ensure editor is updated for model
            if "model" in config:
                self.hyperparam_editor.update_for_model(config["model"])

            self.hyperparam_editor.set_values(config["hyperparams"])

        # Set training config values (epochs, batch_size, learning_rate, etc)
        # These can be in 'config' top level or mixed in 'hyperparams' depending on source
        # We merge them to be safe
        tc_values = {}
        for k, v in config.items():
            if k in [
                "epochs",
                "batch_size",
                "learning_rate",
                "gradient_method",
                "use_compile",
                "use_kernel",
                "monitor_dynamics",
                "gamma",
                "seq_len",
            ]:
                tc_values[k] = v

        if hasattr(self, "training_config"):
            self.training_config.set_values(tc_values)
