import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox

from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool


class DreamWorker(QThread):
    """
    Background worker for 'Dreaming' (Input Optimization).
    Optimizes input image to maximize activation of target class.
    """

    progress = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self, model, target_class, input_shape, steps=100, lr=0.1, parent=None
    ):
        super().__init__(parent)
        self.model = model
        self.target = target_class
        self.shape = input_shape
        self.steps = steps
        self.lr = lr
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            device = next(self.model.parameters()).device
            # Start from gray noise
            x = torch.randn(1, *self.shape, device=device) * 0.1
            x.requires_grad_(True)

            optimizer = torch.optim.SGD([x], lr=self.lr)

            # Switch model to eval (we optimize x, not weights)
            self.model.eval()

            for i in range(self.steps):
                if self._stop:
                    break

                optimizer.zero_grad()

                # Forward pass
                # Handle models that might require steps arg
                try:
                    out = self.model(x)
                except TypeError:
                    out = self.model(x, steps=30)

                # Loss: Maximize target logit
                # We minimize negative target score
                loss = -out[0, self.target]

                loss.backward()
                optimizer.step()

                # Regularization / Constraints
                with torch.no_grad():
                    # Blur or jitter could be added here for better viz
                    x.clamp_(-2.5, 2.5)  # Assuming approx normalized range

                # Emit progress
                if i % 2 == 0:
                    img_np = x.detach().cpu().numpy().squeeze()
                    self.progress.emit(img_np)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            import traceback

            traceback.print_exc()


@ToolRegistry.register("dreaming", requires=["dreaming"])
class DreamingTool(BaseTool):
    ICON = "💤"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Dreaming Tool"))
        self.layout.addWidget(
            QLabel("Optimize input to maximize target class activation.")
        )

        # Controls
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Target Class:"))
        self.class_spin = QSpinBox()
        self.class_spin.setRange(0, 9)  # Assume 10 classes for now
        ctrl_layout.addWidget(self.class_spin)

        ctrl_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 500)
        self.steps_spin.setValue(100)
        ctrl_layout.addWidget(self.steps_spin)

        self.layout.addLayout(ctrl_layout)

        # Image Display
        self.img_label = QLabel()
        self.img_label.setMinimumSize(300, 300)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background-color: #000; border: 1px solid #333;")
        self.layout.addWidget(self.img_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Dreaming")
        self.start_btn.clicked.connect(self._start_dreaming)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_dreaming)
        btn_layout.addWidget(self.stop_btn)

        self.layout.addLayout(btn_layout)

        self.layout.addStretch()

    def _start_dreaming(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return

        # Determine shape
        # Simple heuristic
        ds_name = "mnist"
        if hasattr(self.model, "config") and hasattr(self.model.config, "dataset"):
            ds_name = self.model.config.dataset

        if "mnist" in ds_name:
            shape = (1, 28, 28)
            # Check model type for flattening
            use_flatten = True
            try:
                from bioplausible.models.registry import get_model_spec

                spec = get_model_spec(self.model.config.model)
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            if use_flatten:
                shape = (784,)
            else:
                shape = (1, 28, 28)

        elif "cifar" in ds_name or "svhn" in ds_name:
            use_flatten = True
            try:
                from bioplausible.models.registry import get_model_spec

                spec = get_model_spec(self.model.config.model)
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            if use_flatten:
                shape = (3072,)
            else:
                shape = (3, 32, 32)
        else:
            shape = (784,)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.class_spin.setEnabled(False)

        self.worker = DreamWorker(
            self.model, self.class_spin.value(), shape, steps=self.steps_spin.value()
        )
        self.worker.progress.connect(self._update_image)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_dreaming(self):
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()

    def _update_image(self, img):
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)

        # Handle shapes
        if img.ndim == 1:
            # Flattened, try to infer square
            d = int(np.sqrt(img.shape[0]))
            if d * d == img.shape[0]:
                img = img.reshape(d, d)
            elif img.shape[0] == 3072:  # CIFAR flattened
                img = img.reshape(3, 32, 32)
            else:
                # Fallback, just make it a line? Or square approximation
                w = int(np.ceil(np.sqrt(img.shape[0])))
                h = int(np.ceil(img.shape[0] / w))
                # Pad
                padded = np.zeros(h * w, dtype=img.dtype)
                padded[: img.shape[0]] = img
                img = padded.reshape(h, w)

        if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
            img = np.ascontiguousarray(img)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)

        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            300, 300, Qt.AspectRatioMode.KeepAspectRatio
        )
        self.img_label.setPixmap(pixmap)

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.class_spin.setEnabled(True)

    def _on_error(self, msg):
        self._on_finished()
        QMessageBox.warning(self, "Dream Error", msg)
