import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QSpinBox, QVBoxLayout)

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool


class DiffusionSamplingWorker(QThread):
    finished = pyqtSignal(np.ndarray)  # Returns grid image
    error = pyqtSignal(str)

    def __init__(self, model, num_samples=16, device="cuda"):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.device = device

    def run(self):
        try:
            # Run sampling
            # samples: [B, C, H, W] in [-1, 1]
            samples = self.model.sample(
                num_samples=self.num_samples,
                img_size=(1, 28, 28),  # Hardcoded for MNIST for now
                device=self.device,
            )

            # Convert to numpy grid
            # Normalize to [0, 255]
            samples = (samples + 1) / 2.0
            samples = samples.clamp(0, 1)

            # Create grid (e.g. 4x4)
            grid_size = int(np.ceil(np.sqrt(self.num_samples)))
            B, C, H, W = samples.shape

            grid_h = grid_size * H
            grid_w = grid_size * W
            grid = torch.zeros(C, grid_h, grid_w)

            for i in range(B):
                row = i // grid_size
                col = i % grid_size
                grid[:, row * H : (row + 1) * H, col * W : (col + 1) * W] = samples[i]

            # To Numpy [H, W] (assuming grayscale)
            img_np = grid.squeeze(0).cpu().numpy()
            self.finished.emit(img_np)

        except Exception as e:
            self.error.emit(str(e))


@ToolRegistry.register("Diffusion Sampling", requires=["diffusion_sample"])
class DiffusionSamplingTool(BaseTool):
    ICON = "🌫️"

    def init_ui(self):
        super().init_ui()

        # Controls
        controls = QGroupBox("Configuration")
        ctrl_layout = QVBoxLayout(controls)

        row = QHBoxLayout()
        row.addWidget(QLabel("Samples:"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 64)
        self.samples_spin.setValue(16)
        row.addWidget(self.samples_spin)
        ctrl_layout.addLayout(row)

        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self._generate)
        ctrl_layout.addWidget(self.gen_btn)

        self.layout.addWidget(controls)

        # Display
        display_group = QGroupBox("Generated Samples")
        display_layout = QVBoxLayout(display_group)

        if HAS_PYQTGRAPH:
            self.img_view = pg.ImageView()
            self.img_view.ui.histogram.hide()
            self.img_view.ui.roiBtn.hide()
            self.img_view.ui.menuBtn.hide()
            display_layout.addWidget(self.img_view)
        else:
            display_layout.addWidget(QLabel("PyQtGraph not installed."))

        self.layout.addWidget(display_group)

    def _generate(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "No model loaded.")
            return

        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("Generating...")

        device = next(self.model.parameters()).device

        self.gen_worker = DiffusionSamplingWorker(
            self.model, num_samples=self.samples_spin.value(), device=device
        )
        self.gen_worker.finished.connect(self._on_finished)
        self.gen_worker.error.connect(self._on_error)
        self.gen_worker.start()

    def _on_finished(self, img):
        if HAS_PYQTGRAPH:
            # Transpose for pyqtgraph (W, H)
            self.img_view.setImage(img.T)
            self.img_view.autoRange()

        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate")

    def _on_error(self, err):
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate")
        QMessageBox.critical(self, "Generation Error", err)
