import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@ToolRegistry.register("cube_viz", requires=["cube_viz"])
class CubeVizTool(BaseTool):
    ICON = "🧊"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Cube Visualizer"))
        self.layout.addWidget(QLabel("Visualize 3D Neural Cube topology."))

        self.viz_btn = QPushButton("Launch Visualizer")
        self.viz_btn.clicked.connect(self._launch_viz)
        self.layout.addWidget(self.viz_btn)

        # Viz container
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.layout.addWidget(self.viz_container)
        self.viz_container.hide()

        self.layout.addStretch()

    def _launch_viz(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return

        try:
            # Run inference on one sample to get state
            from bioplausible.datasets import get_vision_dataset

            # Use MNIST as default if not known, or try to guess
            ds_name = "mnist"
            if hasattr(self.model, "config") and hasattr(self.model.config, "dataset"):
                ds_name = self.model.config.dataset

            # NeuralCube takes flattened input
            dataset = get_vision_dataset(ds_name, train=False, flatten=True)

            # Get random sample
            idx = np.random.randint(0, len(dataset))
            x, _ = dataset[idx]
            x = torch.tensor(x).unsqueeze(0).to(next(self.model.parameters()).device)

            # Run forward with trajectory
            self.model.eval()
            with torch.no_grad():
                # NeuralCube forward returns out or (out, traj)
                out, traj = self.model(x, return_trajectory=True)
                h_final = traj[-1]  # [1, n_neurons]

            # Show Viz
            self._show_viz(h_final, getattr(self.model, "cube_size", 10))

        except Exception as e:
            QMessageBox.critical(self, "Viz Error", str(e))
            import traceback

            traceback.print_exc()

    def _show_viz(self, h_tensor, cube_size):
        self.h = h_tensor.cpu().reshape(cube_size, cube_size, cube_size).numpy()
        self.cube_size = cube_size

        # Clear previous viz
        for i in reversed(range(self.viz_layout.count())):
            self.viz_layout.itemAt(i).widget().setParent(None)

        self.viz_container.show()
        self.viz_btn.hide()

        header = QLabel(
            f"3D Activation Topography ({cube_size}x{cube_size}x{cube_size})"
        )
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_layout.addWidget(header)

        # Slice control
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Z-Slice:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, cube_size - 1)
        self.slider.setValue(cube_size // 2)
        self.slider.valueChanged.connect(self._update_slice)
        slice_layout.addWidget(self.slider)
        self.slice_label = QLabel(f"{cube_size // 2}")
        slice_layout.addWidget(self.slice_label)
        self.viz_layout.addLayout(slice_layout)

        if HAS_PYQTGRAPH:
            self.img_view = pg.ImageView()
            self.img_view.ui.histogram.hide()
            self.img_view.ui.roiBtn.hide()
            self.img_view.ui.menuBtn.hide()
            # Set colormap (Fire)
            self.img_view.setPredefinedGradient("thermal")
            self.viz_layout.addWidget(self.img_view)

            self._update_slice(self.slider.value())

            # Add close button
            close_btn = QPushButton("Close Visualization")
            close_btn.clicked.connect(self._close_viz)
            self.viz_layout.addWidget(close_btn)

        else:
            self.viz_layout.addWidget(QLabel("PyQtGraph required for visualization"))

    def _update_slice(self, z):
        self.slice_label.setText(str(z))
        if HAS_PYQTGRAPH and hasattr(self, "img_view"):
            # Get slice z
            # Shape [Z, Y, X]
            slice_data = self.h[z, :, :]
            self.img_view.setImage(slice_data.T)  # Transpose for display

    def _close_viz(self):
        self.viz_container.hide()
        self.viz_btn.show()
