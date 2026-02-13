import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class OracleWorker(QThread):
    """
    Background worker for Oracle Metric analysis.
    Measures settling time vs uncertainty (noise).
    """

    finished = pyqtSignal(list)  # List of (noise, settling_time) tuples
    error = pyqtSignal(str)

    def __init__(self, model, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset_name = dataset_name

    def run(self):
        try:
            from torch.utils.data import DataLoader

            from bioplausible.datasets import get_vision_dataset
            from bioplausible.models.registry import get_model_spec

            # Setup data
            ds_name = self.dataset_name.lower().replace("-", "_")
            use_flatten = True
            try:
                # Infer flattening from model if possible, otherwise assume flat for MLP
                spec = get_model_spec(
                    self.model.config.name if hasattr(self.model, "config") else ""
                )
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            dataset = get_vision_dataset(ds_name, train=False, flatten=use_flatten)
            # Small subset
            indices = np.random.choice(len(dataset), 50, replace=False)
            subset = torch.utils.data.Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=10, shuffle=False)

            device = next(self.model.parameters()).device
            self.model.eval()

            results = []
            noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

            with torch.no_grad():
                for noise in noise_levels:
                    total_steps = 0
                    count = 0

                    for x, y in loader:
                        x = x.to(device)
                        if noise > 0:
                            x = x + torch.randn_like(x) * noise

                        # Run with dynamics tracking
                        # Ensure we request dynamics
                        try:
                            # Pass return_dynamics=True if supported
                            out, dynamics = self.model(x, return_dynamics=True)

                            deltas = dynamics.get("deltas", [])
                            if deltas:
                                # Find step where delta drops below threshold (e.g. 1e-3)
                                # Normalize delta?
                                threshold = 1e-2  # Heuristic
                                settled_at = len(deltas)
                                for i, d in enumerate(deltas):
                                    if d < threshold:
                                        settled_at = i + 1
                                        break
                                total_steps += settled_at
                            else:
                                total_steps += 30  # Max

                        except (TypeError, ValueError):
                            # Fallback if model doesn't support dynamics
                            # Just run forward
                            self.model(x)
                            total_steps += 30  # Assume max

                        count += 1

                    avg_steps = total_steps / count if count > 0 else 30
                    results.append((noise, avg_steps))

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
            import traceback

            traceback.print_exc()


class OracleDialog(QDialog):
    """Dialog for Oracle Metric (Uncertainty Analysis)."""

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Oracle Metric: Uncertainty vs Time")
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        header = QLabel("🔮 Oracle Analysis")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        desc = QLabel(
            "Visualizing the correlation between Uncertainty (Noise) and Processing Time (Settling Steps).\n"
            "A Bio-Plausible network should take longer to resolve ambiguous inputs."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a0b0;")
        layout.addWidget(desc)

        if HAS_PYQTGRAPH:
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("#0a0a0f")
            plot_widget.setLabel("left", "Settling Time (Steps)")
            plot_widget.setLabel("bottom", "Noise Level (σ)")
            plot_widget.showGrid(x=True, y=True, alpha=0.3)

            # Plot data
            x = [r[0] for r in results]
            y = [r[1] for r in results]

            # Plot points and line
            plot_widget.plot(
                x,
                y,
                symbol="o",
                pen=pg.mkPen("#00d4ff", width=3),
                symbolBrush="#00d4ff",
            )

            layout.addWidget(plot_widget)

            # Calculate correlation
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                corr_label = QLabel(f"Correlation: {corr:.3f}")
                corr_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                if corr > 0.5:
                    corr_label.setStyleSheet(
                        "color: #00ff88;"
                    )  # Good positive correlation
                else:
                    corr_label.setStyleSheet("color: #f1c40f;")
                layout.addWidget(corr_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


@ToolRegistry.register("oracle", requires=["oracle"])
class OracleTool(BaseTool):
    ICON = "🔮"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Oracle Tool"))
        self.layout.addWidget(
            QLabel("Analyze uncertainty by correlating input noise with settling time.")
        )

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(
            ["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"]
        )
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_combo)

        self.run_btn = QPushButton("Run Oracle Analysis")
        self.run_btn.clicked.connect(self._run_oracle)
        self.layout.addWidget(self.run_btn)

        self.layout.addStretch()

    def _run_oracle(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Analyzing...")

        dataset = self.dataset_combo.currentText()
        self.worker = OracleWorker(self.model, dataset)
        self.worker.finished.connect(self._show_results)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _show_results(self, results):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Oracle Analysis")
        dlg = OracleDialog(results, self)
        dlg.exec()

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Oracle Analysis")
        QMessageBox.critical(self, "Error", msg)
