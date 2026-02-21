from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QProgressBar, QGroupBox, QGridLayout, QSplitter)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np

class DashboardPanel(QWidget):
    """
    Real-time dashboard for EquiTile metrics.
    Displays metrics, text generation, and layer-wise analysis.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Splitter: Top (Global Metrics) / Bottom (Layer Analysis)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Top: Global Metrics & Text ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0,0,0,0)

        # Plots Grid (Loss, Speed, Sparsity)
        plots_layout = QGridLayout()

        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.configure_plot(self.loss_plot, "Loss", "#ff00ff")
        self.loss_data = []
        plots_layout.addWidget(self.loss_plot, 0, 0)

        self.speed_plot = pg.PlotWidget(title="Throughput")
        self.configure_plot(self.speed_plot, "Tok/s", "#00ff88")
        self.speed_data = []
        plots_layout.addWidget(self.speed_plot, 0, 1)

        self.sparsity_plot = pg.PlotWidget(title="Global Sparsity")
        self.configure_plot(self.sparsity_plot, "%", "#00ccff")
        self.sparsity_plot.setYRange(0, 100)
        self.sparsity_data = []
        plots_layout.addWidget(self.sparsity_plot, 0, 2)

        top_layout.addLayout(plots_layout, 3)

        # Text Output
        text_group = QGroupBox("Live Gen")
        text_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffffff; border: 1px solid #333; margin-top: 5px; }")
        text_layout = QVBoxLayout(text_group)
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("font-family: 'Courier New'; font-size: 12px; background: #111; color: #eee; border: none;")
        text_layout.addWidget(self.text_output)
        top_layout.addWidget(text_group, 1)

        splitter.addWidget(top_widget)

        # --- Bottom: Layer Analysis ---
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0,0,0,0)

        # Layer Activity Bar Chart
        self.layer_act_plot = pg.PlotWidget(title="Mean Activity per Layer")
        self.layer_act_plot.setBackground('#0a0a0a')
        self.layer_act_plot.setLabel('bottom', 'Layer Index')
        self.layer_act_plot.showGrid(y=True, alpha=0.2)
        self.act_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#00ffcc')
        self.layer_act_plot.addItem(self.act_bars)
        bottom_layout.addWidget(self.layer_act_plot)

        # Layer Sparsity Bar Chart
        self.layer_sparse_plot = pg.PlotWidget(title="Sparsity per Layer")
        self.layer_sparse_plot.setBackground('#0a0a0a')
        self.layer_sparse_plot.setLabel('bottom', 'Layer Index')
        self.layer_sparse_plot.setYRange(0, 1)
        self.layer_sparse_plot.showGrid(y=True, alpha=0.2)
        self.sparse_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#ff00ff')
        self.layer_sparse_plot.addItem(self.sparse_bars)
        bottom_layout.addWidget(self.layer_sparse_plot)

        splitter.addWidget(bottom_widget)

        # Initial Sizes
        splitter.setSizes([300, 300])

        layout.addWidget(splitter)

    def configure_plot(self, plot_widget, label, color):
        plot_widget.setBackground('#0a0a0a')
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        plot_widget.setLabel('left', label)
        plot_widget.getPlotItem().hideAxis('bottom')
        curve = plot_widget.plot(pen=pg.mkPen(color, width=2))
        setattr(plot_widget, 'curve', curve)

    def update_metrics(self, speed_factor, loss, tokens_per_sec, sparsity):
        # Loss
        self.loss_data.append(loss)
        if len(self.loss_data) > 200: self.loss_data.pop(0)
        self.loss_plot.curve.setData(self.loss_data)
        self.loss_plot.setTitle(f"Loss: {loss:.4f}")

        # Speed
        self.speed_data.append(tokens_per_sec)
        if len(self.speed_data) > 200: self.speed_data.pop(0)
        self.speed_plot.curve.setData(self.speed_data)
        self.speed_plot.setTitle(f"Speed: {tokens_per_sec:,.0f} tok/s")

        # Sparsity
        self.sparsity_data.append(sparsity * 100)
        if len(self.sparsity_data) > 200: self.sparsity_data.pop(0)
        self.sparsity_plot.curve.setData(self.sparsity_data)
        self.sparsity_plot.setTitle(f"Sparsity: {sparsity*100:.1f}%")

    def update_layer_analysis(self, layer_activities, layer_importances):
        """
        Update bar charts for layer analysis.
        args: lists of arrays (per layer data)
        """
        num_layers = len(layer_activities)
        x = np.arange(num_layers)

        # Calculate mean activity per layer
        act_heights = [np.mean(act) for act in layer_activities]
        self.act_bars.setOpts(x=x, height=act_heights)

        # Calculate sparsity per layer (fraction of tiles < 0.1 imp)
        sparse_heights = [np.mean(imp < 0.1) for imp in layer_importances]
        self.sparse_bars.setOpts(x=x, height=sparse_heights)

    def update_text(self, text):
        if text:
            self.text_output.append(text)
            self.text_output.verticalScrollBar().setValue(
                self.text_output.verticalScrollBar().maximum()
            )
