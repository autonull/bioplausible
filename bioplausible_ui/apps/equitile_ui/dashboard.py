from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QSplitter)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class DashboardPanel(QWidget):
    """
    Real-time dashboard for EquiTile metrics.
    Displays vertically stacked plots for temporal alignment.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # --- Stacked Time-Series Plots ---
        # Loss
        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.configure_plot(self.loss_plot, "Loss", "#ff00ff")
        self.loss_data = []
        layout.addWidget(self.loss_plot)

        # Speed
        self.speed_plot = pg.PlotWidget(title="Throughput")
        self.configure_plot(self.speed_plot, "Tok/s", "#00ff88")
        self.speed_data = []
        # Link X axis to Loss
        self.speed_plot.setXLink(self.loss_plot)
        layout.addWidget(self.speed_plot)

        # Sparsity
        self.sparsity_plot = pg.PlotWidget(title="Global Sparsity")
        self.configure_plot(self.sparsity_plot, "%", "#00ccff")
        self.sparsity_plot.setYRange(0, 100)
        self.sparsity_data = []
        # Link X axis to Loss
        self.sparsity_plot.setXLink(self.loss_plot)
        layout.addWidget(self.sparsity_plot)

        # --- Layer Analysis (Bar Charts) ---
        # Side-by-side at bottom
        layer_widget = QWidget()
        layer_layout = QHBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)

        # Activity
        self.layer_act_plot = pg.PlotWidget(title="Layer Activity")
        self.configure_bar_plot(self.layer_act_plot, "Mean Act")
        self.act_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#00ffcc')
        self.layer_act_plot.addItem(self.act_bars)
        layer_layout.addWidget(self.layer_act_plot)

        # Sparsity
        self.layer_sparse_plot = pg.PlotWidget(title="Layer Sparsity")
        self.configure_bar_plot(self.layer_sparse_plot, "Sparsity")
        self.layer_sparse_plot.setYRange(0, 1)
        self.sparse_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#ff00ff')
        self.layer_sparse_plot.addItem(self.sparse_bars)
        layer_layout.addWidget(self.layer_sparse_plot)

        layout.addWidget(layer_widget)

    def configure_plot(self, plot_widget, label, color):
        plot_widget.setBackground('#0a0a0a')
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        plot_widget.setLabel('left', label)
        # We keep bottom axis for all because they are stacked, but maybe hide for top ones?
        # User requested alignment, X-axis labels on all helps or just bottom.
        # Typically linking X makes them move together.
        curve = plot_widget.plot(pen=pg.mkPen(color, width=2))
        setattr(plot_widget, 'curve', curve)

    def configure_bar_plot(self, plot_widget, label):
        plot_widget.setBackground('#0a0a0a')
        plot_widget.setLabel('bottom', 'Layer Index')
        plot_widget.setLabel('left', label)
        plot_widget.showGrid(y=True, alpha=0.2)

    def update_metrics(self, speed_factor, loss, tokens_per_sec, sparsity):
        # Loss
        self.loss_data.append(loss)
        if len(self.loss_data) > 300: self.loss_data.pop(0)
        self.loss_plot.curve.setData(self.loss_data)
        self.loss_plot.setTitle(f"Loss: {loss:.4f}")

        # Speed
        self.speed_data.append(tokens_per_sec)
        if len(self.speed_data) > 300: self.speed_data.pop(0)
        self.speed_plot.curve.setData(self.speed_data)
        self.speed_plot.setTitle(f"Throughput: {tokens_per_sec:,.0f} tok/s")

        # Sparsity
        sparsity_pct = sparsity * 100
        self.sparsity_data.append(sparsity_pct)
        if len(self.sparsity_data) > 300: self.sparsity_data.pop(0)
        self.sparsity_plot.curve.setData(self.sparsity_data)
        self.sparsity_plot.setTitle(f"Sparsity: {sparsity_pct:.1f}%")

    def update_layer_analysis(self, layer_activities, layer_importances):
        num_layers = len(layer_activities)
        x = np.arange(num_layers)

        act_heights = [np.mean(act) for act in layer_activities]
        self.act_bars.setOpts(x=x, height=act_heights)

        sparse_heights = [np.mean(imp < 0.1) for imp in layer_importances]
        self.sparse_bars.setOpts(x=x, height=sparse_heights)
