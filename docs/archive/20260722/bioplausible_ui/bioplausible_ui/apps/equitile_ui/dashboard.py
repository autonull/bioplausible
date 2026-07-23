import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


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

        # --- Stacked Time-Series Plots (all X-axis linked) ---
        # Master plot for X-axis sync (Loss & Perplexity)
        self.loss_plot = pg.PlotWidget(title="Loss & Perplexity")
        self.configure_plot(self.loss_plot, "Loss", "#ff00ff")
        self.loss_plot.setYRange(0, 10)
        self.loss_data = []
        self.ppl_data = []
        self.loss_curve = self.loss_plot.plot(
            pen=pg.mkPen("#ff00ff", width=2), name="Loss"
        )
        self.ppl_curve = self.loss_plot.plot(
            pen=pg.mkPen("#ffff00", width=2), name="Perplexity"
        )
        self.loss_plot.addLegend(offset=(10, 10))
        layout.addWidget(self.loss_plot)

        # Accuracy - Train and Test (X-linked to loss_plot)
        self.acc_plot = pg.PlotWidget(title="Accuracy")
        self.configure_plot(self.acc_plot, "%", "#00ff88")
        self.acc_plot.setYRange(0, 100)
        self.acc_plot.setXLink(self.loss_plot)  # Sync X-axis
        self.train_acc_data = []
        self.test_acc_data = []
        self.train_acc_curve = self.acc_plot.plot(
            pen=pg.mkPen("#00ff88", width=2), name="Train"
        )
        self.test_acc_curve = self.acc_plot.plot(
            pen=pg.mkPen("#00ccff", width=2), name="Test"
        )
        self.acc_plot.addLegend(offset=(10, 10))
        layout.addWidget(self.acc_plot)

        # Per-Tile Loss (bar chart - not time series)
        self.tile_loss_plot = pg.PlotWidget(title="Per-Tile Loss Contribution")
        self.tile_loss_plot.setBackground("#0a0a0a")
        self.tile_loss_plot.setLabel("bottom", "Tile Index")
        self.tile_loss_plot.setLabel("left", "Loss Contrib.")
        self.tile_loss_plot.showGrid(y=True, alpha=0.2)
        self.tile_loss_bars = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush="#ff6600"
        )
        self.tile_loss_plot.addItem(self.tile_loss_bars)
        layout.addWidget(self.tile_loss_plot)

        # Speed (X-linked)
        self.speed_plot = pg.PlotWidget(title="Throughput")
        self.configure_plot(self.speed_plot, "Tok/s", "#00ff88")
        self.speed_plot.setXLink(self.loss_plot)  # Sync X-axis
        self.speed_data = []
        layout.addWidget(self.speed_plot)

        # Sparsity (X-linked)
        self.sparsity_plot = pg.PlotWidget(title="Global Sparsity")
        self.configure_plot(self.sparsity_plot, "%", "#00ccff")
        self.sparsity_plot.setYRange(0, 100)
        self.sparsity_plot.setXLink(self.loss_plot)  # Sync X-axis
        self.sparsity_data = []
        layout.addWidget(self.sparsity_plot)

        # --- Layer Analysis (Bar Charts) ---
        # Side-by-side at bottom
        layer_widget = QWidget()
        layer_layout = QHBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)

        # Activity
        self.layer_act_plot = pg.PlotWidget(title="Layer Activity")
        self.configure_bar_plot(self.layer_act_plot, "Mean Act")
        self.act_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#00ffcc")
        self.layer_act_plot.addItem(self.act_bars)
        layer_layout.addWidget(self.layer_act_plot)

        # Sparsity
        self.layer_sparse_plot = pg.PlotWidget(title="Layer Sparsity")
        self.configure_bar_plot(self.layer_sparse_plot, "Sparsity")
        self.layer_sparse_plot.setYRange(0, 1)
        self.sparse_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ff00ff")
        self.layer_sparse_plot.addItem(self.sparse_bars)
        layer_layout.addWidget(self.layer_sparse_plot)

        # Gate States (new)
        self.layer_gate_plot = pg.PlotWidget(title="Layer Gate States (% Open)")
        self.configure_bar_plot(self.layer_gate_plot, "% Open")
        self.layer_gate_plot.setYRange(0, 1)
        self.gate_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ffff00")
        self.layer_gate_plot.addItem(self.gate_bars)
        layer_layout.addWidget(self.layer_gate_plot)

        layout.addWidget(layer_widget)

    def configure_plot(self, plot_widget, label, color):
        plot_widget.setBackground("#0a0a0a")
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        plot_widget.setLabel("left", label)
        # We keep bottom axis for all because they are stacked, but maybe hide for top ones?
        # User requested alignment, X-axis labels on all helps or just bottom.
        # Typically linking X makes them move together.
        curve = plot_widget.plot(pen=pg.mkPen(color, width=2))
        setattr(plot_widget, "curve", curve)

    def configure_bar_plot(self, plot_widget, label):
        plot_widget.setBackground("#0a0a0a")
        plot_widget.setLabel("bottom", "Layer Index")
        plot_widget.setLabel("left", label)
        plot_widget.showGrid(y=True, alpha=0.2)

    def update_metrics(
        self,
        loss,
        tokens_per_sec,
        sparsity,
        train_acc=None,
        test_acc=None,
        perplexity=None,
        tile_losses=None,
    ):
        # Loss and Perplexity
        self.loss_data.append(loss)
        if len(self.loss_data) > 300:
            self.loss_data.pop(0)
        self.loss_curve.setData(self.loss_data)

        if perplexity is not None:
            self.ppl_data.append(perplexity)
            if len(self.ppl_data) > 300:
                self.ppl_data.pop(0)
            self.ppl_curve.setData(self.ppl_data)

        self.loss_plot.setTitle(
            f"Loss: {loss:.4f} | Perplexity: {perplexity:.1f}"
            if perplexity
            else f"Loss: {loss:.4f}"
        )

        # Accuracy
        if train_acc is not None:
            self.train_acc_data.append(train_acc)
            if len(self.train_acc_data) > 300:
                self.train_acc_data.pop(0)
            self.train_acc_curve.setData(self.train_acc_data)

        if test_acc is not None:
            self.test_acc_data.append(test_acc)
            if len(self.test_acc_data) > 300:
                self.test_acc_data.pop(0)
            self.test_acc_curve.setData(self.test_acc_data)

        acc_title = "Accuracy"
        if train_acc is not None:
            acc_title += f" | Train: {train_acc:.1f}%"
        if test_acc is not None:
            acc_title += f" | Test: {test_acc:.1f}%"
        self.acc_plot.setTitle(acc_title)

        # Per-Tile Loss
        if tile_losses is not None and len(tile_losses) > 0:
            x = np.arange(len(tile_losses))
            self.tile_loss_bars.setOpts(x=x, height=tile_losses)

        # Speed
        self.speed_data.append(tokens_per_sec)
        if len(self.speed_data) > 300:
            self.speed_data.pop(0)
        self.speed_plot.curve.setData(self.speed_data)
        self.speed_plot.setTitle(f"Throughput: {tokens_per_sec:,.0f} tok/s")

        # Sparsity
        sparsity_pct = sparsity * 100
        self.sparsity_data.append(sparsity_pct)
        if len(self.sparsity_data) > 300:
            self.sparsity_data.pop(0)
        self.sparsity_plot.curve.setData(self.sparsity_data)
        self.sparsity_plot.setTitle(f"Sparsity: {sparsity_pct:.1f}%")

    def update_layer_analysis(
        self, layer_activities, layer_importances, layer_gate_states=None
    ):
        num_layers = len(layer_activities)
        x = np.arange(num_layers)

        act_heights = [np.mean(act) for act in layer_activities]
        self.act_bars.setOpts(x=x, height=act_heights)

        sparse_heights = [np.mean(imp < 0.1) for imp in layer_importances]
        self.sparse_bars.setOpts(x=x, height=sparse_heights)

        # Gate states (% of tiles with open gates per layer)
        if layer_gate_states is not None:
            gate_heights = [np.mean(gate) for gate in layer_gate_states]
            self.gate_bars.setOpts(x=x, height=gate_heights)
        else:
            # Fallback: assume all gates open
            gate_heights = [1.0] * num_layers
            self.gate_bars.setOpts(x=x, height=gate_heights)
