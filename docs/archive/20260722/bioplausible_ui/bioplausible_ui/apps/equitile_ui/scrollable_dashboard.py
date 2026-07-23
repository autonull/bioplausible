import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ScrollableDashboard(QScrollArea):
    """
    Scrollable dashboard with toggleable visualization panels.
    Solves screen space issues by allowing panels to be hidden/shown.
    """

    def __init__(self, num_layers=4, parent=None):
        super().__init__(parent)
        self.num_layers = num_layers
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create container widget
        container = QWidget()
        self.setWidget(container)
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(8)

        # Panel visibility toggles
        self.panel_toggles = {}
        self.panel_widgets = {}

        self._init_panels()

    def _create_toggle_row(self, name, enabled=True):
        """Create a row with checkbox toggle and label."""
        row = QHBoxLayout()
        checkbox = QCheckBox(f"Show {name}")
        checkbox.setChecked(enabled)
        checkbox.setStyleSheet("font-weight: bold; color: #00ffcc;")
        row.addWidget(checkbox)
        row.addStretch()
        self.main_layout.addLayout(row)
        return checkbox, row

    def _init_panels(self):
        """Initialize all dashboard panels with toggles."""
        # Time-series plots group
        self._init_loss_panel()
        self._init_accuracy_panel()
        self._init_tile_loss_panel()
        self._init_throughput_panel()
        self._init_sparsity_panel()

        # Layer analysis
        self._init_layer_analysis_panel()

    def _init_loss_panel(self):
        """Loss & Perplexity panel."""
        checkbox, _ = self._create_toggle_row("Loss & Perplexity")

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: 1px solid #333; }")
        layout = QVBoxLayout(group)

        self.loss_plot = pg.PlotWidget(title="Loss & Perplexity")
        self.loss_plot.setBackground("#0a0a0a")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self.loss_plot.setYRange(0, 10)
        self.loss_data = []
        self.ppl_data = []
        # Ghost curve for previous run
        self.ghost_loss_curve = self.loss_plot.plot(
            pen=pg.mkPen("#ff00ff", width=1, style=Qt.PenStyle.DotLine)
        )
        self.ghost_loss_curve.setAlpha(0.3, False)

        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen("#ff00ff", width=2))
        self.ppl_curve = self.loss_plot.plot(pen=pg.mkPen("#ffff00", width=2))
        self.loss_plot.addLegend(offset=(10, 10))
        layout.addWidget(self.loss_plot)

        checkbox.toggled.connect(group.setVisible)
        self.panel_toggles["loss"] = checkbox
        self.panel_widgets["loss"] = group
        self.main_layout.addWidget(group)

    def _init_accuracy_panel(self):
        """Accuracy panel."""
        checkbox, _ = self._create_toggle_row("Accuracy")

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: 1px solid #333; }")
        layout = QVBoxLayout(group)

        self.acc_plot = pg.PlotWidget(title="Accuracy")
        self.acc_plot.setBackground("#0a0a0a")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.2)
        self.acc_plot.setXLink(self.loss_plot)
        self.acc_plot.setYRange(0, 100)
        self.train_acc_data = []
        self.test_acc_data = []

        # Ghost curve
        self.ghost_acc_curve = self.acc_plot.plot(
            pen=pg.mkPen("#00ff88", width=1, style=Qt.PenStyle.DotLine)
        )
        self.ghost_acc_curve.setAlpha(0.3, False)

        self.train_acc_curve = self.acc_plot.plot(pen=pg.mkPen("#00ff88", width=2))
        self.test_acc_curve = self.acc_plot.plot(pen=pg.mkPen("#00ccff", width=2))
        self.acc_plot.addLegend(offset=(10, 10))
        layout.addWidget(self.acc_plot)

        checkbox.toggled.connect(group.setVisible)
        self.panel_toggles["accuracy"] = checkbox
        self.panel_widgets["accuracy"] = group
        self.main_layout.addWidget(group)

    def _init_tile_loss_panel(self):
        """Per-tile loss panel."""
        checkbox, _ = self._create_toggle_row("Per-Tile Loss")

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: 1px solid #333; }")
        layout = QVBoxLayout(group)

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

        checkbox.toggled.connect(group.setVisible)
        self.panel_toggles["tile_loss"] = checkbox
        self.panel_widgets["tile_loss"] = group
        self.main_layout.addWidget(group)

    def _init_throughput_panel(self):
        """Throughput panel."""
        checkbox, _ = self._create_toggle_row("Throughput")

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: 1px solid #333; }")
        layout = QVBoxLayout(group)

        self.speed_plot = pg.PlotWidget(title="Throughput")
        self.speed_plot.setBackground("#0a0a0a")
        self.speed_plot.showGrid(x=True, y=True, alpha=0.2)
        self.speed_plot.setXLink(self.loss_plot)
        self.speed_data = []
        self.speed_curve = self.speed_plot.plot(pen=pg.mkPen("#00ff88", width=2))
        layout.addWidget(self.speed_plot)

        checkbox.toggled.connect(group.setVisible)
        self.panel_toggles["throughput"] = checkbox
        self.panel_widgets["throughput"] = group
        self.main_layout.addWidget(group)

    def _init_sparsity_panel(self):
        """Sparsity panel."""
        checkbox, _ = self._create_toggle_row("Sparsity")

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: 1px solid #333; }")
        layout = QVBoxLayout(group)

        self.sparsity_plot = pg.PlotWidget(title="Global Sparsity")
        self.sparsity_plot.setBackground("#0a0a0a")
        self.sparsity_plot.showGrid(x=True, y=True, alpha=0.2)
        self.sparsity_plot.setXLink(self.loss_plot)
        self.sparsity_plot.setYRange(0, 100)
        self.sparsity_data = []
        self.sparsity_curve = self.sparsity_plot.plot(pen=pg.mkPen("#00ccff", width=2))
        layout.addWidget(self.sparsity_plot)

        checkbox.toggled.connect(group.setVisible)
        self.panel_toggles["sparsity"] = checkbox
        self.panel_widgets["sparsity"] = group
        self.main_layout.addWidget(group)

    def _init_layer_analysis_panel(self):
        """Layer analysis panel."""
        checkbox, _ = self._create_toggle_row("Layer Analysis")

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: 1px solid #333; }")
        layout = QVBoxLayout(group)

        # Activity bar chart
        self.layer_act_plot = pg.PlotWidget(title="Layer Activity")
        self.layer_act_plot.setBackground("#0a0a0a")
        self.layer_act_plot.setLabel("bottom", "Layer Index")
        self.layer_act_plot.setLabel("left", "Mean Act")
        self.layer_act_plot.showGrid(y=True, alpha=0.2)
        self.act_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#00ffcc")
        self.layer_act_plot.addItem(self.act_bars)
        layout.addWidget(self.layer_act_plot)

        # Sparsity bar chart
        self.layer_sparse_plot = pg.PlotWidget(title="Layer Sparsity")
        self.layer_sparse_plot.setBackground("#0a0a0a")
        self.layer_sparse_plot.setLabel("bottom", "Layer Index")
        self.layer_sparse_plot.setLabel("left", "Sparsity")
        self.layer_sparse_plot.setYRange(0, 1)
        self.layer_sparse_plot.showGrid(y=True, alpha=0.2)
        self.sparse_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ff00ff")
        self.layer_sparse_plot.addItem(self.sparse_bars)
        layout.addWidget(self.layer_sparse_plot)

        # Gate states bar chart
        self.layer_gate_plot = pg.PlotWidget(title="Layer Gate States (% Open)")
        self.layer_gate_plot.setBackground("#0a0a0a")
        self.layer_gate_plot.setLabel("bottom", "Layer Index")
        self.layer_gate_plot.setLabel("left", "% Open")
        self.layer_gate_plot.setYRange(0, 1)
        self.layer_gate_plot.showGrid(y=True, alpha=0.2)
        self.gate_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ffff00")
        self.layer_gate_plot.addItem(self.gate_bars)
        layout.addWidget(self.layer_gate_plot)

        checkbox.toggled.connect(group.setVisible)
        self.panel_toggles["layer_analysis"] = checkbox
        self.panel_widgets["layer_analysis"] = group
        self.main_layout.addWidget(group)

    def update_loss(self, loss, perplexity):
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

    def update_accuracy(self, train_acc, test_acc):
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

        title = "Accuracy"
        if train_acc:
            title += f" | Train: {train_acc:.1f}%"
        if test_acc:
            title += f" | Test: {test_acc:.1f}%"
        self.acc_plot.setTitle(title)

    def update_tile_loss(self, tile_losses):
        if tile_losses and len(tile_losses) > 0:
            x = np.arange(len(tile_losses))
            self.tile_loss_bars.setOpts(x=x, height=tile_losses)

    def update_throughput(self, tps):
        self.speed_data.append(tps)
        if len(self.speed_data) > 300:
            self.speed_data.pop(0)
        self.speed_curve.setData(self.speed_data)
        self.speed_plot.setTitle(f"Throughput: {tps:,.0f} tok/s")

    def update_sparsity(self, sparsity_pct):
        self.sparsity_data.append(sparsity_pct)
        if len(self.sparsity_data) > 300:
            self.sparsity_data.pop(0)
        self.sparsity_curve.setData(self.sparsity_data)
        self.sparsity_plot.setTitle(f"Sparsity: {sparsity_pct:.1f}%")

    def update_layer_analysis(self, activities, importances, gate_states=None):
        num_layers = len(activities)
        x = np.arange(num_layers)

        act_heights = [np.mean(act) for act in activities]
        self.act_bars.setOpts(x=x, height=act_heights)

        sparse_heights = [np.mean(imp < 0.1) for imp in importances]
        self.sparse_bars.setOpts(x=x, height=sparse_heights)

        # Gate states (% of tiles with open gates per layer)
        if gate_states is not None and hasattr(self, "gate_bars"):
            gate_heights = [np.mean(gate) for gate in gate_states]
            self.gate_bars.setOpts(x=x, height=gate_heights)

    def save_ghost_curve(self):
        """Save current curves as ghosts for comparison."""
        if self.loss_data:
            self.ghost_loss_curve.setData(self.loss_data)
        if self.train_acc_data:
            self.ghost_acc_curve.setData(self.train_acc_data)
