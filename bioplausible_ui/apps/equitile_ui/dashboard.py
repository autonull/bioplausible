from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QProgressBar, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

class DashboardPanel(QWidget):
    """
    Real-time dashboard for EquiTile metrics.
    Displays speed, sparsity, live text generation, and training curves.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # --- Plots Grid ---
        # 1 Row, 3 Columns: Loss, Speed, Sparsity
        plot_layout = QHBoxLayout()

        # Loss Plot
        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.configure_plot(self.loss_plot, "Loss", "#ff00ff") # Magenta
        self.loss_data = []
        plot_layout.addWidget(self.loss_plot)

        # Speed Plot
        self.speed_plot = pg.PlotWidget(title="Throughput (Tokens/sec)")
        self.configure_plot(self.speed_plot, "Tok/s", "#00ff88") # Green
        self.speed_data = []
        plot_layout.addWidget(self.speed_plot)

        # Sparsity Plot
        self.sparsity_plot = pg.PlotWidget(title="Global Sparsity (%)")
        self.configure_plot(self.sparsity_plot, "% Sparse", "#00ccff") # Cyan
        self.sparsity_plot.setYRange(0, 100)
        self.sparsity_data = []
        plot_layout.addWidget(self.sparsity_plot)

        layout.addLayout(plot_layout, 2) # Stretch factor 2

        # --- Live Text Generation ---
        text_group = QGroupBox("Live Generation")
        text_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; color: #ffffff; border: 1px solid #333; margin-top: 5px; }")
        text_layout = QVBoxLayout(text_group)
        text_layout.setContentsMargins(5, 15, 5, 5)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 13px;
            background: #111;
            color: #eeeeee;
            border: 1px solid #333;
        """)
        self.text_output.setMaximumHeight(120)
        text_layout.addWidget(self.text_output)

        layout.addWidget(text_group, 1) # Stretch factor 1

    def configure_plot(self, plot_widget, label, color):
        plot_widget.setBackground('#0a0a0a')
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        plot_widget.setLabel('left', label)
        plot_widget.getPlotItem().hideAxis('bottom')
        plot_widget.addLegend()
        # Create curve and store reference
        curve = plot_widget.plot(pen=pg.mkPen(color, width=2), name=label)
        setattr(plot_widget, 'curve', curve)

    def update_metrics(self, speed_factor, loss, tokens_per_sec, sparsity):
        """Update dashboard plots."""
        # Note: speed_factor is ignored now as requested (no fake gauge)

        # Loss
        self.loss_data.append(loss)
        if len(self.loss_data) > 300: self.loss_data.pop(0)
        self.loss_plot.curve.setData(self.loss_data)
        self.loss_plot.setTitle(f"Loss: {loss:.4f}")

        # Speed
        self.speed_data.append(tokens_per_sec)
        if len(self.speed_data) > 300: self.speed_data.pop(0)
        self.speed_plot.curve.setData(self.speed_data)
        self.speed_plot.setTitle(f"Speed: {tokens_per_sec:,.0f} tok/s")

        # Sparsity
        sparsity_pct = sparsity * 100
        self.sparsity_data.append(sparsity_pct)
        if len(self.sparsity_data) > 300: self.sparsity_data.pop(0)
        self.sparsity_plot.curve.setData(self.sparsity_data)
        self.sparsity_plot.setTitle(f"Sparsity: {sparsity_pct:.1f}%")

    def update_text(self, text):
        """Append generated text."""
        if text:
            self.text_output.append(text)
            self.text_output.verticalScrollBar().setValue(
                self.text_output.verticalScrollBar().maximum()
            )
