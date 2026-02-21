from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QProgressBar, QGroupBox)
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # --- Speed Gauge ("17x FASTER") ---
        self.speed_label = QLabel("1.0x FASTER")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speed_label.setStyleSheet("""
            font-size: 48px; font-weight: bold; color: #00ff88;
            background: rgba(0,255,100,0.1); border: 3px solid #00ff88; border-radius: 10px;
            padding: 10px;
        """)
        layout.addWidget(self.speed_label)

        # --- Metrics Grid ---
        metrics_group = QGroupBox("Training Metrics")
        metrics_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; color: #aaffff; }")
        metrics_layout = QVBoxLayout(metrics_group)

        self.loss_label = QLabel("Loss: --")
        self.loss_label.setStyleSheet("font-size: 18px; color: #ff8800;") # Orange
        metrics_layout.addWidget(self.loss_label)

        self.tps_label = QLabel("Tokens/sec: --")
        self.tps_label.setStyleSheet("font-size: 18px; color: #ffff00;") # Yellow
        metrics_layout.addWidget(self.tps_label)

        self.sparsity_label = QLabel("Active Tiles: --%")
        self.sparsity_label.setStyleSheet("font-size: 18px; color: #00ccff;") # Cyan
        metrics_layout.addWidget(self.sparsity_label)

        layout.addWidget(metrics_group)

        # --- Live Text Generation ---
        text_group = QGroupBox("Live Generation")
        text_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; color: #ff00ff; }")
        text_layout = QVBoxLayout(text_group)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 14px;
            background: #111;
            color: #00ff88;
            border: 1px solid #333;
        """)
        self.text_output.setMaximumHeight(150)
        text_layout.addWidget(self.text_output)

        layout.addWidget(text_group)

        # --- Training Curve (Loss vs Time) ---
        self.plot_widget = pg.PlotWidget(title="Training Loss")
        self.plot_widget.setBackground('#0a0a0a')
        self.plot_widget.setLabel('left', 'Loss')
        self.plot_widget.setLabel('bottom', 'Step')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.loss_curve = self.plot_widget.plot(pen=pg.mkPen('#ff00ff', width=3)) # Magenta curve

        self.loss_data = [] # Store data points

        layout.addWidget(self.plot_widget)

    def update_metrics(self, speed_factor, loss, tokens_per_sec, sparsity):
        """Update dashboard labels and plots."""
        # Speed Gauge Animation (simple color pulse if > 15x)
        if speed_factor > 15.0:
            self.speed_label.setStyleSheet("""
                font-size: 52px; font-weight: bold; color: #00ffff;
                background: rgba(0,255,255,0.2); border: 4px solid #00ffff; border-radius: 12px;
                padding: 10px;
            """)
        else:
            self.speed_label.setStyleSheet("""
                font-size: 48px; font-weight: bold; color: #00ff88;
                background: rgba(0,255,100,0.1); border: 3px solid #00ff88; border-radius: 10px;
                padding: 10px;
            """)

        self.speed_label.setText(f"{speed_factor:.1f}x FASTER")

        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.tps_label.setText(f"Tokens/sec: {tokens_per_sec:,.0f}")
        self.sparsity_label.setText(f"Active Tiles: {sparsity * 100:.1f}%")

        # Update Plot
        self.loss_data.append(loss)
        if len(self.loss_data) > 200: # Limit visible history
            self.loss_data.pop(0)

        self.loss_curve.setData(self.loss_data)

    def update_text(self, text):
        """Append generated text."""
        if text:
            self.text_output.append(text)
            # Auto-scroll to bottom
            self.text_output.verticalScrollBar().setValue(
                self.text_output.verticalScrollBar().maximum()
            )
