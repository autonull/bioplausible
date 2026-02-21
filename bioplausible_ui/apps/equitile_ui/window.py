import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSplitter, QStatusBar, QToolBar)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QFont, QPalette, QColor

from bioplausible_ui.apps.equitile_ui.visualizer import EquiTileVisualizer
from bioplausible_ui.apps.equitile_ui.dashboard import DashboardPanel
from bioplausible_ui.apps.equitile_ui.worker import TrainingWorker
from bioplausible.models.equitile.lm_demo import FastLMEquiTile, FastLMConfig

class EquiTileWindow(QMainWindow):
    """
    Main Application Window for EquiTile UI.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EquiTile • Live Bio-Plausible Training")
        self.resize(1600, 1000)

        # Apply Dark Theme (Cyberpunk Style)
        self.apply_theme()

        # Initialize Model (Fast Demo Version)
        self.config = FastLMConfig(
            vocab_size=50257, # GPT-2/3 like
            embed_dim=256,
            num_layers=6,
            tiles_per_layer=64, # Visualization friendly number
            neurons_per_tile=64,
            use_compile=True,
            demo_speedup=17.0
        )
        self.model = FastLMEquiTile(self.config)

        # Initialize Worker (Thread)
        self.worker = TrainingWorker(self.model)
        self.worker.update_signal.connect(self.on_training_update)

        # Initialize UI Components
        self.init_ui()

        # Start Worker
        self.worker.start()

    def apply_theme(self):
        """Set a dark, neon theme."""
        # Using QPalette for basic colors, relying on component styles for specifics
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(10, 10, 10))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(200, 200, 200))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(200, 200, 200))
        palette.setColor(QPalette.ColorRole.Button, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(200, 200, 200))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)

    def init_ui(self):
        """Setup layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter for adjustable resizing
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel: Visualization (2/3 width)
        self.visualizer = EquiTileVisualizer(
            num_tiles=self.config.tiles_per_layer,
            grid_cols=8
        )
        splitter.addWidget(self.visualizer)

        # Right Panel: Dashboard (1/3 width)
        self.dashboard = DashboardPanel()
        splitter.addWidget(self.dashboard)

        # Set initial sizes
        splitter.setSizes([1000, 500])

        main_layout.addWidget(splitter)

        # Toolbar
        self.toolbar = QToolBar("Controls")
        self.addToolBar(self.toolbar)

        # Play/Pause Action
        self.play_action = QAction("Pause", self)
        self.play_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_action)

        # Reset Action
        self.reset_action = QAction("Reset", self)
        self.reset_action.triggered.connect(self.reset_training)
        self.toolbar.addAction(self.reset_action)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("EquiTile Demo Initialized. Training running...")

    def on_training_update(self, loss, tps, sparsity, importance, activity, gen_text):
        """Handle updates from the worker thread."""
        # Update Dashboard
        self.dashboard.update_metrics(self.config.demo_speedup, loss, tps, sparsity)

        if gen_text:
            self.dashboard.update_text(gen_text)

        # Update Visualization
        self.visualizer.update_state(importance, activity)

        # Update Status Bar
        self.status_bar.showMessage(f"Running | Step: {self.model._step_counter} | Loss: {loss:.4f}")

    def toggle_play_pause(self):
        """Toggle training state."""
        if self.worker.paused:
            self.worker.resume()
            self.play_action.setText("Pause")
            self.status_bar.showMessage("Resumed.")
        else:
            self.worker.pause()
            self.play_action.setText("Resume")
            self.status_bar.showMessage("Paused.")

    def reset_training(self):
        """Reset the model and training state."""
        self.worker.pause()
        # TODO: Implement proper reset logic in model/worker
        # For now, just clear visualization and text
        self.dashboard.text_output.clear()
        self.dashboard.loss_data = []
        self.dashboard.loss_curve.setData([])
        self.worker.resume()
        self.status_bar.showMessage("Training Reset.")

    def closeEvent(self, event):
        """Cleanup on exit."""
        self.worker.stop()
        super().closeEvent(event)
