import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QStatusBar, QToolBar, QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QPalette, QColor

from bioplausible_ui.apps.equitile_ui.visualizer import EquiTileVisualizer
from bioplausible_ui.apps.equitile_ui.dashboard import DashboardPanel
from bioplausible_ui.apps.equitile_ui.controls import ControlPanel
from bioplausible_ui.apps.equitile_ui.inspector import TileInspector
from bioplausible_ui.apps.equitile_ui.worker import TrainingWorker
from bioplausible.models.equitile.live_demo_model import FastLMEquiTile, FastLMConfig

class EquiTileWindow(QMainWindow):
    """
    Main Application Window for EquiTile UI.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EquiTile • Live Bio-Plausible Training")
        self.resize(1600, 1000)
        self.apply_theme()

        try:
            self.config = FastLMConfig(
                vocab_size=50257,
                embed_dim=256,
                num_layers=6,
                tiles_per_layer=64,
                neurons_per_tile=64,
                num_heads=4,
                mot_k=4,
                use_compile=True,
                demo_speedup=17.0
            )

            self.model = FastLMEquiTile(self.config)
            self.worker = TrainingWorker(self.model)

            # Connect Signals
            self.worker.update_signal.connect(self.on_training_update)
            self.worker.tile_details_signal.connect(self.on_tile_details)

            self.init_ui()
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to start EquiTile Demo:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def apply_theme(self):
        """Set a dark, neon theme."""
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

        # Main Splitter: [Visualizer | Right Panel]
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Visualizer
        self.visualizer = EquiTileVisualizer(
            num_tiles=self.config.tiles_per_layer,
            grid_cols=8
        )
        self.visualizer.tile_clicked.connect(self.on_tile_selected)
        splitter.addWidget(self.visualizer)

        # Right: Tabs (Dashboard / Controls / Inspector)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0,0,0,0)

        # Dashboard is always visible at top? Or integrated?
        # Let's put Dashboard at top, then tabs for Controls/Inspector below.

        self.dashboard = DashboardPanel()
        right_layout.addWidget(self.dashboard, 1)

        self.tabs = QTabWidget()

        self.controls = ControlPanel()
        self.controls.params_changed.connect(self.worker.update_params)
        self.tabs.addTab(self.controls, "Controls")

        self.inspector = TileInspector()
        self.tabs.addTab(self.inspector, "Inspector")

        right_layout.addWidget(self.tabs, 2)

        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 600])

        main_layout.addWidget(splitter)

        # Toolbar
        self.toolbar = QToolBar("Controls")
        self.addToolBar(self.toolbar)

        self.play_action = QAction("Pause", self)
        self.play_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_action)

        self.reset_action = QAction("Reset", self)
        self.reset_action.triggered.connect(self.reset_training)
        self.toolbar.addAction(self.reset_action)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("EquiTile Demo Initialized. Training running...")

    def on_training_update(self, loss, tps, sparsity, importance, activity, gen_text):
        try:
            self.dashboard.update_metrics(self.config.demo_speedup, loss, tps, sparsity)
            if gen_text:
                self.dashboard.update_text(gen_text)
            self.visualizer.update_state(importance, activity)
            self.status_bar.showMessage(f"Running | Step: {self.model._step_counter} | Loss: {loss:.4f}")
        except Exception as e:
            print(f"Error updating UI: {e}")

    def on_tile_selected(self, index):
        """Handle tile selection from visualizer."""
        self.visualizer.set_selected_tile(index)
        self.worker.request_tile_details(index)
        self.tabs.setCurrentWidget(self.inspector) # Switch to inspector tab

    def on_tile_details(self, tile_id, imp, act, neurons):
        """Update inspector with detailed data."""
        self.inspector.update_tile_data(tile_id, imp, act, neurons)

    def toggle_play_pause(self):
        if self.worker.paused:
            self.worker.resume()
            self.play_action.setText("Pause")
            self.status_bar.showMessage("Resumed.")
        else:
            self.worker.pause()
            self.play_action.setText("Resume")
            self.status_bar.showMessage("Paused.")

    def reset_training(self):
        self.worker.pause()
        self.dashboard.text_output.clear()
        self.dashboard.loss_data = []
        self.dashboard.loss_curve.setData([])
        self.inspector.clear_inspector()
        self.visualizer.set_selected_tile(None)
        self.worker.resume()
        self.status_bar.showMessage("Training Reset.")

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
        super().closeEvent(event)
