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

        self.worker = None
        self.visualizer = None
        self.dashboard = None
        self.controls = None
        self.inspector = None

        # Initial Architecture
        initial_config = {
            "num_layers": 6,
            "tiles_per_layer": 64,
            "neurons_per_tile": 64
        }

        self.init_ui(initial_config)
        self.reconfigure_model(initial_config)

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

    def init_ui(self, initial_config):
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
            num_layers=initial_config["num_layers"],
            tiles_per_layer=initial_config["tiles_per_layer"],
            grid_cols=8
        )
        self.visualizer.tile_clicked.connect(self.on_tile_selected)
        splitter.addWidget(self.visualizer)

        # Right: Dashboard + Tabs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0,0,0,0)

        self.dashboard = DashboardPanel()
        right_layout.addWidget(self.dashboard, 2)

        self.tabs = QTabWidget()

        self.controls = ControlPanel()
        self.controls.reconfigure_requested.connect(self.reconfigure_model) # Hook up reconfig
        self.tabs.addTab(self.controls, "Controls")

        self.inspector = TileInspector()
        self.tabs.addTab(self.inspector, "Inspector")

        right_layout.addWidget(self.tabs, 1)

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
        self.status_bar.showMessage("EquiTile Demo Initialized.")

    def reconfigure_model(self, config_dict):
        """Rebuild model and restart training with new architecture."""
        try:
            self.status_bar.showMessage("Reconfiguring model...")
            if self.worker:
                self.worker.stop()
                self.worker.deleteLater()

            # Create Config
            self.config = FastLMConfig(
                vocab_size=50257,
                embed_dim=256, # Keeping fixed for demo or add to controls
                num_layers=config_dict.get("num_layers", 6),
                tiles_per_layer=config_dict.get("tiles_per_layer", 64),
                neurons_per_tile=config_dict.get("neurons_per_tile", 64),
                num_heads=4,
                mot_k=4,
                use_compile=True,
                demo_speedup=17.0
            )

            # Create Model
            self.model = FastLMEquiTile(self.config)

            # Create Worker
            self.worker = TrainingWorker(self.model)
            self.worker.update_signal.connect(self.on_training_update)
            self.worker.tile_details_signal.connect(self.on_tile_details)

            # Re-connect controls
            self.controls.params_changed.connect(self.worker.update_params)

            # Reset Visualizer
            # Re-create visualizer or re-init?
            # Easier to re-create to handle layout logic, but need to replace in splitter.
            # Visualizer supports re-init if we expose it, but _init_grid depends on constructor args.
            # Let's verify Visualizer code... it uses self.num_layers etc.
            # We need to update those and call _init_grid.

            self.visualizer.num_layers = self.config.num_layers
            self.visualizer.tiles_per_layer = self.config.tiles_per_layer
            self.visualizer._init_grid()

            # Reset Dashboard
            self.dashboard.loss_data = []
            self.dashboard.speed_data = []
            self.dashboard.sparsity_data = []

            # Start
            self.worker.start()
            self.status_bar.showMessage("Training started with new configuration.")

        except Exception as e:
            QMessageBox.critical(self, "Reconfiguration Error", str(e))
            import traceback
            traceback.print_exc()

    def on_training_update(self, loss, tps, sparsity, all_importances, all_activities, gen_text):
        try:
            self.dashboard.update_metrics(self.config.demo_speedup, loss, tps, sparsity)
            self.dashboard.update_layer_analysis(all_activities, all_importances)

            if gen_text:
                self.dashboard.update_text(gen_text)

            self.visualizer.update_state(all_importances, all_activities)
            self.status_bar.showMessage(f"Running | Step: {self.model._step_counter} | Loss: {loss:.4f}")
        except Exception as e:
            print(f"Error updating UI: {e}")

    def on_tile_selected(self, layer_idx, tile_idx):
        self.visualizer.set_selected_tile(layer_idx, tile_idx)
        self.worker.request_tile_details(layer_idx, tile_idx)
        self.tabs.setCurrentWidget(self.inspector)

    def on_tile_details(self, layer_id, tile_id, imp, act, neurons):
        self.inspector.update_tile_data(layer_id, tile_id, imp, act, neurons)

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
        # Just trigger reconfigure with current settings
        current_config = {
            "num_layers": self.config.num_layers,
            "tiles_per_layer": self.config.tiles_per_layer,
            "neurons_per_tile": self.config.neurons_per_tile
        }
        self.reconfigure_model(current_config)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
        super().closeEvent(event)
