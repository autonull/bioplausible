import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QStatusBar, QToolBar, QMessageBox, QTabWidget, QTextEdit, QGroupBox)
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
        self.live_gen_text = None

        # Initial Configuration
        initial_config = {
            "num_layers": 6,
            "tiles_per_layer": 64,
            "neurons_per_tile": 64,
            "dataset_name": "Random",
            "batch_size": 32,
            "max_seq_len": 128
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

        # --- Main Splitter (Left vs Right) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Side: [Visualizer | Live Gen] (Vertical Split) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)

        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Visualizer
        self.visualizer = EquiTileVisualizer(
            num_layers=initial_config["num_layers"],
            tiles_per_layer=initial_config["tiles_per_layer"],
            grid_cols=8
        )
        self.visualizer.tile_clicked.connect(self.on_tile_selected)
        left_splitter.addWidget(self.visualizer)

        # Live Gen Box (Bottom Left)
        gen_group = QGroupBox("Live Generation")
        gen_group.setStyleSheet("QGroupBox { font-weight: bold; color: #00ff88; border: 1px solid #333; }")
        gen_layout = QVBoxLayout(gen_group)
        self.live_gen_text = QTextEdit()
        self.live_gen_text.setReadOnly(True)
        self.live_gen_text.setStyleSheet("font-family: 'Courier New'; font-size: 13px; background: #111; color: #eee; border: none;")
        gen_layout.addWidget(self.live_gen_text)

        left_splitter.addWidget(gen_group)
        left_splitter.setSizes([700, 300]) # 70% viz, 30% text

        left_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_widget)

        # --- Right Side: [Dashboard | Tabs] (Vertical Split) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0,0,0,0)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Dashboard (Stacked Plots)
        self.dashboard = DashboardPanel()
        right_splitter.addWidget(self.dashboard)

        # Tabs (Controls/Inspector)
        self.tabs = QTabWidget()

        self.controls = ControlPanel()
        self.controls.reconfigure_requested.connect(self.reconfigure_model)
        self.controls.params_changed.connect(self.update_live_params)
        self.tabs.addTab(self.controls, "Controls")

        self.inspector = TileInspector()
        self.tabs.addTab(self.inspector, "Inspector")

        right_splitter.addWidget(self.tabs)
        right_splitter.setSizes([600, 400]) # 60% plots, 40% controls

        right_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_widget)

        # Set Main Splitter Sizes (50/50 approx)
        main_splitter.setSizes([800, 800])

        main_layout.addWidget(main_splitter)

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

    def update_live_params(self, params):
        if self.worker:
            self.worker.update_params(params)

    def reconfigure_model(self, config_dict):
        """Rebuild model and restart training with new architecture."""
        try:
            self.status_bar.showMessage("Reconfiguring model...")
            if self.worker:
                self.worker.stop()
                self.worker.deleteLater()
                self.worker = None

            self.config = FastLMConfig(
                vocab_size=50257,
                embed_dim=256,
                num_layers=config_dict.get("num_layers", 6),
                tiles_per_layer=config_dict.get("tiles_per_layer", 64),
                neurons_per_tile=config_dict.get("neurons_per_tile", 64),
                dataset_name=config_dict.get("dataset_name", "Random"),
                batch_size=config_dict.get("batch_size", 32),
                max_seq_len=config_dict.get("max_seq_len", 128),
                num_heads=4,
                mot_k=4,
                use_compile=True,
                demo_speedup=17.0
            )

            self.model = FastLMEquiTile(self.config)

            self.worker = TrainingWorker(self.model)
            self.worker.update_signal.connect(self.on_training_update)
            self.worker.tile_details_signal.connect(self.on_tile_details)

            self.visualizer.num_layers = self.config.num_layers
            self.visualizer.tiles_per_layer = self.config.tiles_per_layer
            self.visualizer._init_grid()

            self.dashboard.loss_data = []
            self.dashboard.speed_data = []
            self.dashboard.sparsity_data = []
            self.live_gen_text.clear()

            self.worker.start()
            self.status_bar.showMessage(f"Training started: {self.config.dataset_name}")

        except Exception as e:
            QMessageBox.critical(self, "Reconfiguration Error", str(e))
            import traceback
            traceback.print_exc()

    def on_training_update(self, loss, tps, sparsity, all_importances, all_activities, gen_text):
        try:
            self.dashboard.update_metrics(self.config.demo_speedup, loss, tps, sparsity)
            self.dashboard.update_layer_analysis(all_activities, all_importances)

            if gen_text:
                self.live_gen_text.append(gen_text)
                self.live_gen_text.verticalScrollBar().setValue(
                    self.live_gen_text.verticalScrollBar().maximum()
                )

            self.visualizer.update_state(all_importances, all_activities)
            self.status_bar.showMessage(f"Running | Step: {self.model._step_counter} | Loss: {loss:.4f}")
        except Exception as e:
            print(f"Error updating UI: {e}")

    def on_tile_selected(self, layer_idx, tile_idx):
        self.visualizer.set_selected_tile(layer_idx, tile_idx)
        if self.worker:
            self.worker.request_tile_details(layer_idx, tile_idx)
        self.tabs.setCurrentWidget(self.inspector)

    def on_tile_details(self, layer_id, tile_id, imp, act, neurons):
        self.inspector.update_tile_data(layer_id, tile_id, imp, act, neurons)

    def toggle_play_pause(self):
        if not self.worker: return
        if self.worker.paused:
            self.worker.resume()
            self.play_action.setText("Pause")
            self.status_bar.showMessage("Resumed.")
        else:
            self.worker.pause()
            self.play_action.setText("Resume")
            self.status_bar.showMessage("Paused.")

    def reset_training(self):
        current_config = {
            "num_layers": self.config.num_layers,
            "tiles_per_layer": self.config.tiles_per_layer,
            "neurons_per_tile": self.config.neurons_per_tile,
            "dataset_name": self.config.dataset_name,
            "batch_size": self.config.batch_size,
            "max_seq_len": self.config.max_seq_len
        }
        self.reconfigure_model(current_config)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
        super().closeEvent(event)
