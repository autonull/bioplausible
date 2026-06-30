import os
import sys

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QPalette
from PyQt6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QSplitter, QStatusBar,
                             QTabWidget, QTextEdit, QToolBar, QVBoxLayout,
                             QWidget)

from bioplausible_ui.apps.equitile_ui.config_dialog import ModelConfigDialog
from bioplausible_ui.apps.equitile_ui.controls import ControlPanel
from bioplausible_ui.apps.equitile_ui.diagnostics import (
    ActivationDistributionPanel, AnomalyDetector, GradientHealthPanel,
    ModelHealthSummary, SparsityTimelinePanel)
from bioplausible_ui.apps.equitile_ui.inspector import TileInspector
from bioplausible_ui.apps.equitile_ui.model_wrapper import LiveModelWrapper
from bioplausible_ui.apps.equitile_ui.queue_manager import (QueueManager,
                                                            QueuePanel)
from bioplausible_ui.apps.equitile_ui.scientist_panel import AutoScientistPanel
from bioplausible_ui.apps.equitile_ui.scrollable_dashboard import \
    ScrollableDashboard
from bioplausible_ui.apps.equitile_ui.visualizer import LayerGridVisualizer
from bioplausible_ui.apps.equitile_ui.worker import TrainingWorker


class EquiTileWindow(QMainWindow):
    """
    Main Application Window for Bio-Plausible Model Studio.
    Supports generic models via LiveModelWrapper and Experiment Queue.
    """

    def __init__(self, initial_config=None):
        super().__init__()
        self.setWindowTitle("Bio-Plausible Studio • Live Training")
        self.resize(1600, 1000)
        self.apply_theme()

        self.worker = None
        self.visualizer = None
        self.dashboard = None
        self.controls = None
        self.inspector = None
        self.live_gen_text = None
        self.wrapper = None
        self.config = initial_config
        self._training_active = False
        self._initializing = False

        # Queue Management
        self.queue_manager = QueueManager()
        self.queue_active = False
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self.check_queue_status)

        # Diagnostic panels
        self.gradient_panel = None
        self.sparsity_timeline = None
        self.activation_dist = None
        self.anomaly_detector = None
        self.model_health = None

        self.panel_actions = {}

        # Default Configuration if none provided
        if self.config is None:
            self.config = {
                "name": "EquiTile",
                "num_layers": 4,
                "tiles_per_layer": 16,
                "neurons_per_tile": 32,
                "dataset_name": "Tiny Shakespeare",
                "batch_size": 16,
                "max_seq_len": 64,
                "task_type": "lm",
            }

        self.init_ui(self.config)
        self.setup_model(self.config)

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

        self._create_menu_bar()

        # --- Main Splitter (Left vs Right) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Side: [Visualizer | Live Gen] ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Visualizer (Generic)
        self.visualizer = LayerGridVisualizer(layer_sizes=[])
        self.visualizer.tile_clicked.connect(self.on_tile_selected)
        left_splitter.addWidget(self.visualizer)

        # Live Gen Box
        gen_group = QGroupBox("Live Generation / Output")
        gen_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #00ff88; border: 1px solid #333; }"
        )
        gen_layout = QVBoxLayout(gen_group)
        self.live_gen_text = QTextEdit()
        self.live_gen_text.setReadOnly(True)
        self.live_gen_text.setStyleSheet(
            "font-family: 'Courier New'; font-size: 13px; background: #111; color: #eee; border: none;"
        )
        self.live_gen_text.setPlaceholderText(
            "Generated text or logs will appear here..."
        )
        gen_layout.addWidget(self.live_gen_text)

        left_splitter.addWidget(gen_group)
        left_splitter.setSizes([700, 300])

        left_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_widget)

        # --- Right Side: [Dashboard | Tabs] ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Dashboard
        self.dashboard = ScrollableDashboard(
            num_layers=initial_config.get("num_layers", 4)
        )
        right_splitter.addWidget(self.dashboard)

        # Tabs
        self.tabs = QTabWidget()

        self.controls = ControlPanel()
        self.controls.reconfigure_requested.connect(
            self.on_reconfigure_requested
        )  # Shows dialog
        self.controls.params_changed.connect(self.update_live_params)
        self.tabs.addTab(self.controls, "Controls")

        # Queue Panel
        self.queue_panel = QueuePanel(self.queue_manager)
        self.queue_panel.start_queue_signal.connect(self.run_queue)
        self.tabs.addTab(self.queue_panel, "Queue")

        # Scientist Panel
        self.scientist_panel = AutoScientistPanel()
        self.scientist_panel.experiment_approved.connect(self.on_scientist_proposal)
        self.tabs.addTab(self.scientist_panel, "Auto-Scientist")

        self.inspector = TileInspector()
        self.tabs.addTab(self.inspector, "Inspector")

        # Diagnostics
        diag_widget = QWidget()
        diag_layout = QVBoxLayout(diag_widget)

        self.model_health = ModelHealthSummary()
        diag_layout.addWidget(self.model_health)

        self.gradient_panel = GradientHealthPanel(
            num_layers=initial_config.get("num_layers", 4)
        )
        diag_layout.addWidget(self.gradient_panel)

        self.sparsity_timeline = SparsityTimelinePanel()
        diag_layout.addWidget(self.sparsity_timeline)

        self.activation_dist = ActivationDistributionPanel()
        diag_layout.addWidget(self.activation_dist)

        diag_layout.addStretch()
        self.tabs.addTab(diag_widget, "Diagnostics")

        right_splitter.addWidget(self.tabs)
        right_splitter.setSizes([600, 400])

        right_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_widget)

        main_splitter.setSizes([800, 800])
        main_layout.addWidget(main_splitter)

        # Toolbar
        self.toolbar = QToolBar("Controls")
        self.addToolBar(self.toolbar)

        self.play_action = QAction("▶ Play", self)
        self.play_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_action)

        self.reset_action = QAction("⟲ Reset", self)
        self.reset_action.triggered.connect(self.reset_training)
        self.toolbar.addAction(self.reset_action)

        self.toolbar.addSeparator()
        config_action = QAction("⚙ Config", self)
        config_action.triggered.connect(self.on_reconfigure_requested)
        self.toolbar.addAction(config_action)

        # Progress
        self.toolbar.addSeparator()
        self.status_label = QLabel("⏹ Stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: #888; padding: 5px;")
        self.toolbar.addWidget(self.status_label)

        self.toolbar.addSeparator()
        self.progress_label = QLabel("Step: 0")
        self.progress_label.setStyleSheet(
            "font-weight: bold; color: #888; padding: 5px;"
        )
        self.toolbar.addWidget(self.progress_label)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("✓ Ready")

    def _create_menu_bar(self):
        """Create menubar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Model...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.on_reconfigure_requested)
        file_menu.addAction(new_action)

        save_action = QAction("&Save Model...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_model)
        file_menu.addAction(save_action)

        load_action = QAction("&Load Model...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_model)
        file_menu.addAction(load_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menubar.addMenu("&View")

        dashboard_menu = view_menu.addMenu("&Dashboard Panels")
        panel_configs = [
            ("loss", "Loss & Perplexity", True),
            ("accuracy", "Accuracy", True),
            ("tile_loss", "Per-Tile Loss", True),
            ("throughput", "Throughput", True),
            ("sparsity", "Sparsity", True),
            ("layer_analysis", "Layer Analysis", True),
        ]
        for key, label, checked in panel_configs:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(checked)
            action.triggered.connect(
                lambda v, k=key: self._toggle_dashboard_panel(k, v)
            )
            dashboard_menu.addAction(action)
            self.panel_actions[key] = action

        diag_menu = view_menu.addMenu("&Diagnostics")
        diag_configs = [
            ("health", "Model Health Summary", True),
            ("gradients", "Gradient Health", True),
            ("sparsity_timeline", "Sparsity Timeline", True),
            ("activations", "Activation Distribution", True),
        ]
        for key, label, checked in diag_configs:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(checked)
            action.triggered.connect(
                lambda v, k=key: self._toggle_diagnostic_panel(k, v)
            )
            diag_menu.addAction(action)
            self.panel_actions[f"diag_{key}"] = action

    def _toggle_dashboard_panel(self, key, visible):
        if (
            hasattr(self, "dashboard")
            and self.dashboard
            and key in self.dashboard.panel_widgets
        ):
            self.dashboard.panel_widgets[key].setVisible(visible)

    def _toggle_diagnostic_panel(self, key, visible):
        panel_map = {
            "health": "model_health",
            "gradients": "gradient_panel",
            "sparsity_timeline": "sparsity_timeline",
            "activations": "activation_dist",
        }
        if key in panel_map:
            panel_name = panel_map[key]
            if hasattr(self, panel_name):
                panel = getattr(self, panel_name)
                if panel:
                    panel.setVisible(visible)

    def set_model(self, model_instance, config=None):
        """Set an external model instance."""
        try:
            if self.worker:
                self.worker.stop()
                self.worker.wait(1000)
                self.worker.deleteLater()
                self.worker = None

            self._initializing = True
            self.status_bar.showMessage("⚙ Wrapping external model...")
            QApplication.processEvents()

            if config is None:
                if hasattr(model_instance, "config"):
                    c = model_instance.config
                    if hasattr(c, "__dict__"):
                        c = c.__dict__
                    config = c
                else:
                    config = {"name": model_instance.__class__.__name__}

            self.config = config
            model_name = config.get("name", "Custom Model")

            self.wrapper = LiveModelWrapper(
                model_name, config, model_instance=model_instance
            )

            self.worker = TrainingWorker(self.wrapper)
            self.worker.update_signal.connect(self.on_training_update)
            self.worker.tile_details_signal.connect(self.on_tile_details)

            if hasattr(self.wrapper, "layer_sizes"):
                self.visualizer.layer_sizes = self.wrapper.layer_sizes
                self.visualizer._init_grid()

            self._finalize_setup()

        except Exception as e:
            self._initializing = False
            QMessageBox.critical(self, "External Model Error", str(e))
            import traceback

            traceback.print_exc()

    def setup_model(self, config_dict):
        """Initialize a new model from scratch."""
        try:
            self._initializing = True
            self.status_bar.showMessage(
                f"⚙ Loading model: {config_dict.get('name', 'Unknown')}..."
            )
            QApplication.processEvents()

            model_name = config_dict.get("name", "EquiTile")
            self.wrapper = LiveModelWrapper(model_name, config_dict)
            self.config = config_dict

            self.worker = TrainingWorker(self.wrapper)
            self.worker.update_signal.connect(self.on_training_update)
            self.worker.tile_details_signal.connect(self.on_tile_details)

            if hasattr(self.wrapper, "layer_sizes"):
                self.visualizer.layer_sizes = self.wrapper.layer_sizes
                self.visualizer._init_grid()

            self._finalize_setup()

        except Exception as e:
            self._initializing = False
            QMessageBox.critical(self, "Model Setup Error", str(e))
            import traceback

            traceback.print_exc()
            self.status_bar.showMessage(f"✗ Error: {e}")

    def _finalize_setup(self):
        self.dashboard.loss_data = []
        self.dashboard.speed_data = []
        self.dashboard.sparsity_data = []
        self.live_gen_text.clear()

        if self.anomaly_detector:
            self.anomaly_detector = AnomalyDetector()
        if self.gradient_panel:
            num_layers = (
                len(self.wrapper.layer_sizes) if self.wrapper.layer_sizes else 4
            )
            self.gradient_panel.num_layers = num_layers
            self.gradient_panel.update_gradients([0.0] * num_layers)

        self._training_active = False
        self._initializing = False
        self.play_action.setText("▶ Play")
        self.play_action.setEnabled(True)
        self.status_label.setText("⏹ Stopped")

        task = self.config.get("task_type", "lm")
        ds = self.config.get("dataset_name", "Data")
        model_name = self.config.get("name", "Model")
        self.status_bar.showMessage(f"✓ Ready - {model_name} on {task} ({ds})")

    def on_training_update(self, data):
        """Handle update from worker."""
        try:
            loss = data.get("loss", 0.0)
            tps = data.get("tps", 0.0)
            train_acc = data.get("train_acc", 0.0)
            test_acc = data.get("test_acc", 0.0)
            perplexity = data.get("perplexity", 0.0)
            importances = data.get("importances", [])
            activities = data.get("activities", [])
            gen_text = data.get("gen_text", "")
            step = data.get("step", 0)

            # Sparsity (generic)
            sparsity = 0.0
            if importances:
                total = sum(imp.size for imp in importances)
                active = sum((imp > 0.1).sum() for imp in importances)
                sparsity = 1.0 - (active / max(1, total))

            self.dashboard.update_loss(loss, perplexity)
            self.dashboard.update_accuracy(train_acc, test_acc)
            self.dashboard.update_throughput(tps)
            self.dashboard.update_sparsity(sparsity * 100)
            self.dashboard.update_layer_analysis(activities, importances, None)

            if gen_text:
                self.live_gen_text.append(gen_text)
                self.live_gen_text.verticalScrollBar().setValue(
                    self.live_gen_text.verticalScrollBar().maximum()
                )

            self.visualizer.update_state(importances, activities)

            self.progress_label.setText(f"Step: {step:,}")
            self.status_bar.showMessage(
                f"▶ Training | Step: {step:,} | "
                f"Loss: {loss:.4f} | Throughput: {tps:,.0f} items/s"
            )

            if self.activation_dist:
                self.activation_dist.update_activations(activities)

        except Exception as e:
            print(f"Error updating UI: {e}")
            import traceback

            traceback.print_exc()

    def on_tile_selected(self, layer_idx, tile_idx):
        self.visualizer.set_selected_tile(layer_idx, tile_idx)
        if self.worker:
            self.worker.request_tile_details(layer_idx, tile_idx)
        self.tabs.setCurrentWidget(self.inspector)

    def on_tile_details(self, layer_id, tile_id, imp, act, neurons):
        self.inspector.update_tile_data(layer_id, tile_id, imp, act, neurons)

    def on_scientist_proposal(self, config):
        """Handle experiment proposal from Auto-Scientist."""
        # Add to queue automatically
        config["queue_requested"] = True
        self.queue_panel.add_job(config)
        self.status_bar.showMessage(
            f"✓ Scientist proposal added to queue: {config.get('name')}"
        )
        # Switch to queue tab to show activity?
        # self.tabs.setCurrentWidget(self.queue_panel)

    def on_reconfigure_requested(self):
        """Open config dialog."""
        dialog = ModelConfigDialog(self, self.config)
        if dialog.exec():
            new_config = dialog.result_config
            if new_config:
                if new_config.get("queue_requested", False):
                    self.queue_panel.add_job(new_config)
                    self.tabs.setCurrentWidget(self.queue_panel)
                    self.status_bar.showMessage("✓ Job added to queue")
                else:
                    self.reconfigure_model(new_config)

    def reconfigure_model(self, config_dict):
        """Rebuild model."""
        was_running = self._training_active
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker.deleteLater()
            self.worker = None

        self.setup_model(config_dict)

        if was_running:
            self.start_training()

    def update_live_params(self, params):
        if self.wrapper:
            self.wrapper.update_params(params)

    def start_training(self):
        if self.worker and not self._training_active and not self._initializing:
            self.worker.start()
            self.worker.resume()
            self._training_active = True
            self.play_action.setText("⏸ Pause")
            self.status_label.setText("▶ Running")
            self.status_label.setStyleSheet(
                "font-weight: bold; color: #00ff88; padding: 5px;"
            )

    def stop_training(self):
        if self.worker and self._training_active:
            self.worker.pause()
            self._training_active = False
            self.play_action.setText("▶ Play")
            self.status_label.setText("⏸ Paused")
            self.status_label.setStyleSheet(
                "font-weight: bold; color: #ffaa00; padding: 5px;"
            )

    def toggle_play_pause(self):
        if not self.worker:
            return
        if self._initializing:
            return
        if self._training_active:
            self.stop_training()
        else:
            self.start_training()

    def save_model(self):
        if not self.wrapper:
            return
        from PyQt6.QtWidgets import QFileDialog

        was_active = self._training_active
        if was_active:
            self.stop_training()

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model", os.getcwd(), "PyTorch Checkpoints (*.pt)"
        )
        if filepath:
            try:
                self.wrapper.save_checkpoint(filepath)
                self.status_bar.showMessage(f"✓ Saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

        if was_active:
            self.start_training()

    def load_model(self):
        if not self.wrapper:
            return
        from PyQt6.QtWidgets import QFileDialog

        was_active = self._training_active
        if was_active:
            self.stop_training()

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", os.getcwd(), "PyTorch Checkpoints (*.pt)"
        )
        if filepath:
            try:
                self.wrapper.load_checkpoint(filepath)
                self.status_bar.showMessage(f"✓ Loaded from {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

        if was_active:
            self.start_training()

    def reset_training(self):
        # Save current run as ghost
        if self.dashboard:
            self.dashboard.save_ghost_curve()
        self.reconfigure_model(self.config)

    def closeEvent(self, event):
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(event)

    # Queue Handling
    def run_queue(self):
        """Start running the experiment queue."""
        self.queue_active = True
        self.queue_timer.start(1000)  # Check status every second
        self.run_next_in_queue()

    def run_next_in_queue(self):
        """Run the next job in the queue."""
        job = self.queue_manager.pop_job()
        if job:
            self.queue_panel.update_list()
            self.status_bar.showMessage(f"▶ Starting Queue Job: {job['name']}")

            # Setup and start
            self.reconfigure_model(job)
            self.start_training()

            # Auto-stop after N steps?
            # Ideally the job config should have max_steps.
            # Let's assume 1000 steps for demo queue if not set.
            self.target_steps = job.get("max_steps", 1000)
        else:
            self.queue_active = False
            self.queue_timer.stop()
            self.status_bar.showMessage("✓ Queue Finished")
            self.stop_training()

    def check_queue_status(self):
        """Check if current job is done."""
        if not self.queue_active:
            return

        if self.wrapper and self.wrapper.step_counter >= self.target_steps:
            # Job Done
            self.stop_training()

            # Auto-save
            name = self.config.get("name", "model").replace(" ", "_")
            path = f"queue_results/{name}_step{self.wrapper.step_counter}.pt"
            os.makedirs("queue_results", exist_ok=True)
            self.wrapper.save_checkpoint(path)
            print(f"Saved queue result to {path}")

            # Next
            self.run_next_in_queue()
