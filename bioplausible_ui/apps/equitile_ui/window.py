import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QStatusBar, QToolBar, QMessageBox, QTabWidget, QTextEdit, QGroupBox, QLabel, QApplication)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QPalette, QColor

from bioplausible_ui.apps.equitile_ui.visualizer import EquiTileVisualizer
from bioplausible_ui.apps.equitile_ui.scrollable_dashboard import ScrollableDashboard
from bioplausible_ui.apps.equitile_ui.controls import ControlPanel
from bioplausible_ui.apps.equitile_ui.inspector import TileInspector
from bioplausible_ui.apps.equitile_ui.diagnostics import (
    GradientHealthPanel, SparsityTimelinePanel, 
    ActivationDistributionPanel, AnomalyDetector, ModelHealthSummary
)
from bioplausible_ui.apps.equitile_ui.worker import TrainingWorker
from bioplausible.models.equitile.live_demo_model import FastLMEquiTile, FastLMConfig


class EquiTileWindow(QMainWindow):
    """Main Application Window for EquiTile UI."""
    
    def __init__(self, initial_config=None):
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
        self.model = None
        self.config = None
        self._training_active = False
        self._initializing = False
        
        # Diagnostic panels
        self.gradient_panel = None
        self.sparsity_timeline = None
        self.activation_dist = None
        self.anomaly_detector = None
        self.model_health = None
        
        # Panel toggle actions
        self.panel_actions = {}

        # Default Configuration - balanced for demo performance
        if initial_config is None:
            initial_config = {
                "num_layers": 4,
                "tiles_per_layer": 16,
                "neurons_per_tile": 32,
                "dataset_name": "Tiny Shakespeare",
                "batch_size": 16,
                "max_seq_len": 64
            }

        self.init_ui(initial_config)
        self.setup_model(initial_config)

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

        # Create menubar for panel toggles
        self._create_menu_bar()

        # --- Main Splitter (Left vs Right) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Side: [Visualizer | Live Gen] (Vertical Split) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

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
        self.live_gen_text.setPlaceholderText("Generated text will appear here during training...")
        gen_layout.addWidget(self.live_gen_text)

        left_splitter.addWidget(gen_group)
        left_splitter.setSizes([700, 300])

        left_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_widget)

        # --- Right Side: [Dashboard | Tabs] (Vertical Split) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Dashboard (Scrollable with toggleable panels)
        self.dashboard = ScrollableDashboard(num_layers=initial_config["num_layers"])
        right_splitter.addWidget(self.dashboard)

        # Tabs (Controls/Inspector/Diagnostics)
        self.tabs = QTabWidget()

        self.controls = ControlPanel()
        self.controls.reconfigure_requested.connect(self.reconfigure_model)
        self.controls.params_changed.connect(self.update_live_params)
        self.tabs.addTab(self.controls, "Controls")

        self.inspector = TileInspector()
        self.tabs.addTab(self.inspector, "Inspector")
        
        # Diagnostics tab
        diag_widget = QWidget()
        diag_layout = QVBoxLayout(diag_widget)
        
        self.model_health = ModelHealthSummary()
        diag_layout.addWidget(self.model_health)
        
        self.gradient_panel = GradientHealthPanel(num_layers=initial_config["num_layers"])
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

        # Play button
        self.play_action = QAction("▶ Play", self)
        self.play_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_action)

        self.reset_action = QAction("⟲ Reset", self)
        self.reset_action.triggered.connect(self.reset_training)
        self.toolbar.addAction(self.reset_action)

        # Progress indicator
        self.toolbar.addSeparator()
        self.status_label = QLabel("⏹ Stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: #888; padding: 5px;")
        self.toolbar.addWidget(self.status_label)

        self.toolbar.addSeparator()
        self.progress_label = QLabel("Step: 0")
        self.progress_label.setStyleSheet("font-weight: bold; color: #888; padding: 5px;")
        self.toolbar.addWidget(self.progress_label)

        # Status Bar - detailed messages
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("✓ Ready - Press Play to start training on Tiny Shakespeare")

    def _create_menu_bar(self):
        """Create menubar with panel toggle options."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

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

        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Dashboard panels submenu
        dashboard_menu = view_menu.addMenu("&Dashboard Panels")
        
        panel_configs = [
            ('loss', "Loss & Perplexity", True),
            ('accuracy', "Accuracy", True),
            ('tile_loss', "Per-Tile Loss", True),
            ('throughput', "Throughput", True),
            ('sparsity', "Sparsity", True),
            ('layer_analysis', "Layer Analysis", True),
        ]
        
        for key, label, checked in panel_configs:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(checked)
            action.triggered.connect(lambda v, k=key: self._toggle_dashboard_panel(k, v))
            dashboard_menu.addAction(action)
            self.panel_actions[key] = action
        
        # Diagnostics submenu
        diag_menu = view_menu.addMenu("&Diagnostics")
        
        diag_configs = [
            ('health', "Model Health Summary", True),
            ('gradients', "Gradient Health", True),
            ('sparsity_timeline', "Sparsity Timeline", True),
            ('activations', "Activation Distribution", True),
        ]
        
        for key, label, checked in diag_configs:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(checked)
            action.triggered.connect(lambda v, k=key: self._toggle_diagnostic_panel(k, v))
            diag_menu.addAction(action)
            self.panel_actions[f'diag_{key}'] = action
        
        # Visualizer options
        viz_menu = view_menu.addMenu("&Visualizer")
        
        viz_action = QAction("Show Tile Activity", self)
        viz_action.setCheckable(True)
        viz_action.setChecked(True)
        viz_menu.addAction(viz_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _toggle_dashboard_panel(self, key, visible):
        """Toggle dashboard panel visibility."""
        if hasattr(self, 'dashboard') and self.dashboard and key in self.dashboard.panel_widgets:
            self.dashboard.panel_widgets[key].setVisible(visible)
    
    def _toggle_diagnostic_panel(self, key, visible):
        """Toggle diagnostic panel visibility."""
        panel_map = {
            'health': 'model_health',
            'gradients': 'gradient_panel',
            'sparsity_timeline': 'sparsity_timeline',
            'activations': 'activation_dist'
        }
        if key in panel_map:
            panel_name = panel_map[key]
            if hasattr(self, panel_name):
                panel = getattr(self, panel_name)
                if panel:
                    panel.setVisible(visible)
    
    def _show_about(self):
        """Show about dialog."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "About EquiTile",
            "EquiTile Live Training Demo\n\n"
            "A biologically-plausible neural network\n"
            "with adaptive tile importance.\n\n"
            "Controls:\n"
            "• Play/Pause: Start/stop training\n"
            "• Reset: Restart with current settings\n"
            "• View menu: Toggle panels"
        )

    def setup_model(self, config_dict):
        """Initialize the model without starting training."""
        try:
            self._initializing = True
            self.status_bar.showMessage("⚙ Loading model architecture...")
            QApplication.processEvents()
            
            self.config = FastLMConfig(
                vocab_size=1000,
                embed_dim=128,
                num_layers=config_dict.get("num_layers", 4),
                tiles_per_layer=config_dict.get("tiles_per_layer", 16),
                neurons_per_tile=config_dict.get("neurons_per_tile", 32),
                dataset_name=config_dict.get("dataset_name", "Tiny Shakespeare"),
                batch_size=config_dict.get("batch_size", 16),
                max_seq_len=config_dict.get("max_seq_len", 64),
                num_heads=4,
                mot_k=4,
                use_compile=False
            )

            self.status_bar.showMessage("⚙ Building EquiTile model...")
            QApplication.processEvents()
            self.model = FastLMEquiTile(self.config)

            self.status_bar.showMessage(f"⚙ Loading dataset: {self.config.dataset_name}...")
            QApplication.processEvents()

            # Create worker (starts paused by default)
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

            # Initialize diagnostics
            self.anomaly_detector = AnomalyDetector()
            if self.gradient_panel:
                self.gradient_panel.num_layers = self.config.num_layers
                self.gradient_panel.update_gradients([0.0] * self.config.num_layers)
            if self.sparsity_timeline:
                self.sparsity_timeline.history = []
            if self.activation_dist:
                pass  # Will update with training data

            self._training_active = False
            self._initializing = False
            self.play_action.setText("▶ Play")
            self.play_action.setEnabled(True)
            self.status_label.setText("⏹ Stopped")
            self.status_label.setStyleSheet("font-weight: bold; color: #888; padding: 5px;")
            self.progress_label.setStyleSheet("font-weight: bold; color: #888; padding: 5px;")
            self.status_bar.showMessage(f"✓ Ready - Press Play to start training on {self.config.dataset_name}")

        except Exception as e:
            self._initializing = False
            QMessageBox.critical(self, "Model Setup Error", str(e))
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage(f"✗ Error: {e}")

    def update_live_params(self, params):
        if self.worker:
            self.worker.update_params(params)

    def reconfigure_model(self, config_dict):
        """Rebuild model with new architecture."""
        try:
            was_running = self._training_active
            if self.worker:
                self.worker.stop()
                self.worker.wait(1000)
                self.worker.deleteLater()
                self.worker = None

            self.setup_model(config_dict)
            
            if was_running:
                self.start_training()

        except Exception as e:
            QMessageBox.critical(self, "Reconfiguration Error", str(e))
            import traceback
            traceback.print_exc()

    def start_training(self):
        """Start the training worker."""
        if self.worker and not self._training_active and not self._initializing:
            self.status_bar.showMessage("▶ Starting training thread...")
            QApplication.processEvents()
            
            self.worker.start()
            
            self.status_bar.showMessage("▶ Resuming training loop...")
            QApplication.processEvents()
            self.worker.resume()
            
            self._training_active = True
            self.play_action.setText("⏸ Pause")
            self.status_label.setText("▶ Running")
            self.status_label.setStyleSheet("font-weight: bold; color: #00ff88; padding: 5px;")
            self.progress_label.setStyleSheet("font-weight: bold; color: #00ff88; padding: 5px;")
            self.status_bar.showMessage(f"▶ Training started on {self.config.dataset_name} - Step 1...")

    def stop_training(self):
        """Pause the training worker."""
        if self.worker and self._training_active:
            self.worker.pause()
            self._training_active = False
            self.play_action.setText("▶ Play")
            self.status_label.setText("⏸ Paused")
            self.status_label.setStyleSheet("font-weight: bold; color: #ffaa00; padding: 5px;")
            self.progress_label.setStyleSheet("font-weight: bold; color: #888; padding: 5px;")
            self.status_bar.showMessage("⏸ Training paused - Press Play to resume")

    def toggle_play_pause(self):
        if not self.worker:
            return
        if self._initializing:
            self.status_bar.showMessage("⏳ Please wait, model is loading...")
            return
        if self._training_active:
            self.stop_training()
        else:
            self.start_training()

    def on_training_update(self, loss, tps, sparsity, train_acc, test_acc, perplexity, all_importances, all_activities, gen_text, tile_losses, all_gate_states):
        try:
            # Update dashboard panels
            self.dashboard.update_loss(loss, perplexity)
            self.dashboard.update_accuracy(train_acc, test_acc)
            self.dashboard.update_tile_loss(tile_losses)
            self.dashboard.update_throughput(tps)
            self.dashboard.update_sparsity(sparsity * 100)
            self.dashboard.update_layer_analysis(all_activities, all_importances, all_gate_states)

            if gen_text:
                self.live_gen_text.append(gen_text)
                self.live_gen_text.verticalScrollBar().setValue(
                    self.live_gen_text.verticalScrollBar().maximum()
                )

            self.visualizer.update_state(all_importances, all_activities)

            # Update progress indicator and status bar
            step = self.model._step_counter
            self.progress_label.setText(f"Step: {step:,}")
            self.status_bar.showMessage(
                f"▶ Training | Step: {step:,} | "
                f"Loss: {loss:.4f} | Throughput: {tps:,.0f} tok/s | "
                f"Sparsity: {sparsity*100:.1f}%"
            )
            
            # Update diagnostic panels
            if self.anomaly_detector:
                grad_norms = []
                for layer in self.model.layers:
                    if hasattr(layer, 'tile_importance') and layer.tile_importance.grad is not None:
                        grad_norms.append(layer.tile_importance.grad.norm().item())
                    else:
                        grad_norms.append(0.0)
                
                grad_stats = {
                    'mean': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
                    'max': max(grad_norms) if grad_norms else 0,
                    'min': min(grad_norms) if grad_norms else 0,
                    'zero_pct': sum(1 for g in grad_norms if g == 0) / len(grad_norms) * 100 if grad_norms else 0
                }
                
                anomaly = self.anomaly_detector.check(step, loss, sparsity, grad_norms)
                if anomaly:
                    self.status_bar.showMessage(
                        f"⚠️ ANOMALY: {' | '.join(anomaly['alerts'])} - "
                        f"{' | '.join(anomaly['suggestions'])}"
                    )
                
                if self.gradient_panel:
                    self.gradient_panel.update_gradients(grad_norms)
                
                if self.sparsity_timeline:
                    self.sparsity_timeline.update_sparsity(sparsity * 100)
                
                if self.activation_dist:
                    self.activation_dist.update_activations(all_activities)
                
                if self.model_health:
                    self.anomaly_detector.loss_history.append(loss)
                    self.model_health.update(
                        grad_stats, train_acc, test_acc, sparsity * 100,
                        self.anomaly_detector.loss_history
                    )
                
                # Update visualizer with tile activity (dynamic!) and losses
                self.visualizer.update_state(all_importances, all_activities, tile_losses)
                    
        except Exception as e:
            print(f"Error updating UI: {e}")

    def on_tile_selected(self, layer_idx, tile_idx):
        self.visualizer.set_selected_tile(layer_idx, tile_idx)
        if self.worker:
            self.worker.request_tile_details(layer_idx, tile_idx)
        self.tabs.setCurrentWidget(self.inspector)

    def on_tile_details(self, layer_id, tile_id, imp, act, neurons):
        self.inspector.update_tile_data(layer_id, tile_id, imp, act, neurons)

    def save_model(self):
        """Save current model checkpoint."""
        if not self.model:
            return

        from PyQt6.QtWidgets import QFileDialog
        import os

        # Pause training if active
        was_active = self._training_active
        if was_active:
            self.stop_training()

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model Checkpoint",
            os.getcwd(),
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)"
        )

        if filepath:
            try:
                self.model.save_checkpoint(filepath)
                self.status_bar.showMessage(f"✓ Model saved to {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))
                self.status_bar.showMessage("✗ Save failed")

        # Resume if was active
        if was_active:
            self.start_training()

    def load_model(self):
        """Load model checkpoint."""
        from PyQt6.QtWidgets import QFileDialog
        import os

        # Pause training if active
        was_active = self._training_active
        if was_active:
            self.stop_training()

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model Checkpoint",
            os.getcwd(),
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)"
        )

        if filepath:
            try:
                # We need a model instance to load into
                if not self.model:
                    QMessageBox.warning(self, "No Model", "Please initialize a model first (by starting the app).")
                else:
                    self.model.load_checkpoint(filepath)
                    self.status_bar.showMessage(f"✓ Model loaded from {os.path.basename(filepath)}")
                    # Update iteration counter in UI
                    step = self.model._step_counter
                    self.progress_label.setText(f"Step: {step:,}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))
                self.status_bar.showMessage("✗ Load failed")

        # Resume if was active
        if was_active:
            self.start_training()

    def reset_training(self):
        current_config = {
            "num_layers": self.config.num_layers,
            "tiles_per_layer": self.config.tiles_per_layer,
            "neurons_per_tile": self.config.neurons_per_tile,
            "dataset_name": self.config.dataset_name,
            "batch_size": self.config.batch_size,
            "max_seq_len": self.config.max_seq_len
        }
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker.deleteLater()
            self.worker = None
        self._training_active = False
        self.setup_model(current_config)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(event)
