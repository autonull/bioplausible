"""
Radar View Widget

Advanced visualization of the model search space.
Architecture:
- RadarLogic (business logic)
- RadarControlPanel (configuration)
- RadarPlot (pyqtgraph visualization)
- DetailOverlay (interactive popover)
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSizePolicy, QSplitter,
                             QToolTip, QVBoxLayout, QWidget)

from .radar_logic import ProjectionEngine, VisualMapper


class ProjectionWorker(QThread):
    """Background worker for computing projections."""

    finished = pyqtSignal(object)  # Emits (embedding, trial_ids, status_msg)

    def __init__(self, trials, method, selected_params):
        super().__init__()
        self.trials = trials
        self.method = method
        self.selected_params = selected_params

    def run(self):
        try:
            embedding, ids, err = ProjectionEngine.project(
                self.trials, self.method, self.selected_params
            )
            if err:
                self.finished.emit((None, [], err))
            else:
                self.finished.emit((embedding, ids, "Ready"))
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.finished.emit((None, [], f"Error: {str(e)}"))


class RadarControlPanel(QFrame):
    """Configuration panel for Radar View."""

    settings_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: none;
                border-right: 1px solid #334155;
            }
            QLabel { color: #94a3b8; font-weight: 600; margin-top: 8px; }
            QComboBox { 
                background-color: #0f172a; color: #e2e8f0; border: 1px solid #475569; padding: 4px;
            }
            QCheckBox { color: #cbd5e1; }
        """)
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        title = QLabel("📡 Radar Config")
        title.setStyleSheet(
            "color: #a855f7; font-size: 16px; font-weight: bold; margin-top: 0;"
        )
        layout.addWidget(title)

        # 1. Projection
        layout.addWidget(QLabel("Projection Algorithm"))
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(ProjectionEngine.get_available_methods())
        # Default to UMAP if present
        if "UMAP" in ProjectionEngine.get_available_methods():
            self.proj_combo.setCurrentText("UMAP")

        self.proj_combo.currentTextChanged.connect(self.emit_settings)
        layout.addWidget(self.proj_combo)

        # 2. Visualization
        layout.addWidget(QLabel("Color By"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["Model Family", "Accuracy", "Loss", "Param Count", "Time"]
        )
        self.color_combo.currentTextChanged.connect(self.emit_settings)
        layout.addWidget(self.color_combo)

        layout.addWidget(QLabel("Size By"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Uniform", "Accuracy", "Param Count"])
        self.size_combo.currentTextChanged.connect(self.emit_settings)
        layout.addWidget(self.size_combo)

        # 3. Parameters
        layout.addWidget(QLabel("Vectorize Params"))

        # Scrollable area for params
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setStyleSheet("background: transparent; border: none;")

        params_widget = QWidget()
        params_widget.setStyleSheet("background: transparent; border: none;")
        self.params_layout = QVBoxLayout(params_widget)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(4)

        self.param_checks = {}
        default_params = ["lr", "hidden_dim", "num_layers", "beta", "steps", "dropout"]

        # Select All Helper
        btn_layout = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.setStyleSheet(
            "background: #334155; color: white; border-radius: 4px; padding: 2px;"
        )
        sel_all.clicked.connect(self.select_all)
        sel_none = QPushButton("None")
        sel_none.setStyleSheet(
            "background: #334155; color: white; border-radius: 4px; padding: 2px;"
        )
        sel_none.clicked.connect(self.select_none)
        btn_layout.addWidget(sel_all)
        btn_layout.addWidget(sel_none)
        layout.addLayout(btn_layout)

        for p in default_params:
            cb = QCheckBox(p)
            cb.setChecked(True)
            cb.stateChanged.connect(self.emit_settings)
            self.param_checks[p] = cb
            self.params_layout.addWidget(cb)

        self.params_layout.addStretch()
        params_scroll.setWidget(params_widget)
        layout.addWidget(params_scroll)

    def select_all(self):
        for cb in self.param_checks.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.emit_settings()

    def select_none(self):
        for cb in self.param_checks.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.emit_settings()

    def emit_settings(self):
        selected_params = [p for p, cb in self.param_checks.items() if cb.isChecked()]
        settings = {
            "method": self.proj_combo.currentText(),
            "color_by": self.color_combo.currentText(),
            "size_by": self.size_combo.currentText(),
            "params": selected_params,
        }
        self.settings_changed.emit(settings)


class DetailOverlay(QFrame):
    """Floating overlay for trial details."""

    request_train = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 23, 42, 245);
                border: 1px solid #a855f7;
                border-radius: 12px;
                color: #e2e8f0;
            }
            QLabel { background: transparent; border: none;}
            QPushButton {
                background-color: #7c3aed;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #8b5cf6; }
        """)
        self.hide()

        # Use simple layout
        self.main_layout = QVBoxLayout(self)
        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.main_layout.addWidget(self.content_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("🚀 Train This")
        self.train_btn.clicked.connect(self._on_train)
        btn_layout.addWidget(self.train_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet("background-color: #334155; color: #cbd5e1;")
        self.close_btn.clicked.connect(self.hide)
        btn_layout.addWidget(self.close_btn)

        self.main_layout.addLayout(btn_layout)
        self.current_trial = None  # Store current trial data

    def show_trial(self, trial):
        self.current_trial = trial

        name = trial.get("model_name") or trial.get("model") or "Unknown Model"

        html = f"""
        <div style='font-family: sans-serif; font-size: 13px; line-height: 1.4;'>
            <div style='font-size: 15px; font-weight: bold; color: #a855f7; margin-bottom: 4px;'>
                {name}
            </div>
            <div style='color: #64748b; font-size: 11px; margin-bottom: 8px;'>ID: {trial.get('trial_id', 'N/A')}</div>
            
            <table style='width: 100%; border-spacing: 0;'>
                <tr>
                    <td style='color: #94a3b8; padding-right: 8px;'>Accuracy:</td>
                    <td style='color: #4ade80; font-weight: bold;'>{trial.get('accuracy', 0.0):.2%}</td>
                </tr>
                <tr>
                    <td style='color: #94a3b8; padding-right: 8px;'>Params:</td>
                    <td style='color: #e2e8f0;'>{trial.get('param_count', 0.0):.2f}M</td>
                </tr>
                <tr>
                    <td style='color: #94a3b8; padding-right: 8px;'>Time:</td>
                    <td style='color: #e2e8f0;'>{trial.get('iteration_time', 0.0):.3f}s</td>
                </tr>
            </table>
            
            <hr style='border: 1px solid #334155; margin: 8px 0;'>
            
            <div style='color: #cbd5e1; font-size: 12px;'>
        """

        config = trial.get("config", {})
        for k, v in sorted(config.items()):
            # Format value
            s_val = f"{v:.4f}" if isinstance(v, float) else str(v)
            html += f"<div><span style='color: #94a3b8;'>{k}:</span> {s_val}</div>"

        html += "</div></div>"

        self.content_label.setText(html)
        self.resize(250, 350)
        self.raise_()
        self.show()

    def _on_train(self):
        if self.current_trial:
            self.request_train.emit(self.current_trial)


class RadarPlot(pg.PlotWidget):
    """Interactive Plot Widget."""

    point_clicked = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setBackground("#0f172a")
        self.showGrid(x=True, y=True, alpha=0.1)
        self.getPlotItem().hideAxis("bottom")
        self.getPlotItem().hideAxis("left")
        self.setMouseEnabled(x=True, y=True)

        # Initialize scatter
        self.scatter = pg.ScatterPlotItem(
            pxMode=True, hoverable=True, hoverSymbol="o", hoverSize=15
        )
        self.scatter.sigClicked.connect(self._on_click)
        self.addItem(self.scatter)

        # Status text
        self.status = pg.TextItem("Ready", color="#64748b", anchor=(0, 1))
        self.status.setPos(0, 0)
        self.addItem(self.status)

    def _on_click(self, plot, points):
        if points:
            # Get topmost point
            pt = points[0]
            data = pt.data()
            if data:
                self.point_clicked.emit(data)

    def update_data(self, embedding, trials, settings):
        """Update scatter points."""
        self.scatter.clear()
        if embedding is None or len(embedding) == 0:
            return

        spots = []
        ranges = VisualMapper.get_ranges(trials)

        for i, (point, trial) in enumerate(zip(embedding, trials)):
            brush = VisualMapper.get_brush(trial, settings["color_by"], ranges)
            size = VisualMapper.get_size(trial, settings["size_by"], ranges)

            spots.append(
                {
                    "pos": point,
                    "size": size,
                    "brush": brush,
                    "pen": pg.mkPen(None),
                    "data": trial,
                }
            )

        self.scatter.addPoints(spots)

        # Recenter view with safety checks
        if len(embedding) > 0:
            min_x, min_y = np.min(embedding, axis=0)
            max_x, max_y = np.max(embedding, axis=0)

            # Prevent 0-range if single point or all same
            if max_x == min_x:
                min_x -= 1.0
                max_x += 1.0
            if max_y == min_y:
                min_y -= 1.0
                max_y += 1.0

            pad_x = (max_x - min_x) * 0.1
            pad_y = (max_y - min_y) * 0.1

            # Ensure not NaN
            if np.isnan(min_x) or np.isnan(max_x) or np.isnan(min_y) or np.isnan(max_y):
                return

            self.setXRange(min_x - pad_x, max_x + pad_x)
            self.setYRange(min_y - pad_y, max_y + pad_y)


class RadarView(QWidget):
    """
    Main Radar View Composite.
    """

    request_training = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle { background-color: #334155; width: 2px; }
        """)

        # 1. Controls
        self.controls = RadarControlPanel()
        self.controls.settings_changed.connect(self.request_update)
        splitter.addWidget(self.controls)

        # 2. Plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.plot = RadarPlot()
        self.plot.point_clicked.connect(self.show_details)
        plot_layout.addWidget(self.plot)

        # Attach overlay to Plot
        self.overlay = DetailOverlay(self.plot)
        self.overlay.request_train.connect(self.request_training)

        splitter.addWidget(plot_container)
        splitter.setStretchFactor(1, 4)

        layout.addWidget(splitter)

        # State
        self.trials = []
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)  # Debounce ms
        self.timer.timeout.connect(self.run_projection)

    def add_result(self, result):
        self.trials.append(result)
        self.timer.start()

    def request_update(self, settings=None):
        self.timer.start()

    def run_projection(self):
        # Cancel running worker if needed
        if hasattr(self, "worker") and self.worker.isRunning():
            self.timer.start()  # try again later
            return

        if not self.trials:
            return

        settings = self._get_settings()
        self.plot.status.setText(f"Computing {settings['method']}...")

        self.worker = ProjectionWorker(
            self.trials, settings["method"], settings["params"]
        )
        self.worker.finished.connect(lambda res: self.on_projection_done(res, settings))
        self.worker.start()

    def on_projection_done(self, result, settings):
        embedding, ids, msg = result
        if embedding is not None:
            self.plot.update_data(embedding, self.trials, settings)
            self.plot.status.setText(f"{settings['method']} | {len(embedding)} Trials")
        else:
            self.plot.status.setText(f"Error: {msg}")

    def show_details(self, trial):
        # Place overlay intelligently (top right of plot)
        rect = self.plot.rect()
        w, h = 260, 360
        x = rect.width() - w - 24
        y = 24
        self.overlay.resize(w, h)
        self.overlay.move(x, y)
        self.overlay.show_trial(trial)

    def _get_settings(self):
        return {
            "method": self.controls.proj_combo.currentText(),
            "color_by": self.controls.color_combo.currentText(),
            "size_by": self.controls.size_combo.currentText(),
            "params": [
                p for p, cb in self.controls.param_checks.items() if cb.isChecked()
            ],
        }
