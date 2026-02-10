from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QRadioButton,
                             QTextEdit, QVBoxLayout)

from bioplausible.p2p import Worker
from bioplausible.p2p.evolution import P2PEvolution
from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool


class P2PWorkerBridge(QObject):
    """Bridges P2P Worker/Evolution callbacks to Qt Signals."""

    status_changed = pyqtSignal(str, int, int)  # status, points, jobs
    log_received = pyqtSignal(str)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.on_status_change = self.emit_status
        self.worker.on_log = self.emit_log

    def emit_status(self, status, points, jobs):
        self.status_changed.emit(status, points, jobs)

    def emit_log(self, msg):
        self.log_received.emit(msg)


@ToolRegistry.register("p2p_grid", requires=["p2p"])
class P2PGridTool(BaseTool):
    ICON = "🕸️"

    def init_ui(self):
        super().init_ui()

        # Status Group
        status_group = QGroupBox("📡 Connection Status")
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel("DISCONNECTED")
        self.status_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #ff5555; border: 2px solid #ff5555; border-radius: 5px; padding: 10px;"
        )
        status_layout.addWidget(self.status_label)

        stats_layout = QHBoxLayout()
        # Points
        self.points_label = QLabel("0")
        self.points_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.points_label.setStyleSheet("color: #f39c12;")
        points_container = QVBoxLayout()
        points_container.addWidget(QLabel("Contribution Points (CP)"))
        points_container.addWidget(self.points_label)
        stats_layout.addLayout(points_container)

        # Jobs
        self.jobs_label = QLabel("0")
        self.jobs_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.jobs_label.setStyleSheet("color: #00d4ff;")
        jobs_container = QVBoxLayout()
        jobs_container.addWidget(QLabel("Jobs Completed"))
        jobs_container.addWidget(self.jobs_label)
        stats_layout.addLayout(jobs_container)

        status_layout.addLayout(stats_layout)
        self.layout.addWidget(status_group)

        # Connection Controls
        conn_group = QGroupBox("🔌 Network Settings")
        conn_layout = QVBoxLayout(conn_group)

        # Mode Selection
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)

        self.radio_coord = QRadioButton("Standard (Coordinator)")
        self.radio_coord.setChecked(True)
        self.mode_group.addButton(self.radio_coord)
        mode_layout.addWidget(self.radio_coord)

        self.radio_dht = QRadioButton("True P2P (DHT Mesh)")
        self.mode_group.addButton(self.radio_dht)
        mode_layout.addWidget(self.radio_dht)

        conn_layout.addLayout(mode_layout)

        self.radio_coord.toggled.connect(self._update_input_label)

        self.input_label = QLabel("Coordinator URL:")
        conn_layout.addWidget(self.input_label)

        # Bootstrap Nodes Dropdown (for DHT mode)
        self.bootstrap_combo = QComboBox()
        self.bootstrap_combo.setEditable(True)
        self.bootstrap_combo.addItems(
            [
                "bootstrap1.bioplausible.org",
                "bootstrap2.bioplausible.org",
                "127.0.0.1:8468 (Local Test)",
                "",  # Empty for self-bootstrap
            ]
        )
        self.bootstrap_combo.setPlaceholderText("Leave empty to start new network")
        self.bootstrap_combo.setVisible(False)
        conn_layout.addWidget(self.bootstrap_combo)

        self.url_input = QLineEdit("http://localhost:8000")  # Default for local testing
        self.url_input.setPlaceholderText("http://grid.bioplausible.org")
        conn_layout.addWidget(self.url_input)

        self.connect_btn = QPushButton("🚀 Join Network")
        self.connect_btn.setMinimumHeight(50)
        self.connect_btn.setStyleSheet(
            "font-weight: bold; font-size: 14px; background-color: #27ae60;"
        )
        # Task Selection
        self.task_combo = QComboBox()
        self.task_combo.addItems(
            ["shakespeare", "tiny_shakespeare", "mnist", "cifar10", "cartpole"]
        )
        self.task_combo.setToolTip("Target task to contribute to")
        conn_layout.addWidget(QLabel("Target Task:"))
        conn_layout.addWidget(self.task_combo)

        self.connect_btn.clicked.connect(self._toggle_connection)
        conn_layout.addWidget(self.connect_btn)

        self.layout.addWidget(conn_group)

        # Log
        log_group = QGroupBox("📜 Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "font-family: Consolas; font-size: 10px; background-color: #1a1a1e;"
        )
        log_layout.addWidget(self.log_output)
        self.layout.addWidget(log_group)

        self.worker = None
        self.bridge = None

    def _update_input_label(self):
        if self.radio_coord.isChecked():
            self.input_label.setText("Coordinator URL:")
            self.url_input.setVisible(True)
            self.bootstrap_combo.setVisible(False)
            self.url_input.setPlaceholderText("http://grid.bioplausible.org")
            self.url_input.setText("http://localhost:8000")
        else:
            self.input_label.setText("Bootstrap Node (IP):")
            self.url_input.setVisible(False)
            self.bootstrap_combo.setVisible(True)
            self.bootstrap_combo.setCurrentIndex(2)  # Default to local for safety

    def _toggle_connection(self):
        if self.worker and self.worker.running:
            # Stop
            self.worker.stop()
            self.connect_btn.setText("🚀 Join Network")
            self.connect_btn.setStyleSheet(
                "font-weight: bold; font-size: 14px; background-color: #27ae60;"
            )
            self.status_label.setText("DISCONNECTED")
            self.status_label.setStyleSheet(
                "color: #ff5555; border: 2px solid #ff5555; border-radius: 5px; padding: 10px;"
            )
            self._log("Worker stopped.")
            self.radio_coord.setEnabled(True)
            self.radio_dht.setEnabled(True)
        else:
            # Start
            self.radio_coord.setEnabled(False)
            self.radio_dht.setEnabled(False)

            if self.radio_coord.isChecked():
                # Client Mode
                target = self.url_input.text()
                if not target:
                    target = "http://localhost:8000"
                self.worker = Worker(target)
                self.worker.start_loop()
            else:
                # DHT Mode
                target = self.bootstrap_combo.currentText()
                ip = None
                port = 8468
                if target and "bootstrap" not in target:
                    parts = target.split(":")
                    ip = parts[0]
                    if len(parts) > 1:
                        port = int(parts[1])

                self.worker = P2PEvolution(
                    bootstrap_ip=ip,
                    bootstrap_port=port,
                    discovery_mode="quick",
                    task=self.task_combo.currentText(),
                )
                self.worker.start(auto_nice=True)

            # Setup Bridge
            self.bridge = P2PWorkerBridge(self.worker)
            self.bridge.status_changed.connect(self._on_status_changed)
            self.bridge.log_received.connect(self._log)

            self.connect_btn.setText("⏹ Stop Contributing")
            self.connect_btn.setStyleSheet(
                "font-weight: bold; font-size: 14px; background-color: #c0392b;"
            )
            self.status_label.setText("CONNECTING...")
            self.status_label.setStyleSheet(
                "color: #f39c12; border: 2px solid #f39c12; border-radius: 5px; padding: 10px;"
            )

    def _on_status_changed(self, status, points, jobs):
        self.status_label.setText(status.upper())
        if "Running" in status or "Mesh" in status or "Evaluating" in status:
            self.status_label.setStyleSheet(
                "color: #00ff88; border: 2px solid #00ff88; border-radius: 5px; padding: 10px;"
            )
        elif "Idle" in status or "Resting" in status:
            self.status_label.setStyleSheet(
                "color: #3498db; border: 2px solid #3498db; border-radius: 5px; padding: 10px;"
            )

        self.points_label.setText(str(points))
        self.jobs_label.setText(str(jobs))

    def _log(self, msg):
        self.log_output.append(msg)
