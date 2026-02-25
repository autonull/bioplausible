from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QPushButton, QGroupBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import time
import random

class MockScientistWorker(QThread):
    """
    Simulates the AutoScientist agent running in a background thread.
    Emits logs and proposals.
    """
    log_signal = pyqtSignal(str, str) # message, style
    proposal_signal = pyqtSignal(dict, str) # config, explanation

    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False

    def run(self):
        self.running = True
        step = 0

        # Initial thought process
        self.log_signal.emit("AutoScientist initialized.", "bold green")
        time.sleep(1)
        self.log_signal.emit("Analyzing previous results database...", "blue")
        time.sleep(1.5)

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            step += 1

            # Simulate thinking
            self.log_signal.emit(f"Step {step}: Formulating hypothesis...", "white")
            time.sleep(2)

            # Propose Experiment
            model_type = random.choice(["EquiTile", "EqProp MLP", "Custom Stacked Model"])
            self.log_signal.emit(f"Hypothesis: {model_type} might improve stability on non-stationary data.", "cyan")

            config = {
                "name": model_type,
                "task_type": "lm",
                "dataset_name": "Tiny Shakespeare",
                "num_layers": random.randint(2, 6),
                "tiles_per_layer": random.choice([16, 32, 64]),
                "learning_rate": 0.001
            }

            self.proposal_signal.emit(config, f"Testing {model_type} with L={config['num_layers']} to verify stability hypothesis.")

            # Wait for user interaction (simulated by pause)
            self.paused = True

    def stop(self):
        self.running = False
        self.wait()

    def resume_search(self):
        self.paused = False


class AutoScientistPanel(QWidget):
    """
    Panel to interact with the Autonomous Scientist Agent.
    """
    experiment_approved = pyqtSignal(dict) # Emitted when user accepts a proposal

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = MockScientistWorker()
        self.worker.log_signal.connect(self.log_message)
        self.worker.proposal_signal.connect(self.show_proposal)

        self.current_proposal = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header / Status
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: #888;")
        layout.addWidget(self.status_label)

        # Log View (Thought Process)
        log_group = QGroupBox("Scientist's Thought Process")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #111; color: #ddd; font-family: Monospace;")
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, stretch=2)

        # Current Proposal Area
        prop_group = QGroupBox("Proposed Experiment")
        prop_group.setStyleSheet("QGroupBox { border: 1px solid #00aaaa; margin-top: 10px; }")
        prop_layout = QVBoxLayout(prop_group)

        self.hypothesis_label = QLabel("Waiting for hypothesis...")
        self.hypothesis_label.setWordWrap(True)
        self.hypothesis_label.setStyleSheet("font-style: italic; color: #00ffff;")

        self.config_preview = QLabel("-")
        self.config_preview.setStyleSheet("color: #aaa;")

        btn_layout = QHBoxLayout()
        self.approve_btn = QPushButton("✅ Approve & Queue")
        self.approve_btn.setStyleSheet("background-color: #00aa00; color: white;")
        self.approve_btn.clicked.connect(self.approve_proposal)
        self.approve_btn.setEnabled(False)

        self.reject_btn = QPushButton("❌ Reject")
        self.reject_btn.setStyleSheet("background-color: #aa0000; color: white;")
        self.reject_btn.clicked.connect(self.reject_proposal)
        self.reject_btn.setEnabled(False)

        btn_layout.addWidget(self.approve_btn)
        btn_layout.addWidget(self.reject_btn)

        prop_layout.addWidget(self.hypothesis_label)
        prop_layout.addWidget(self.config_preview)
        prop_layout.addLayout(btn_layout)

        layout.addWidget(prop_group, stretch=1)

        # Main Control
        self.start_btn = QPushButton("Start Auto-Discovery")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.toggle_scientist)
        layout.addWidget(self.start_btn)

    def toggle_scientist(self):
        if self.start_btn.isChecked():
            self.start_btn.setText("Stop Auto-Discovery")
            self.status_label.setText("Status: Running...")
            self.status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
            if not self.worker.isRunning():
                self.worker.start()
        else:
            self.start_btn.setText("Start Auto-Discovery")
            self.status_label.setText("Status: Stopped")
            self.status_label.setStyleSheet("font-weight: bold; color: #888;")
            self.worker.stop()

    def log_message(self, msg, style):
        # Apply HTML styling based on style arg (simplified)
        color = "#ddd"
        if "green" in style: color = "#00ff00"
        elif "red" in style: color = "#ff0000"
        elif "blue" in style: color = "#0088ff"
        elif "cyan" in style: color = "#00ffff"
        elif "yellow" in style: color = "#ffff00"

        html = f"<span style='color:{color}'>{msg}</span>"
        self.log_view.append(html)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def show_proposal(self, config, explanation):
        self.current_proposal = config
        self.hypothesis_label.setText(explanation)

        # Format config preview
        details = [f"{k}: {v}" for k, v in config.items()]
        self.config_preview.setText("\n".join(details))

        self.approve_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
        self.log_message("Waiting for human approval...", "yellow")

    def approve_proposal(self):
        if self.current_proposal:
            self.log_message("Proposal Approved. Adding to queue.", "green")
            self.experiment_approved.emit(self.current_proposal)
            self._reset_proposal_ui()
            self.worker.resume_search()

    def reject_proposal(self):
        self.log_message("Proposal Rejected.", "red")
        self._reset_proposal_ui()
        self.worker.resume_search()

    def _reset_proposal_ui(self):
        self.current_proposal = None
        self.hypothesis_label.setText("Generating next hypothesis...")
        self.config_preview.setText("-")
        self.approve_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
