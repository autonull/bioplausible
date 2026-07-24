import psutil
import torch
from bioplausible_ui.app.tabs.benchmarks_tab import BenchmarksTab
from bioplausible_ui.app.tabs.compare_tab import CompareTab
from bioplausible_ui.app.tabs.console_tab import ConsoleTab
from bioplausible_ui.app.tabs.deploy_tab import DeployTab
from bioplausible_ui.app.tabs.experiment_tab import ExperimentTab
from bioplausible_ui.app.tabs.home_tab import HomeTab
from bioplausible_ui.app.tabs.p2p_tab import P2PTab
from bioplausible_ui.app.tabs.results_tab import ResultsTab
from bioplausible_ui.app.tabs.settings_tab import SettingsTab
from bioplausible_ui.app.tabs.train_tab import TrainTab
from bioplausible_ui.core.themes import Theme
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QMainWindow, QTabWidget


class AppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bioplausible Trainer (biopl)")
        self.resize(1200, 800)
        self.setStyleSheet(Theme.get_stylesheet())

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.device_label = QLabel(
            f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
        )
        self.mem_label = QLabel("Mem: -")

        self.status_bar.addWidget(self.status_label, 1)  # Stretch
        self.status_bar.addPermanentWidget(self.device_label)
        self.status_bar.addPermanentWidget(self.mem_label)

        # Monitor Timer
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._update_status)
        self.monitor_timer.start(2000)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.home_tab = HomeTab()
        self.train_tab = TrainTab()
        self.compare_tab = CompareTab()
        self.experiment_tab = ExperimentTab()
        self.results_tab = ResultsTab()
        self.benchmarks_tab = BenchmarksTab()
        self.deploy_tab = DeployTab()
        self.p2p_tab = P2PTab()
        self.console_tab = ConsoleTab()
        self.settings_tab = SettingsTab()

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.compare_tab, "Compare")
        self.tabs.addTab(self.experiment_tab, "Experiment")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.benchmarks_tab, "Benchmarks")
        self.tabs.addTab(self.deploy_tab, "Deploy")
        self.tabs.addTab(self.p2p_tab, "Community")
        self.tabs.addTab(self.console_tab, "Console")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Connect Home Tab Signals
        self.home_tab.request_tab_change.connect(self._switch_to_tab)

        # Connect Experiment -> Train
        self.experiment_tab.transfer_config.connect(self._on_transfer_config)

    def _switch_to_tab(self, tab_name):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == tab_name:
                self.tabs.setCurrentIndex(i)
                break

    def _on_transfer_config(self, config):
        self.train_tab.set_config(config)
        self.tabs.setCurrentWidget(self.train_tab)

    def _update_status(self):
        # Update memory usage
        try:
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024  # MB
            self.mem_label.setText(f"Mem: {mem:.0f} MB")

            # If GPU
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                self.device_label.setText(f"Device: CUDA ({gpu_mem:.0f} MB)")

        except Exception:
            pass

    def closeEvent(self, event):
        """Clean up tabs on exit."""
        self.monitor_timer.stop()
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if hasattr(widget, "shutdown"):
                widget.shutdown()
        super().closeEvent(event)
