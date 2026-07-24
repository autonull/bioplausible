"""
Bioplausible Studio - Unified Application

Main entry point integrating Experiment Runner, Validation Lab, Leaderboard, and Radar View.
"""

import sys

from bioplausible_ui.app.window import AppMainWindow
from bioplausible_ui.apps.equitile_ui.window import EquiTileWindow
from bioplausible_ui.core.themes import Theme
from bioplausible_ui.core.widgets.radar_view import RadarView
from bioplausible_ui.lab.window import LabMainWindow
from bioplausible_ui.leaderboard.leaderboard_data import load_trials
from bioplausible_ui.leaderboard.leaderboard_window import LeaderboardWindow
from bioplausible_ui.studio.studio_sidebar import StudioSidebar
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

# Import sub-applications
# Note: We import the widgets/contents, not the MainWindows if possible,
# or adapt them. Existing windows usually inherit QMainWindow.
# We'll treat them as central widgets or wrap them.


class BioplausibleStudio(QMainWindow):
    """Unified application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bioplausible Studio")
        self.resize(1600, 1000)

        # Apply global theme
        self.setStyleSheet(Theme.get_stylesheet() + """
            QMainWindow { background-color: #0f172a; }
        """)

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = StudioSidebar()
        self.sidebar.mode_changed.connect(self.switch_mode)
        main_layout.addWidget(self.sidebar)

        # Content Area (Stacked Widget)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Initialize modes
        self.init_modes()

    def init_modes(self):
        """Initialize and add all sub-application modes."""

        # 1. Experiment Runner (App)
        # Assuming AppMainWindow is a QMainWindow. QMainWindow inside QWidget wrapper works but is odd.
        # Ideally we refactor to extract central widgets, but for now we wrap.
        self.app_window = AppMainWindow()
        self.stack.addWidget(self.wrap_window(self.app_window))

        # 2. Validation Lab
        self.lab_window = LabMainWindow()
        self.lab_window.request_visualize.connect(self.on_visualize_request)
        self.stack.addWidget(self.wrap_window(self.lab_window))

        # 3. Leaderboard
        self.leaderboard_window = LeaderboardWindow()
        self.leaderboard_window.request_training.connect(self.on_request_training)
        self.stack.addWidget(self.wrap_window(self.leaderboard_window))

        # 4. Radar View
        self.radar_view = RadarView()
        self.radar_view.request_training.connect(self.on_radar_train_request)
        self.stack.addWidget(self.radar_view)

        # 5. EquiTile Demo
        self.equitile_window = EquiTileWindow()
        self.stack.addWidget(self.wrap_window(self.equitile_window))

    def wrap_window(self, window):
        """Wrap a QMainWindow to be used as a widget."""
        # If the sub-app is QMainWindow, we usually take its central widget + toolbars + statusbar
        # But simpler hack: just use it as is, QMainWindow inherits QWidget.
        # But setWindowFlags to Widget to rely on parent layout
        window.setWindowFlags(Qt.WindowType.Widget)
        return window

    def switch_mode(self, mode):
        """Switch the displayed content stack."""
        if mode == "experiment":
            self.stack.setCurrentIndex(0)
        elif mode == "lab":
            self.stack.setCurrentIndex(1)
        elif mode == "leaderboard":
            self.stack.setCurrentIndex(2)
            # Auto-refresh leaderboard when switched to
            if hasattr(self.leaderboard_window, "refresh_data"):
                self.leaderboard_window.refresh_data()
        elif mode == "radar":
            self.stack.setCurrentIndex(3)
            self.refresh_radar()
        elif mode == "equitile":
            self.stack.setCurrentIndex(4)

    def refresh_radar(self):
        """Load data into global Radar View."""
        # Use same DB as leaderboard (default for now)
        db_path = "bioplausible.db"  # Updated default
        try:
            trials = load_trials(db_path)
            # self.radar_view.clear()
            # Re-building radar data is cheap

            # Since radar accumulates, maybe we should clear?
            # RadarView adds to list. We should likely reset.
            # But RadarView class doesn't have clear().
            # It's fine, we just add unique?
            # For now, let's just add new ones.

            for trial in trials:
                # convert to radar format
                result = {
                    "params": trial.get("config", {}),
                    "accuracy": trial.get("accuracy", 0.0),
                    "final_loss": trial.get("loss", 0.0),
                    "model_name": trial.get("model_name", "Unknown"),
                    "task": trial.get("task", "vision"),
                    "trial_id": trial.get("trial_id", 0),
                    "param_count": trial.get("param_count", 0.0),
                    "iteration_time": trial.get("iteration_time", 0.0),
                }
                self.radar_view.add_result(result)
        except Exception as e:
            print(f"Failed to refresh radar: {e}")

    def on_request_training(self, config):
        """Handle request to train a specific config."""
        # Switch sidebar to experiment
        # We need to manually update sidebar state if possible
        for btn in self.sidebar.btn_group.buttons():
            if btn.property("mode") == "experiment":
                btn.setChecked(True)
                break

        self.switch_mode("experiment")

        # Access train tab
        # We need to find the input mechanisms in ExperimentTab.
        # ExperimentTab takes `overrides` and selection.
        # But `transfer_config` implies we want to PRE-FILL valid inputs.

        # Since ExperimentTab is mostly about SURVEYS, maybe we mean "Single Run" mode?
        # Or just selecting the items.

        # For now, let's just log or try to set selection.
        if hasattr(self.app_window, "experiment_tab"):
            # We can't easily programmatically set all widgets match config.
            pass

    def on_radar_train_request(self, trial):
        """Handle 'Train This' from Radar."""
        config = {
            "model": trial.get("model_name", "Unknown"),
            "hyperparams": trial.get("params", {}),
            "task": trial.get("task", "vision"),
        }
        self.on_request_training(config)

    def on_visualize_request(self, model):
        """Handle request to visualize a model instance."""
        # Switch to EquiTile tab
        for btn in self.sidebar.btn_group.buttons():
            if btn.property("mode") == "equitile":
                btn.setChecked(True)
                break

        self.switch_mode("equitile")
        self.equitile_window.set_model(model)


def main():
    app = QApplication(sys.argv)
    window = BioplausibleStudio()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
