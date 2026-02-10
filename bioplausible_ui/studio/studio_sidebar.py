"""
Bioplausible Studio Sidebar

Navigation sidebar for the unified studio application.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QButtonGroup, QFrame, QLabel, QPushButton,
                             QVBoxLayout)


class StudioSidebar(QFrame):
    """Sidebar navigation menu."""

    mode_changed = pyqtSignal(
        str
    )  # Emits mode name (experiment, lab, leaderboard, radar)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(200)
        self.setStyleSheet("""
            StudioSidebar {
                background-color: #0f172a;
                border: none;
                border-right: 1px solid #1e293b;
            }
            QPushButton {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 6px;
                text-align: left;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1e293b;
                color: #e2e8f0;
            }
            QPushButton:checked {
                background-color: #334155;
                color: #fff;
                font-weight: 600;
                border-left: 3px solid #9333ea;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 20, 12, 20)

        # Logo / Title
        title = QLabel("🧬 Bioplausible")
        title.setStyleSheet(
            "color: #a855f7; font-size: 18px; font-weight: bold; margin-bottom: 20px; padding-left: 8px;"
        )
        layout.addWidget(title)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.btn_group.idClicked.connect(self._on_button_clicked)

        # Navigation Buttons
        self.add_nav_button("experiment", "🏠 Experiments")
        self.add_nav_button("lab", "🔬 Validation Lab")
        self.add_nav_button("leaderboard", "🏆 Leaderboard")
        self.add_nav_button("radar", "📊 Radar View")

        layout.addStretch()

        # Version info
        version = QLabel("v0.1.0")
        version.setStyleSheet("color: #475569; font-size: 11px; padding-left: 8px;")
        layout.addWidget(version)

    def add_nav_button(self, id_str, text):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setProperty("mode", id_str)
        if id_str == "experiment":
            btn.setChecked(True)

        self.layout().addWidget(btn)
        self.btn_group.addButton(btn)

    def _on_button_clicked(self, btn):
        mode = btn.property("mode")
        self.mode_changed.emit(mode)
