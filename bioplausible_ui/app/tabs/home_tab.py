from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QFrame, QGridLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)


class HomeTab(QWidget):
    """
    Landing page / Dashboard for the application.
    Provides quick access to main workflows.
    """

    # Signals to request tab changes
    request_tab_change = pyqtSignal(str)  # "train", "search", "results", "lab"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        title = QLabel("Bioplausible AI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        title.setStyleSheet("color: #4ecdc4; margin-bottom: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Next-Gen Equilibrium Propagation Research Platform")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 16))
        subtitle.setStyleSheet("color: #dfe6e9; margin-bottom: 30px;")
        layout.addWidget(subtitle)

        # Grid of Actions
        grid = QGridLayout()
        grid.setSpacing(20)

        # Helper to create nice buttons
        def create_card(title, desc, icon, callback):
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #353b48;
                    border: 1px solid #555;
                    border-radius: 12px;
                }
                QFrame:hover {
                    border-color: #4ecdc4;
                    background-color: #3d4554;
                }
            """)
            card_layout = QVBoxLayout(card)

            lbl_icon = QLabel(icon)
            lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_icon.setFont(QFont("Segoe UI", 48))

            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            lbl_title.setStyleSheet("color: #ffffff;")

            lbl_desc = QLabel(desc)
            lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_desc.setWordWrap(True)
            lbl_desc.setStyleSheet("color: #b2bec3;")

            btn = QPushButton("Open")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(callback)

            card_layout.addWidget(lbl_icon)
            card_layout.addWidget(lbl_title)
            card_layout.addWidget(lbl_desc)
            card_layout.addWidget(btn)

            return card

        # Train Card
        grid.addWidget(
            create_card(
                "New Experiment",
                "Configure and train new models (Vision, LM, RL) using EqProp or Backprop.",
                "🧠",
                lambda: self.request_tab_change.emit("Train"),
            ),
            0,
            0,
        )

        # Search Card
        grid.addWidget(
            create_card(
                "Hyperopt Search",
                "Discover optimal architectures using evolutionary search or grid search.",
                "🔍",
                lambda: self.request_tab_change.emit("Search"),
            ),
            0,
            1,
        )

        # Results Card
        grid.addWidget(
            create_card(
                "Results & Analysis",
                "Compare runs, export models, and analyze dynamics in the Lab.",
                "📊",
                lambda: self.request_tab_change.emit("Results"),
            ),
            1,
            0,
        )

        # Benchmarks Card
        grid.addWidget(
            create_card(
                "Benchmarks",
                "Run verification tracks to ensure framework stability and correctness.",
                "✅",
                lambda: self.request_tab_change.emit("Benchmarks"),
            ),
            1,
            1,
        )

        # Community Grid Card
        grid.addWidget(
            create_card(
                "Community Grid",
                "Participate in decentralized Neural Architecture Search (P2P).",
                "🕸️",
                lambda: self.request_tab_change.emit("Community"),
            ),
            2,
            0,
        )

        # Deploy Card
        grid.addWidget(
            create_card(
                "Model Serving",
                "Export models to ONNX/TorchScript or serve via REST API.",
                "🚀",
                lambda: self.request_tab_change.emit("Deploy"),
            ),
            2,
            1,
        )

        layout.addLayout(grid)
        layout.addStretch()
