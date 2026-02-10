class Theme:
    PRIMARY = "#4ecdc4"
    SECONDARY = "#ff6b6b"
    BACKGROUND = "#2d3436"
    SURFACE = "#353b48"
    TEXT = "#dfe6e9"
    ACCENT = "#00cec9"

    @staticmethod
    def get_stylesheet():
        return f"""
        QMainWindow, QWidget {{
            background-color: {Theme.BACKGROUND};
            color: {Theme.TEXT};
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            font-size: 14px;
        }}
        QTabWidget::pane {{
            border: 1px solid {Theme.SURFACE};
            background: {Theme.BACKGROUND};
        }}
        QTabBar::tab {{
            background: {Theme.SURFACE};
            color: {Theme.TEXT};
            padding: 10px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background: {Theme.PRIMARY};
            color: #ffffff;
            font-weight: bold;
        }}
        QPushButton {{
            background-color: {Theme.SURFACE};
            border: 2px solid {Theme.PRIMARY};
            border-radius: 6px;
            padding: 6px 12px;
            color: {Theme.PRIMARY};
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {Theme.PRIMARY};
            color: {Theme.BACKGROUND};
        }}
        QPushButton:disabled {{
            border-color: #555;
            color: #888;
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {Theme.SURFACE};
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px;
            color: {Theme.TEXT};
        }}
        QGroupBox {{
            border: 1px solid #555;
            border-radius: 6px;
            margin-top: 20px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            color: {Theme.ACCENT};
        }}
        QListWidget, QTableWidget {{
            background-color: {Theme.SURFACE};
            border: 1px solid #555;
            gridline-color: #444;
        }}
        QHeaderView::section {{
            background-color: {Theme.BACKGROUND};
            color: {Theme.TEXT};
            border: 1px solid #555;
            padding: 4px;
        }}
        QLabel {{
            color: {Theme.TEXT};
        }}
        """
