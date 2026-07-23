import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from bioplausible_ui.core.themes import Theme


class BasePlotWidget(QWidget):
    def __init__(self, title="", xlabel="", ylabel="", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Configure Look & Feel before creation
        self.plot_widget = pg.PlotWidget(title=title)
        self.plot_widget.setBackground(Theme.BACKGROUND)

        # Style axes
        styles = {"color": Theme.TEXT, "font-size": "14px"}
        self.plot_widget.setLabel("bottom", xlabel, **styles)
        self.plot_widget.setLabel("left", ylabel, **styles)
        self.plot_widget.getAxis("bottom").setPen(Theme.TEXT)
        self.plot_widget.getAxis("left").setPen(Theme.TEXT)

        # Title color logic handles HTML usually, pg defaults to grey
        # We can set title style via HTML in title string if needed, or:
        self.plot_widget.setTitle(title, color=Theme.TEXT, size="12pt")

        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color=Theme.PRIMARY, width=2))
        self.data_x = []
        self.data_y = []

    def add_point(self, x, y):
        self.data_x.append(x)
        self.data_y.append(y)
        self.curve.setData(self.data_x, self.data_y)

    def clear(self):
        self.data_x = []
        self.data_y = []
        self.curve.setData([], [])
