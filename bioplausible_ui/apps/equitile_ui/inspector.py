import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class TileInspector(QWidget):
    """
    Widget to inspect the internal state of a single selected tile.
    Shows neuron activations and tile-specific metrics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header
        self.group = QGroupBox("Tile Inspector")
        self.group.setStyleSheet(
            "QGroupBox { font-size: 14px; font-weight: bold; color: #00ffcc; border: 1px solid #333; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }"
        )
        group_layout = QVBoxLayout(self.group)

        # Info Labels
        info_layout = QHBoxLayout()
        self.id_label = QLabel("Layer: - | Tile: -")
        self.id_label.setStyleSheet("color: #aaaaaa;")
        self.imp_label = QLabel("Imp: --")
        self.imp_label.setStyleSheet("color: #00ffcc;")
        info_layout.addWidget(self.id_label)
        info_layout.addWidget(self.imp_label)
        group_layout.addLayout(info_layout)

        # Activity Bar
        self.activity_bar = QProgressBar()
        self.activity_bar.setRange(0, 100)
        self.activity_bar.setTextVisible(False)
        self.activity_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                background-color: #111;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #00ffcc;
            }
        """)
        group_layout.addWidget(QLabel("Mean Activity:"))
        group_layout.addWidget(self.activity_bar)

        # Neuron Heatmap/Grid
        self.neuron_plot = pg.PlotWidget()
        self.neuron_plot.setBackground("#0a0a0a")
        self.neuron_plot.setMouseEnabled(x=False, y=False)
        self.neuron_plot.hideAxis("bottom")
        self.neuron_plot.hideAxis("left")
        self.neuron_plot.setTitle("Neuron Activations (64)")

        self.heatmap_item = pg.ImageItem()
        self.heatmap_item.setOpts(axisOrder="row-major")

        pos = np.array([0.0, 0.25, 0.75, 1.0])
        color = np.array(
            [
                [0, 0, 0, 255],
                [0, 0, 255, 255],
                [0, 255, 255, 255],
                [255, 255, 255, 255],
            ],
            dtype=np.ubyte,
        )
        cmap = pg.ColorMap(pos, color)
        self.heatmap_item.setLookupTable(cmap.getLookupTable())

        self.neuron_plot.addItem(self.heatmap_item)
        group_layout.addWidget(self.neuron_plot)

        layout.addWidget(self.group)

        self.clear_inspector()

    def update_tile_data(
        self, layer_id, tile_id, importance, activity_val, neuron_states
    ):
        """
        Update the inspector with data for the specific tile.
        """
        self.id_label.setText(f"Layer: {layer_id} | Tile: {tile_id}")
        self.imp_label.setText(f"Imp: {importance:.2f}")

        val = int(min(activity_val * 50, 100))  # Scale up activity for visibility
        self.activity_bar.setValue(val)

        if neuron_states is not None:
            size = neuron_states.size
            side = int(np.sqrt(size))
            if side * side == size:
                img_data = neuron_states.reshape(side, side)
            else:
                img_data = neuron_states.reshape(1, size)

            self.heatmap_item.setImage(img_data, autoLevels=False, levels=(0.0, 2.0))

    def clear_inspector(self):
        self.id_label.setText("Layer: - | Tile: -")
        self.imp_label.setText("Imp: --")
        self.activity_bar.setValue(0)
        self.heatmap_item.setImage(np.zeros((8, 8)))
