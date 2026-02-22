import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem, QGraphicsDropShadowEffect, QGraphicsSimpleTextItem
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter, QFont

class ClickableTileItem(QGraphicsRectItem):
    """A tile that emits a signal when clicked."""
    def __init__(self, layer_idx, tile_idx, x, y, size):
        super().__init__(0, 0, size, size)
        self.layer_idx = layer_idx
        self.tile_idx = tile_idx
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.visualizer = None

    def mousePressEvent(self, event):
        if self.visualizer:
            self.visualizer.tile_clicked.emit(self.layer_idx, self.tile_idx)
        super().mousePressEvent(event)

class EquiTileVisualizer(QGraphicsView):
    """
    A hypnotic, dark-themed visualization of the EquiTile neural network.
    Displays multiple layers of tiles as glowing grid cells.
    """
    tile_clicked = pyqtSignal(int, int) # layer_idx, tile_idx

    def __init__(self, num_layers=6, tiles_per_layer=64, grid_cols=8, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#0a0a0a")))

        self.num_layers = num_layers
        self.tiles_per_layer = tiles_per_layer
        self.grid_cols = grid_cols

        # Structure: self.tiles[layer_idx][tile_idx]
        self.tiles = []
        self.selected_tile = None # (layer_idx, tile_idx)

        # We don't call _init_grid here immediately if num_layers might be updated shortly,
        # but for default constructor behavior it's fine.
        # Window calls _init_grid via reconfigure anyway.
        # self._init_grid()

    def _init_grid(self):
        """Initializes the multi-layer grid."""
        self.scene().clear()
        self.tiles = []

        tile_size = 30
        padding = 5
        layer_padding = 40

        # Determine layout: Grid of Layers
        # e.g. 2 rows of 3 layers
        layer_grid_cols = max(1, min(3, self.num_layers))

        start_x = 40
        start_y = 40

        layer_width = self.grid_cols * (tile_size + padding)
        layer_height = ((self.tiles_per_layer + self.grid_cols - 1) // self.grid_cols) * (tile_size + padding)

        for l in range(self.num_layers):
            layer_tiles = []

            l_row = l // layer_grid_cols
            l_col = l % layer_grid_cols

            layer_offset_x = start_x + l_col * (layer_width + layer_padding)
            layer_offset_y = start_y + l_row * (layer_height + layer_padding + 30)

            # Label
            label = QGraphicsSimpleTextItem(f"Layer {l}")
            label.setBrush(QBrush(QColor("#00ffcc")))
            label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            label.setPos(layer_offset_x, layer_offset_y - 25)
            self.scene().addItem(label)

            for i in range(self.tiles_per_layer):
                row = i // self.grid_cols
                col = i % self.grid_cols

                x = layer_offset_x + col * (tile_size + padding)
                y = layer_offset_y + row * (tile_size + padding)

                tile = ClickableTileItem(l, i, x, y, tile_size)
                tile.visualizer = self

                # Default Appearance
                tile.setBrush(QBrush(QColor("#111111")))
                tile.setPen(QPen(QColor("#333333"), 1))

                # Glow
                effect = QGraphicsDropShadowEffect()
                effect.setBlurRadius(10)
                effect.setColor(QColor("#00ffcc"))
                effect.setEnabled(False)
                tile.setGraphicsEffect(effect)

                self.scene().addItem(tile)
                layer_tiles.append(tile)

            self.tiles.append(layer_tiles)

        # Adjust scene rect to fit content
        self.scene().setSceneRect(self.scene().itemsBoundingRect())


    def update_state(self, all_importances, all_activities, tile_losses=None):
        """
        Updates visual state for all layers.
        Args:
            all_importances: list of numpy arrays (tile importance sigmoids)
            all_activities: list of numpy arrays (tile activities)
            tile_losses: optional list of per-tile loss contributions
        """
        if len(all_importances) != self.num_layers:
            return

        for l in range(self.num_layers):
            layer_imps = all_importances[l]
            layer_acts = all_activities[l]

            if l >= len(self.tiles):
                continue

            for i, tile in enumerate(self.tiles[l]):
                if i >= len(layer_imps):
                    break

                imp = layer_imps[i]
                act = layer_acts[i]

                # Color based on importance (cyan to magenta gradient)
                hue = int(180 - (imp * 140))
                saturation = 255
                value = int(50 + imp * 150)
                base_color = QColor.fromHsv(hue, saturation, value)

                # Activity affects brightness and glow (THIS IS THE DYNAMIC PART!)
                if act > 0.02:
                    brightness_boost = int(100 + min(act * 100, 60))
                    color = base_color.lighter(brightness_boost)
                    
                    effect = tile.graphicsEffect()
                    if effect:
                        effect.setColor(color)
                        effect.setBlurRadius(int(5 + act * 25))
                        effect.setEnabled(True)
                else:
                    color = base_color.darker(140)
                    effect = tile.graphicsEffect()
                    if effect:
                        effect.setEnabled(False)

                tile.setBrush(QBrush(color))

                # Border based on importance
                pen_width = 1
                pen_color = QColor("#444444")

                is_selected = (self.selected_tile == (l, i))

                if is_selected:
                    pen_color = QColor("#ff00ff")
                    pen_width = 4
                elif imp > 0.7:
                    pen_color = QColor("#ffffff")
                    pen_width = 3
                elif imp > 0.4:
                    pen_color = QColor("#aaaaaa")
                    pen_width = 2
                elif imp < 0.1:
                    pen_color = QColor("#663333")
                    pen_width = 2

                tile.setPen(QPen(pen_color, pen_width))

    def set_selected_tile(self, layer_idx, tile_idx):
        """Sets the currently selected tile visually."""
        self.selected_tile = (layer_idx, tile_idx)
