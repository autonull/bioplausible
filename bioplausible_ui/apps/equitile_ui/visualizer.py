import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter

class EquiTileVisualizer(QGraphicsView):
    """
    A hypnotic, dark-themed visualization of the EquiTile neural network.
    Displays tiles as glowing grid cells that pulse with activity and change color based on importance.
    """
    def __init__(self, num_tiles=64, grid_cols=8, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#0a0a0a"))) # Deep void black background

        self.num_tiles = num_tiles
        self.grid_cols = grid_cols
        self.tiles = []

        self._init_grid()

    def _init_grid(self):
        """Initializes the grid of tiles."""
        self.scene().clear()
        self.tiles = []

        tile_size = 40
        padding = 10
        start_x = 50
        start_y = 50

        rows = (self.num_tiles + self.grid_cols - 1) // self.grid_cols

        for i in range(self.num_tiles):
            row = i // self.grid_cols
            col = i % self.grid_cols

            x = start_x + col * (tile_size + padding)
            y = start_y + row * (tile_size + padding)

            # Create Tile Item
            tile = QGraphicsRectItem(0, 0, tile_size, tile_size)
            tile.setPos(x, y)

            # Default Appearance (Inactive)
            tile.setBrush(QBrush(QColor("#111111")))
            tile.setPen(QPen(QColor("#333333"), 1))

            # Add Glow Effect (Initially off/low)
            effect = QGraphicsDropShadowEffect()
            effect.setBlurRadius(15)
            effect.setColor(QColor("#00ffcc"))
            effect.setOffset(0, 0)
            effect.setEnabled(False) # Enable when active
            tile.setGraphicsEffect(effect)

            self.scene().addItem(tile)
            self.tiles.append(tile)

    def update_state(self, importance_scores, activity_levels):
        """
        Updates the visual state of the tiles based on model metrics.

        Args:
            importance_scores (list/array): Importance values [0, 1] per tile.
            activity_levels (list/array): Activity/activation levels [0, 1] per tile.
        """
        if len(importance_scores) != self.num_tiles:
            return # Mismatch in tile count

        for i, tile in enumerate(self.tiles):
            imp = importance_scores[i]
            act = activity_levels[i] if activity_levels is not None else 0.0

            # Color Mapping:
            # Low importance -> Dark / Gray
            # High importance -> Cyan / Neon Green
            # High activity -> Bright / White flash

            # Base color based on importance (Cyan to Lime gradient)
            # Hue: Cyan (180) to Lime (120)
            hue = 180 - (imp * 60)
            saturation = 200 # 0-255
            value = 50 + (imp * 150) # Brightness based on importance

            color = QColor.fromHsv(int(hue), int(saturation), int(value))

            # Pulse/Flash based on activity
            if act > 0.1:
                # Add white to the color for "activation flash"
                color = color.lighter(int(100 + act * 100))

                # Enable glow
                effect = tile.graphicsEffect()
                if effect:
                    effect.setColor(color)
                    effect.setBlurRadius(15 + act * 20) # Larger glow for higher activity
                    effect.setEnabled(True)
            else:
                # Disable glow for inactive tiles
                effect = tile.graphicsEffect()
                if effect:
                    effect.setEnabled(False)

            tile.setBrush(QBrush(color))

            # Border highlight for very important tiles
            if imp > 0.8:
                tile.setPen(QPen(QColor("#ffffff"), 2))
            else:
                tile.setPen(QPen(QColor("#333333"), 1))
