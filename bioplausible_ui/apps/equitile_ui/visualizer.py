import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter

class ClickableTileItem(QGraphicsRectItem):
    """A tile that emits a signal when clicked."""
    def __init__(self, index, x, y, size):
        super().__init__(0, 0, size, size)
        self.index = index
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        # Store reference to visualizer to emit signal
        self.visualizer = None

    def mousePressEvent(self, event):
        if self.visualizer:
            self.visualizer.tile_clicked.emit(self.index)
        super().mousePressEvent(event)

class EquiTileVisualizer(QGraphicsView):
    """
    A hypnotic, dark-themed visualization of the EquiTile neural network.
    Displays tiles as glowing grid cells that pulse with activity and change color based on importance.
    """
    tile_clicked = pyqtSignal(int)

    def __init__(self, num_tiles=64, grid_cols=8, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#0a0a0a"))) # Deep void black background

        self.num_tiles = num_tiles
        self.grid_cols = grid_cols
        self.tiles = []
        self.selected_tile_index = None

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
            tile = ClickableTileItem(i, x, y, tile_size)
            tile.visualizer = self

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

            # Border logic (Selected vs Important vs Normal)
            pen_width = 1
            pen_color = QColor("#333333")

            if i == self.selected_tile_index:
                pen_color = QColor("#ff00ff") # Magenta for selection
                pen_width = 3
            elif imp > 0.8:
                pen_color = QColor("#ffffff") # White for importance
                pen_width = 2

            tile.setPen(QPen(pen_color, pen_width))

    def set_selected_tile(self, index):
        """Sets the currently selected tile visually."""
        self.selected_tile_index = index
        # Trigger update to redraw borders? No, wait for next update cycle or force redraw
        # We can just update the pen of all tiles, but easier to wait for update_state
        # or manually update if paused.
        # Let's manually update just the borders to be responsive
        for i, tile in enumerate(self.tiles):
            pen_width = 1
            pen_color = QColor("#333333")
            # We don't have 'imp' here easily without storing it,
            # so we might lose the 'importance' highlight until next tick.
            # That's acceptable for now.
            if i == self.selected_tile_index:
                pen_color = QColor("#ff00ff")
                pen_width = 3
            tile.setPen(QPen(pen_color, pen_width))
