import math

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QGraphicsDropShadowEffect, QGraphicsItem,
                             QGraphicsRectItem, QGraphicsScene,
                             QGraphicsSimpleTextItem, QGraphicsView)


class ClickableTileItem(QGraphicsRectItem):
    """A tile that emits a signal when clicked."""

    def __init__(self, layer_idx, tile_idx, x, y, size):
        super().__init__(0, 0, size, size)
        self.layer_idx = layer_idx
        self.tile_idx = tile_idx
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.visualizer = None
        self.setToolTip(f"Layer {layer_idx}, Tile {tile_idx}")

    def mousePressEvent(self, event):
        if self.visualizer:
            self.visualizer.tile_clicked.emit(self.layer_idx, self.tile_idx)
        super().mousePressEvent(event)

    def set_tooltip_data(self, act, imp):
        self.setToolTip(
            f"Layer {self.layer_idx}, Tile {self.tile_idx}\nAct: {act:.3f}\nImp: {imp:.3f}"
        )


class LayerGridVisualizer(QGraphicsView):
    """
    A hypnotic, dark-themed visualization of generic neural network layers.
    Displays multiple layers of units (tiles) as glowing grid cells.
    """

    tile_clicked = pyqtSignal(int, int)  # layer_idx, tile_idx

    def __init__(
        self,
        layer_sizes=None,
        tiles_per_layer=None,
        num_layers=None,
        grid_cols=None,
        parent=None,
    ):
        """
        Args:
            layer_sizes: List of ints, number of units per layer.
            tiles_per_layer: (Legacy) If layer_sizes is None, uses this * num_layers.
            num_layers: (Legacy) Used with tiles_per_layer.
            grid_cols: Override columns per layer grid. If None, calculated automatically.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#0a0a0a")))

        # Handle legacy or flexible init
        if layer_sizes is None:
            if tiles_per_layer is not None and num_layers is not None:
                self.layer_sizes = [tiles_per_layer] * num_layers
            else:
                self.layer_sizes = []
        else:
            self.layer_sizes = layer_sizes

        self.user_grid_cols = grid_cols

        # Structure: self.tiles[layer_idx][tile_idx]
        self.tiles = []
        self.selected_tile = None  # (layer_idx, tile_idx)

        # Initialize if we have sizes
        if self.layer_sizes:
            self._init_grid()

    def _init_grid(self):
        """Initializes the multi-layer grid."""
        self.scene().clear()
        self.tiles = []

        if not self.layer_sizes:
            return

        tile_size = 30
        padding = 5
        layer_padding = 40

        num_layers = len(self.layer_sizes)

        # Determine layout: Grid of Layers
        # e.g. 2 rows of 3 layers
        layer_grid_cols = max(1, min(3, num_layers))

        start_x = 40
        start_y = 40

        # Determine max layer width to align nicely
        # Or calculate per layer.

        # We need to track current position
        current_x = start_x
        current_y = start_y
        max_row_h = 0

        for l, size in enumerate(self.layer_sizes):
            layer_tiles = []

            # Determine grid dimensions for this layer
            if self.user_grid_cols:
                cols = self.user_grid_cols
            else:
                # Heuristic: try to be square-ish
                cols = math.ceil(math.sqrt(size))
                # Cap columns if too wide?
                if cols > 16:
                    cols = 16

            if size == 0:
                cols = 1  # Safety
            rows = (size + cols - 1) // cols

            layer_w = cols * (tile_size + padding)
            layer_h = rows * (tile_size + padding)

            # Position the layer
            # Simple grid of layers
            l_row = l // layer_grid_cols
            l_col = l % layer_grid_cols

            # We assume roughly equal sized layers for simple layout
            # A more complex packer could be used but this is fine.
            # Let's assume max possible width/height based on largest layer or just flow.
            # To keep it simple, we re-calculate fixed offsets like before but adaptively.
            # But since layer sizes vary, fixed offset is risky.

            # Let's just use the previous logic but with dynamic inner grid
            # If we want a strictly aligned grid of layers, we need max dimensions.

            # For now, let's just use fixed large slots
            slot_w = 400
            slot_h = 400

            layer_offset_x = start_x + l_col * (slot_w + layer_padding)
            layer_offset_y = start_y + l_row * (slot_h + layer_padding + 30)

            # Label
            label = QGraphicsSimpleTextItem(f"Layer {l} ({size})")
            label.setBrush(QBrush(QColor("#00ffcc")))
            label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            label.setPos(layer_offset_x, layer_offset_y - 25)
            self.scene().addItem(label)

            for i in range(size):
                # Inner grid
                row = i // cols
                col = i % cols

                x = layer_offset_x + col * (tile_size + padding)
                y = layer_offset_y + row * (tile_size + padding)

                # Check if we overflow the slot (just purely visual)
                # If layer is huge, we might overlap.
                # Ideally we'd scroll or shrink tiles.
                # For huge layers (e.g. 784), tile_size=30 is too big -> 28x30 ~ 900px
                # Maybe scale tile size?
                this_tile_size = tile_size
                if size > 256:
                    this_tile_size = 10
                    # Recalculate x,y with smaller size
                    x = layer_offset_x + col * (this_tile_size + 2)
                    y = layer_offset_y + row * (this_tile_size + 2)

                tile = ClickableTileItem(l, i, x, y, this_tile_size)
                tile.visualizer = self

                # Default Appearance
                tile.setBrush(QBrush(QColor("#111111")))
                tile.setPen(QPen(QColor("#333333"), 1))

                # Glow (expensive for many tiles, disable initially)
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

    # Backward compatibility alias properties
    @property
    def num_layers(self):
        return len(self.layer_sizes)

    @num_layers.setter
    def num_layers(self, value):
        # Can't easily set length without sizes.
        # Assume reconfiguration will happen.
        pass

    @property
    def tiles_per_layer(self):
        return self.layer_sizes[0] if self.layer_sizes else 0

    @tiles_per_layer.setter
    def tiles_per_layer(self, value):
        if self.layer_sizes:
            self.layer_sizes = [value] * len(self.layer_sizes)

    def update_state(self, all_importances, all_activities, tile_losses=None):
        """
        Updates visual state for all layers.
        Args:
            all_importances: list of numpy arrays (tile importance sigmoids)
            all_activities: list of numpy arrays (tile activities)
            tile_losses: optional list of per-tile loss contributions
        """
        # Ensure lists
        if not isinstance(all_importances, list):
            return
        if not isinstance(all_activities, list):
            return

        for l in range(len(self.tiles)):
            if l >= len(all_importances) or l >= len(all_activities):
                break

            layer_imps = all_importances[l]
            layer_acts = all_activities[l]

            # Handle mismatch in size (e.g. during reconfiguration)
            if len(layer_imps) != len(self.tiles[l]):
                continue

            for i, tile in enumerate(self.tiles[l]):
                imp = layer_imps[i]
                act = layer_acts[i]

                # Color based on importance (cyan to magenta gradient)
                hue = int(180 - (imp * 140))
                saturation = 255
                value = int(50 + imp * 150)
                base_color = QColor.fromHsv(hue, saturation, value)

                # Activity affects brightness and glow
                # For generic models, act might be unbounded or negative (tanh).
                # Normalize heuristic: abs(tanh) is 0..1. Relu is 0..inf.
                # Just clip visual brightness.

                vis_act = abs(act)

                if vis_act > 0.05:
                    brightness_boost = int(100 + min(vis_act * 100, 60))
                    color = base_color.lighter(brightness_boost)

                    # Only apply glow if not too many tiles (perf)
                    if len(self.tiles[l]) < 300:
                        effect = tile.graphicsEffect()
                        if effect:
                            effect.setColor(color)
                            effect.setBlurRadius(int(5 + min(vis_act * 25, 20)))
                            effect.setEnabled(True)
                else:
                    color = base_color.darker(140)
                    effect = tile.graphicsEffect()
                    if effect:
                        effect.setEnabled(False)

                tile.setBrush(QBrush(color))

                # Update tooltip
                tile.set_tooltip_data(act, imp)

                # Border based on importance
                pen_width = 1
                pen_color = QColor("#444444")

                is_selected = self.selected_tile == (l, i)

                if is_selected:
                    pen_color = QColor("#ff00ff")
                    pen_width = 4
                elif imp > 0.7:
                    pen_color = QColor("#ffffff")
                    pen_width = 3 if tile.rect().width() > 15 else 1

                tile.setPen(QPen(pen_color, pen_width))

    def set_selected_tile(self, layer_idx, tile_idx):
        """Sets the currently selected tile visually."""
        self.selected_tile = (layer_idx, tile_idx)


# Alias for backward compatibility
EquiTileVisualizer = LayerGridVisualizer
