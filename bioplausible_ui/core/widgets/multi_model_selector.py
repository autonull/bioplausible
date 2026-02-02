"""
Multi-Model Selector Widget

Allows selecting multiple models/algorithms for parallel comparison.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QVBoxLayout,
                             QWidget)

from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec


class MultiModelSelector(QWidget):
    """Widget for selecting multiple models."""

    valueChanged = pyqtSignal(list)  # Emits list of selected model names

    def __init__(self, task="vision", parent=None):
        super().__init__(parent)
        self.task = task
        self.setup_ui()
        self.update_models(task)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with actions
        header = QHBoxLayout()
        header.addWidget(QLabel("Algorithms to Compare:"))
        header.addStretch()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 4px 8px; }
        """)
        self.select_all_btn.clicked.connect(self.select_all)
        header.addWidget(self.select_all_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 4px 8px; }
        """)
        self.clear_btn.clicked.connect(self.clear_selection)
        header.addWidget(self.clear_btn)

        layout.addLayout(header)

        # List of models
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QListWidget.SelectionMode.NoSelection
        )  # We use checkboxes
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

    def set_task(self, task):
        self.task = task
        self.update_models(task)

    def update_models(self, task):
        self.list_widget.clear()

        # Group models by family
        models_by_family = {}
        for name, spec in MODEL_REGISTRY.items():
            if spec.task_compat is None or task in spec.task_compat:
                family = spec.family if hasattr(spec, "family") else "Other"
                models_by_family.setdefault(family, []).append(name)

        # Sort families
        sorted_families = sorted(models_by_family.keys())

        # Move baseline/backprop to top
        if "baseline" in sorted_families:
            sorted_families.remove("baseline")
            sorted_families.insert(0, "baseline")

        for family in sorted_families:
            # Family Header (optional, visual only)
            # family_item = QListWidgetItem(f"--- {family.upper()} ---")
            # family_item.setFlags(Qt.ItemFlag.NoItemFlags)
            # self.list_widget.addItem(family_item)

            for model in sorted(models_by_family[family]):
                item = QListWidgetItem()
                self.list_widget.addItem(item)

                # Checkbox widget
                chk = QCheckBox(model)
                chk.toggled.connect(self._emit_change)
                # Store checkbox in item widget
                self.list_widget.setItemWidget(item, chk)
                # Store model name in item data for retrieval (though widget is separate)
                item.setData(Qt.ItemDataRole.UserRole, model)

    def get_selected_models(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            chk = self.list_widget.itemWidget(item)
            if chk and chk.isChecked():
                selected.append(chk.text())
        return selected

    def select_all(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            chk = self.list_widget.itemWidget(item)
            if chk:
                chk.setChecked(True)

    def clear_selection(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            chk = self.list_widget.itemWidget(item)
            if chk:
                chk.setChecked(False)

    def _on_item_changed(self, item):
        pass  # Managed by checkbox signals

    def _emit_change(self):
        self.valueChanged.emit(self.get_selected_models())
