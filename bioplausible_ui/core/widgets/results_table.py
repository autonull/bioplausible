from PyQt6.QtWidgets import (QHeaderView, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)


class ResultsTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Timestamp", "ID", "Task", "Model", "Metric"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def add_run(self, run_id, timestamp, task, model, metric_val):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.table.setItem(row, 1, QTableWidgetItem(str(run_id)))
        self.table.setItem(row, 2, QTableWidgetItem(task))
        self.table.setItem(row, 3, QTableWidgetItem(model))
        self.table.setItem(row, 4, QTableWidgetItem(f"{metric_val:.4f}"))

    def clear_table(self):
        self.table.setRowCount(0)

    def get_selected_run_id(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            return self.table.item(current_row, 1).text()
        return None
