import numpy as np
import torch
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QComboBox, QDialog, QHeaderView, QLabel,
                             QMessageBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)
from torch.utils.data import DataLoader

from bioplausible.datasets import get_vision_dataset
from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool


class RobustnessDialog(QDialog):
    """Dialog to show robustness check results."""

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robustness Analysis")
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        label = QLabel("Noise Tolerance Analysis (Gaussian Noise)")
        label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(label)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Noise Level (σ)", "Accuracy"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setRowCount(len(results))

        for i, (noise, acc) in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(f"{noise:.1f}"))

            acc_item = QTableWidgetItem(f"{acc:.1%}")
            if acc > 0.8:
                acc_item.setForeground(QColor("#00ff88"))
            elif acc > 0.5:
                acc_item.setForeground(QColor("#f1c40f"))
            else:
                acc_item.setForeground(QColor("#ff5555"))

            self.table.setItem(i, 1, acc_item)

        layout.addWidget(self.table)

        # Summary
        drops = [results[0][1] - r[1] for r in results[1:]]
        avg_drop = sum(drops) / len(drops) if drops else 0.0

        summary = "Robustness Score: "
        if avg_drop < 0.1:
            summary += "<span style='color: #00ff88'>Excellent</span>"
        elif avg_drop < 0.3:
            summary += "<span style='color: #f1c40f'>Good</span>"
        else:
            summary += "<span style='color: #ff5555'>Poor</span>"

        sum_label = QLabel(summary)
        sum_label.setFont(QFont("Segoe UI", 14))
        layout.addWidget(sum_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


@ToolRegistry.register("robustness", requires=["robustness"])
class RobustnessTool(BaseTool):
    ICON = "🛡️"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Robustness Tool"))
        self.layout.addWidget(QLabel("Test model robustness against noise."))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(
            ["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"]
        )
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_combo)

        self.test_btn = QPushButton("Run Robustness Test")
        self.test_btn.clicked.connect(self._run_test)
        self.layout.addWidget(self.test_btn)

        self.layout.addStretch()

    def _run_test(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return

        try:
            self.test_btn.setEnabled(False)
            self.test_btn.setText("Running...")

            # Logic adapted from VisionTab._run_robustness_check
            ds_name = self.dataset_combo.currentText().lower().replace("-", "_")
            use_flatten = True  # Simplified assumption for now

            dataset = get_vision_dataset(ds_name, train=False, flatten=use_flatten)
            subset_indices = np.random.choice(len(dataset), 200, replace=False)
            subset = torch.utils.data.Subset(dataset, subset_indices)
            loader = DataLoader(subset, batch_size=50, shuffle=False)

            device = next(self.model.parameters()).device
            self.model.eval()

            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            results = []

            with torch.no_grad():
                for noise_sigma in noise_levels:
                    correct = 0
                    total = 0
                    for x, y in loader:
                        x = x.to(device)
                        y = y.to(device)
                        if noise_sigma > 0:
                            x = x + torch.randn_like(x) * noise_sigma
                        try:
                            out = self.model(x)
                        except TypeError:
                            out = self.model(x, steps=20)
                        pred = out.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                    results.append((noise_sigma, correct / total))

            dlg = RobustnessDialog(results, self)
            dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("Run Robustness Test")
