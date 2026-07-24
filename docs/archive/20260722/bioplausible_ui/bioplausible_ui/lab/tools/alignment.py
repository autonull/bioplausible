import copy

import torch
import torch.nn.functional as F
from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class AlignmentWorker(QThread):
    """
    Background worker for Gradient Alignment Check.
    Compares model's update direction (EqProp) with Backprop gradient.
    """

    finished = pyqtSignal(dict)  # Returns dict of layer -> alignment
    error = pyqtSignal(str)

    def __init__(self, model, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset_name = dataset_name

    def run(self):
        try:
            from torch.utils.data import DataLoader

            from bioplausible.datasets import get_vision_dataset
            from bioplausible.models.registry import get_model_spec

            # 1. Setup Data
            ds_name = self.dataset_name.lower().replace("-", "_")
            use_flatten = True
            try:
                spec = get_model_spec(
                    self.model.config.name if hasattr(self.model, "config") else ""
                )
                use_flatten = spec.model_type != "modern_conv_eqprop"
            except:
                pass

            dataset = get_vision_dataset(ds_name, train=True, flatten=use_flatten)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            x, y = next(iter(loader))

            device = next(self.model.parameters()).device
            x, y = x.to(device), y.to(device)

            # 2. Backprop Reference
            # We need a clean copy to measure pure backprop gradient
            model_bp = copy.deepcopy(self.model)
            model_bp.train()
            model_bp.zero_grad()

            # Standard forward/backward
            # Handle models that require 'steps'
            try:
                out = model_bp(x)
            except TypeError:
                out = model_bp(x, steps=20)

            loss = F.cross_entropy(out, y)
            loss.backward()

            grads_bp = {}
            for name, param in model_bp.named_parameters():
                if param.grad is not None:
                    grads_bp[name] = param.grad.clone().flatten()

            # 3. EqProp Update
            # We need to capture what the model *would* update.
            # If the model has a 'train_step', it might apply optimizer step.
            # We clone to avoid messing up the main model state.
            model_eq = copy.deepcopy(self.model)
            model_eq.train()

            # Force SGD with LR=1.0 to measure update direction directly if internal optimizer used
            # If internal optimizer is created inside train_step, we might need to patch it.
            # Most EqProp models in this repo lazily create `internal_optimizer`.
            # We can force a manual SGD step if we can intercept gradients.

            # If gradient_method='contrastive', train_step computes grads.
            # If gradient_method='bptt', train_step might not exist or delegates.

            if hasattr(model_eq, "train_step"):
                # We want to capture the update.
                # Snapshot weights
                w_before = {n: p.data.clone() for n, p in model_eq.named_parameters()}

                # Mock optimizer to simple SGD LR=1.0 to extract gradient from delta
                # But model creates its own optimizer usually.
                # Try to set `hebbian_lr` to 1.0 if possible
                if hasattr(model_eq, "hebbian_lr"):
                    model_eq.hebbian_lr = 1.0
                if hasattr(model_eq, "learning_rate"):  # Config
                    model_eq.config.learning_rate = 1.0

                # Force reset internal optimizer if it exists
                if hasattr(model_eq, "internal_optimizer"):
                    model_eq.internal_optimizer = torch.optim.SGD(
                        model_eq.parameters(), lr=1.0
                    )

                # Run step
                model_eq.train_step(x, y)

                # Measure update
                grads_eq = {}
                for name, param in model_eq.named_parameters():
                    w_new = param.data
                    w_old = w_before[name]
                    # delta = -lr * grad  =>  grad = -(delta/lr) = -(w_new - w_old) = w_old - w_new
                    grads_eq[name] = (w_old - w_new).flatten()
            else:
                # If no train_step, it uses external trainer (BPTT).
                # Alignment is 1.0 by definition.
                grads_eq = grads_bp

            # 4. Compute Alignment
            alignments = {}
            for name in grads_bp:
                if name in grads_eq:
                    g_bp = grads_bp[name]
                    g_eq = grads_eq[name]

                    if g_bp.numel() > 0 and g_eq.numel() > 0:
                        sim = F.cosine_similarity(
                            g_bp.unsqueeze(0), g_eq.unsqueeze(0)
                        ).item()
                        alignments[name] = sim

            self.finished.emit(alignments)

        except Exception as e:
            self.error.emit(str(e))
            import traceback

            traceback.print_exc()


class AlignmentDialog(QDialog):
    """Dialog showing Gradient Alignment."""

    def __init__(self, alignments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gradient Alignment Analysis")
        self.resize(500, 600)

        layout = QVBoxLayout(self)

        header = QLabel("EqProp vs Backprop Alignment")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        desc = QLabel(
            "Cosine similarity between Equilibrium Propagation updates and standard Backpropagation gradients.\n"
            "1.0 = Perfect Alignment (Mathematically Identical)\n"
            "0.0 = Orthogonal\n"
            "-1.0 = Anti-Aligned"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a0b0;")
        layout.addWidget(desc)

        # Table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Layer", "Alignment (Cos)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Filter and sort
        items = [
            (k, v) for k, v in alignments.items() if "weight" in k
        ]  # Focus on weights
        table.setRowCount(len(items))

        avg_align = 0
        for i, (name, val) in enumerate(items):
            # Shorten name
            short_name = (
                name.replace(".weight", "")
                .replace("layers.", "L")
                .replace("parametrizations.original", "")
            )

            table.setItem(i, 0, QTableWidgetItem(short_name))

            val_item = QTableWidgetItem(f"{val:.4f}")
            if val > 0.9:
                val_item.setForeground(QColor("#00ff88"))
            elif val > 0.5:
                val_item.setForeground(QColor("#f1c40f"))
            else:
                val_item.setForeground(QColor("#ff5555"))
            table.setItem(i, 1, val_item)
            avg_align += val

        if items:
            avg_align /= len(items)

        layout.addWidget(table)

        # Global Score
        score_label = QLabel(f"Global Alignment: {avg_align:.4f}")
        score_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if avg_align > 0.9:
            score_label.setStyleSheet(
                "color: #00ff88; border: 2px solid #00ff88; padding: 10px; border-radius: 5px;"
            )
        else:
            score_label.setStyleSheet(
                "color: #f1c40f; border: 2px solid #f1c40f; padding: 10px; border-radius: 5px;"
            )

        layout.addWidget(score_label)

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


@ToolRegistry.register("alignment", requires=["alignment"])
class AlignmentTool(BaseTool):
    ICON = "📐"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Alignment Tool"))
        self.layout.addWidget(
            QLabel("Measure gradient alignment with backpropagation.")
        )

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(
            ["MNIST", "Fashion-MNIST", "CIFAR-10", "KMNIST", "SVHN"]
        )
        self.layout.addWidget(QLabel("Dataset:"))
        self.layout.addWidget(self.dataset_combo)

        self.check_btn = QPushButton("Check Alignment")
        self.check_btn.clicked.connect(self._check_alignment)
        self.layout.addWidget(self.check_btn)

        self.layout.addStretch()

    def _check_alignment(self):
        self.check_btn.setEnabled(False)
        self.check_btn.setText("Checking...")

        dataset = self.dataset_combo.currentText()
        self.worker = AlignmentWorker(self.model, dataset)
        self.worker.finished.connect(self._show_results)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _show_results(self, results):
        self.check_btn.setEnabled(True)
        self.check_btn.setText("Check Alignment")
        dlg = AlignmentDialog(results, self)
        dlg.exec()

    def _on_error(self, msg):
        self.check_btn.setEnabled(True)
        self.check_btn.setText("Check Alignment")
        QMessageBox.critical(self, "Error", msg)
