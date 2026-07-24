#!/usr/bin/env python3
"""
EquiTile UI - Diagnostic Panel Additions
=========================================

Adds model health monitoring to detect problems early.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout


class GradientHealthPanel(QGroupBox):
    """Shows gradient statistics per layer to detect vanishing/exploding gradients."""

    def __init__(self, num_layers=4):
        super().__init__("Gradient Health")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #ff6600; border: 1px solid #555; }"
        )

        self.num_layers = num_layers
        layout = QVBoxLayout(self)

        # Create gradient bars for each layer
        self.bars = []
        self.labels = []
        self.status_labels = []

        for i in range(num_layers):
            row = QHBoxLayout()

            # Layer label
            lbl = QLabel(f"L{i}")
            lbl.setFixedWidth(30)
            row.addWidget(lbl)

            # Gradient magnitude bar
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v")
            row.addWidget(bar)
            self.bars.append(bar)

            # Status
            status = QLabel("—")
            status.setFixedWidth(80)
            row.addWidget(status)
            self.status_labels.append(status)

            layout.addLayout(row)

        # Warning label
        self.warning = QLabel("")
        self.warning.setStyleSheet("color: #ff0000; font-weight: bold;")
        layout.addWidget(self.warning)

    def update_gradients(self, grad_norms_per_layer):
        """Update with gradient norms from each layer."""
        warnings = []

        for i, norm in enumerate(grad_norms_per_layer):
            if i >= len(self.bars):
                break

            # Scale for display (log scale)
            if norm > 0:
                display_val = min(100, int(np.log10(norm + 1) * 25))
            else:
                display_val = 0

            self.bars[i].setValue(display_val)

            # Status
            if norm == 0:
                self.status_labels[i].setText("⚠ DEAD")
                self.status_labels[i].setStyleSheet("color: #ff0000;")
                warnings.append(f"Layer {i} has zero gradient")
            elif norm < 1e-6:
                self.status_labels[i].setText("⚠ VANISH")
                self.status_labels[i].setStyleSheet("color: #ff6600;")
                warnings.append(f"Layer {i} vanishing gradient ({norm:.2e})")
            elif norm > 10:
                self.status_labels[i].setText("⚠ EXPLODE")
                self.status_labels[i].setStyleSheet("color: #ff0000;")
                warnings.append(f"Layer {i} exploding gradient ({norm:.2f})")
            else:
                self.status_labels[i].setText("✓ OK")
                self.status_labels[i].setStyleSheet("color: #00ff00;")

        self.warning.setText(" | ".join(warnings) if warnings else "")


class SparsityTimelinePanel(QGroupBox):
    """Shows sparsity evolution over time."""

    def __init__(self, max_steps=200):
        super().__init__("Sparsity Timeline")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #00ccff; border: 1px solid #555; }"
        )

        layout = QVBoxLayout(self)

        # Timeline plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0a0a0a")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Steps")
        self.plot.setLabel("left", "Sparsity %")
        self.plot.setYRange(0, 100)

        self.sparsity_curve = self.plot.plot(pen=pg.mkPen("#00ccff", width=2))
        self.target_line = self.plot.plot(
            pen=pg.mkPen("#ff00ff", style=Qt.PenStyle.DashLine)
        )

        layout.addWidget(self.plot)

        # Stats
        self.stats_label = QLabel("Current: — | Min: — | Max: — | Trend: —")
        layout.addWidget(self.stats_label)

        self.history = []
        self.max_steps = max_steps

    def update_sparsity(self, sparsity_pct, target_pct=30):
        """Update with current sparsity percentage."""
        self.history.append(sparsity_pct)
        if len(self.history) > self.max_steps:
            self.history.pop(0)

        self.sparsity_curve.setData(self.history)
        self.target_line.setData([0, len(self.history)], [target_pct, target_pct])

        # Stats
        if len(self.history) > 10:
            recent = np.mean(self.history[-10:])
            older = (
                np.mean(self.history[-20:-10])
                if len(self.history) > 20
                else self.history[0]
            )
            trend = (
                "↑"
                if recent > older * 1.05
                else ("↓" if recent < older * 0.95 else "→")
            )
        else:
            trend = "—"

        self.stats_label.setText(
            f"Current: {sparsity_pct:.1f}% | "
            f"Min: {min(self.history):.1f}% | "
            f"Max: {max(self.history):.1f}% | "
            f"Trend: {trend}"
        )


class ActivationDistributionPanel(QGroupBox):
    """Shows activation histogram to detect dead neurons."""

    def __init__(self):
        super().__init__("Activation Distribution")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #00ff88; border: 1px solid #555; }"
        )

        layout = QVBoxLayout(self)

        # Histogram
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0a0a0a")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Activation")
        self.plot.setLabel("left", "Count")

        self.bars = pg.BarGraphItem(x=[], height=[], width=0.05, brush="#00ff88")
        self.plot.addItem(self.bars)

        layout.addWidget(self.plot)

        # Stats
        self.stats_label = QLabel("Dead: —% | Saturated: —%")
        layout.addWidget(self.stats_label)

    def update_activations(self, all_activities):
        """Update with activation data from all layers."""
        # Flatten all activations
        all_acts = np.concatenate([act.flatten() for act in all_activities])

        if len(all_acts) == 0:
            return

        # Histogram bins
        bins = np.linspace(0, 1, 20)
        hist, _ = np.histogram(all_acts, bins=bins)

        # Update bars
        x = (bins[:-1] + bins[1:]) / 2
        self.bars.setOpts(x=x, height=hist)

        # Stats
        dead = (all_acts < 0.01).sum()
        saturated = (all_acts > 0.9).sum()
        total = len(all_acts)

        self.stats_label.setText(
            f"Dead (<0.01): {dead / total * 100:.1f}% | "
            f"Saturated (>0.9): {saturated / total * 100:.1f}%"
        )

        # Color code dead neurons
        if dead / total > 0.2:
            self.stats_label.setStyleSheet("color: #ff0000;")
        elif dead / total > 0.1:
            self.stats_label.setStyleSheet("color: #ff6600;")
        else:
            self.stats_label.setStyleSheet("color: #00ff88;")


class ModelHealthSummary(QGroupBox):
    """Overall model health at a glance."""

    def __init__(self):
        super().__init__("Model Health Summary")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #00ff00; border: 2px solid #555; }"
        )

        layout = QVBoxLayout(self)

        # Status indicators
        self.status_layout = QVBoxLayout()

        self.gradient_status = QLabel("Gradients: —")
        self.status_layout.addWidget(self.gradient_status)

        self.accuracy_status = QLabel("Accuracy: —")
        self.status_layout.addWidget(self.accuracy_status)

        self.overfit_status = QLabel("Overfitting: —")
        self.status_layout.addWidget(self.overfit_status)

        self.sparsity_status = QLabel("Sparsity: —")
        self.status_layout.addWidget(self.sparsity_status)

        self.convergence_status = QLabel("Convergence: —")
        self.status_layout.addWidget(self.convergence_status)

        layout.addLayout(self.status_layout)

        # Auto-fix suggestions
        self.suggestion_label = QLabel("")
        self.suggestion_label.setStyleSheet("color: #ffff00; font-style: italic;")
        self.suggestion_label.setWordWrap(True)
        layout.addWidget(self.suggestion_label)

    def update(self, grad_stats, train_acc, test_acc, sparsity, loss_history):
        """Update health status."""
        # Gradient health
        if grad_stats["mean"] < 1e-8:
            self.gradient_status.setText("Gradients: ⚠️ VANISHING")
            self.gradient_status.setStyleSheet("color: #ff0000;")
        elif grad_stats["mean"] > 10:
            self.gradient_status.setText("Gradients: ⚠️ EXPLODING")
            self.gradient_status.setStyleSheet("color: #ff0000;")
        else:
            self.gradient_status.setText("Gradients: ✓ Normal")
            self.gradient_status.setStyleSheet("color: #00ff00;")

        # Accuracy trend
        if train_acc > 50:
            self.accuracy_status.setText(f"Accuracy: ✓ Good ({train_acc:.1f}%)")
            self.accuracy_status.setStyleSheet("color: #00ff00;")
        elif train_acc > 20:
            self.accuracy_status.setText(f"Accuracy: ~ Learning ({train_acc:.1f}%)")
            self.accuracy_status.setStyleSheet("color: #ffff00;")
        else:
            self.accuracy_status.setText(f"Accuracy: ⚠️ Low ({train_acc:.1f}%)")
            self.accuracy_status.setStyleSheet("color: #ff6600;")

        # Overfitting check
        if train_acc - test_acc > 10:
            self.overfit_status.setText(
                f"Overfitting: ⚠️ Gap={train_acc - test_acc:.1f}%"
            )
            self.overfit_status.setStyleSheet("color: #ff6600;")
        else:
            self.overfit_status.setText(
                f"Overfitting: ✓ OK (gap={abs(train_acc - test_acc):.1f}%)"
            )
            self.overfit_status.setStyleSheet("color: #00ff00;")

        # Sparsity check
        if sparsity > 40:
            self.sparsity_status.setText(f"Sparsity: ✓ High ({sparsity:.1f}%)")
            self.sparsity_status.setStyleSheet("color: #00ff00;")
        elif sparsity > 20:
            self.sparsity_status.setText(f"Sparsity: ~ Moderate ({sparsity:.1f}%)")
            self.sparsity_status.setStyleSheet("color: #ffff00;")
        else:
            self.sparsity_status.setText(f"Sparsity: ⚠️ Low ({sparsity:.1f}%)")
            self.sparsity_status.setStyleSheet("color: #ff6600;")

        # Convergence check
        if len(loss_history) > 20:
            recent = loss_history[-5:]
            older = loss_history[-20:-15]
            if len(older) > 0:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                change = (older_avg - recent_avg) / older_avg * 100
                if change < 1:
                    self.convergence_status.setText("Convergence: ✓ Converged")
                    self.convergence_status.setStyleSheet("color: #00ff00;")
                elif change > 0:
                    self.convergence_status.setText(
                        f"Convergence: ~ Learning ({change:.1f}%/20 steps)"
                    )
                    self.convergence_status.setStyleSheet("color: #ffff00;")
                else:
                    self.convergence_status.setText(
                        f"Convergence: ⚠️ Diverging ({abs(change):.1f}%)"
                    )
                    self.convergence_status.setStyleSheet("color: #ff0000;")

        # Generate suggestion
        suggestions = []
        if grad_stats["mean"] < 1e-8:
            suggestions.append("Increase learning rate")
        if grad_stats["mean"] > 10:
            suggestions.append("Decrease learning rate, enable gradient clipping")
        if train_acc - test_acc > 10:
            suggestions.append("Add regularization (dropout, increase sparsity)")
        if sparsity < 20:
            suggestions.append("Increase sparsity weight")
        if len(loss_history) > 20 and change < 0:
            suggestions.append(
                "Reduce learning rate, check for catastrophic forgetting"
            )

        if suggestions:
            self.suggestion_label.setText("💡 Try: " + "; ".join(suggestions))
        else:
            self.suggestion_label.setText("")


class AnomalyDetector:
    """Detects training anomalies and suggests fixes."""

    def __init__(self):
        self.loss_history = []
        self.sparsity_history = []
        self.last_alert = None
        self.alert_cooldown = 50  # Steps between alerts

    def check(self, step, loss, sparsity, grad_norms):
        """Check for anomalies. Returns (alert_message, suggestions) or None."""
        self.loss_history.append(loss)
        self.sparsity_history.append(sparsity)

        # Cooldown
        if self.last_alert and step - self.last_alert < self.alert_cooldown:
            return None

        alerts = []
        suggestions = []

        # Check loss spike
        if len(self.loss_history) > 5:
            recent = np.mean(self.loss_history[-5:])
            older = (
                np.mean(self.loss_history[-10:-5])
                if len(self.loss_history) > 10
                else self.loss_history[0]
            )

            if recent > older * 1.2:
                alerts.append("Loss increased >20%")
                suggestions.append("Try: Lower learning rate")

        # Check vanishing gradients
        if any(g < 1e-8 for g in grad_norms):
            alerts.append("Vanishing gradients detected")
            suggestions.append("Try: Higher learning rate, check initialization")

        # Check exploding gradients
        if any(g > 10 for g in grad_norms):
            alerts.append("Exploding gradients detected")
            suggestions.append("Try: Gradient clipping, lower learning rate")

        # Check sparsity collapse
        if len(self.sparsity_history) > 20:
            recent_sparsity = np.mean(self.sparsity_history[-10:])
            if recent_sparsity < 5:
                alerts.append("Sparsity collapsed (<5%)")
                suggestions.append("Try: Higher sparsity weight")

        if alerts:
            self.last_alert = step
            return {"step": step, "alerts": alerts, "suggestions": suggestions}

        return None
