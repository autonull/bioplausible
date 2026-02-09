"""
Experiment Tab - Comprehensive Survey Runner
"""

import itertools

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QTextEdit, QVBoxLayout)

from bioplausible.hyperopt import create_optuna_space, create_study
from bioplausible.hyperopt.eval_tiers import (PatientLevel,
                                              get_evaluation_config)
from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible_ui.core.base import BaseTab
# Import RadarView if available, else standard import
from bioplausible_ui.core.widgets.radar_view import RadarView


class ExperimentWorker(QThread):
    """Worker thread for Comprehensive Experiments (Models x Tasks)."""

    trial_finished = pyqtSignal(dict)
    finished = pyqtSignal()
    search_started = pyqtSignal(str, str)  # Emits (model, task)

    def __init__(
        self, tasks, models, tier=PatientLevel.SHALLOW, overrides=None, parent=None
    ):
        super().__init__(parent)
        self.tasks = tasks if isinstance(tasks, list) else [tasks]
        self.models = models if isinstance(models, list) else [models]
        self.tier = tier
        self.overrides = overrides or {}

        # Get base config for this tier
        self.base_config = get_evaluation_config(tier)

        # Apply overrides
        self.max_trials = self.overrides.get("trials", self.base_config.n_trials)
        self.epochs = self.overrides.get("epochs", self.base_config.epochs)
        self.repeats = self.overrides.get("repeats", 1)

        self.running = True

    def run(self):
        # Cartesian Product: All Models x All Tasks
        combinations = list(itertools.product(self.tasks, self.models))

        for task, model in combinations:
            if not self.running:
                break

            self.search_started.emit(model, task)

            # Create Optuna study per (model, task, tier)
            study_name = f"{model}_{task}_{self.tier.value}"

            try:
                study = create_study(
                    model_names=[model],
                    n_objectives=2,  # accuracy, loss
                    storage=f"sqlite:///bioplausible.db",
                    study_name=study_name,
                    use_pruning=self.base_config.use_pruning,
                    sampler_name="tpe",
                )

                current_model = model
                current_task = task

                def objective(trial):
                    if not self.running:
                        import optuna

                        raise optuna.TrialPruned()

                    # Sample hyperparameters ONCE for all repeats
                    config = create_optuna_space(trial, current_model)

                    # Apply Config (Tier + Overrides)
                    config["epochs"] = self.epochs
                    config["batch_size"] = self.base_config.batch_size

                    # Tag the trial
                    trial.set_user_attr("tier", self.tier.value)
                    trial.set_user_attr("model_family", current_model)
                    trial.set_user_attr("task", current_task)
                    trial.set_user_attr("repeats", self.repeats)

                    aggregated_metrics = {
                        "accuracy": [],
                        "loss": [],
                        "time": [],
                        "param_count": [],
                    }

                    # Execute Repeats
                    for r in range(self.repeats):
                        if not self.running:
                            import optuna

                            raise optuna.TrialPruned()

                        # Run trial
                        # Use shifted ID to avoid collision with future Optuna IDs
                        # and ensure unique logging for each repeat
                        # e.g. Trial 128 -> IDs 12800, 12801, 12802...
                        shifted_id = trial.number * 100 + r

                        metrics = run_single_trial_task(
                            task=current_task,
                            model_name=current_model,
                            config=config,
                            storage_path="bioplausible.db",
                            ## job_id removed - Auto-Increment enforced
                            quick_mode=(self.tier == PatientLevel.SMOKE),
                        )

                        if metrics:
                            # Capture the REAL database ID (PK) from the first successful repeat
                            # This ensures the UI links to the correct row, not the Optuna ID (0, 1...)
                            if len(aggregated_metrics["accuracy"]) == 0:
                                last_db_id = metrics.get("trial_id", -1)

                            aggregated_metrics["accuracy"].append(
                                metrics.get("accuracy", 0.0)
                            )
                            aggregated_metrics["loss"].append(
                                metrics.get("loss", float("inf"))
                            )
                            aggregated_metrics["time"].append(metrics.get("time", 0.0))
                            aggregated_metrics["param_count"].append(
                                metrics.get("param_count", 0.0)
                            )
                        else:
                            # If a repeat fails, we might want to prune or continue
                            pass

                    # Average results
                    n_success = len(aggregated_metrics["accuracy"])
                    if n_success > 0:
                        avg_acc = sum(aggregated_metrics["accuracy"]) / n_success
                        avg_loss = sum(aggregated_metrics["loss"]) / n_success
                        avg_time = sum(aggregated_metrics["time"]) / n_success
                        avg_params = sum(aggregated_metrics["param_count"]) / n_success

                        # Report to UI (Averaged)
                        result = {
                            "model": current_model,
                            "task": current_task,
                            "params": config,
                            "accuracy": avg_acc,
                            "loss": avg_loss,
                            "time": avg_time,
                            "param_count": avg_params,
                            "trial_number": trial.number,
                            "trial_id": last_db_id,  # Use real DB ID (from last repeat)
                            "repeats": n_success,
                        }
                        self.trial_finished.emit(result)

                        return avg_acc, avg_loss
                    else:
                        import optuna

                        raise optuna.TrialPruned()

                study.optimize(
                    objective, n_trials=self.max_trials, show_progress_bar=False
                )

            except Exception as e:
                print(f"Error executing {model} on {task}: {e}")
                # Continue to next combination even if one fails
                continue

        self.finished.emit()

    def stop(self):
        self.running = False


from bioplausible_ui.leaderboard.leaderboard_window import LeaderboardWindow


class ExperimentTab(BaseTab):
    """Experiment Tab - Comprehensive Survey Runner."""

    SCHEMA = None  # We build UI manually
    transfer_config = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.leaderboard_window = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Top Control Panel ---
        controls_group = QGroupBox("Experiment Configuration")
        controls_layout = QHBoxLayout(controls_group)

        # 1. Algorithm Selection
        algo_layout = QVBoxLayout()
        algo_layout.addWidget(QLabel("Algorithms:"))
        self.algo_list = QListWidget()
        # Populate from registry
        for spec in MODEL_REGISTRY:
            item = QListWidgetItem(spec.name)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.algo_list.addItem(item)
        # Select defaults (Smoke favorites?)
        # For now leave unchecked or select backprop
        self.algo_list.item(0).setCheckState(Qt.CheckState.Checked)
        algo_layout.addWidget(self.algo_list)
        controls_layout.addLayout(algo_layout, 2)

        # 2. Task Selection
        task_layout = QVBoxLayout()
        task_layout.addWidget(QLabel("Tasks:"))
        self.task_list = QListWidget()
        tasks = ["vision", "lm", "rl"]
        for t in tasks:
            item = QListWidgetItem(t.upper())
            item.setData(Qt.ItemDataRole.UserRole, t)  # Store internal name
            item.setCheckState(Qt.CheckState.Unchecked)
            self.task_list.addItem(item)
        self.task_list.item(0).setCheckState(Qt.CheckState.Checked)  # Default Vision
        task_layout.addWidget(self.task_list)
        controls_layout.addLayout(task_layout, 1)

        # 3. Settings & Overrides
        settings_layout = QVBoxLayout()

        # Tier
        settings_layout.addWidget(QLabel("Depth (Tier):"))
        self.tier_combo = QComboBox()
        self.tier_combo.addItems(
            ["Smoke (1 min)", "Shallow (10 min)", "Standard (1 hr)", "Deep (Overnight)"]
        )
        settings_layout.addWidget(self.tier_combo)

        # Overrides
        overrides_group = QGroupBox("Overrides")
        overrides_form = QVBoxLayout(overrides_group)

        self.epoch_override = QSpinBox()
        self.epoch_override.setRange(1, 1000)
        self.epoch_override.setValue(5)  # Default placeholder
        self.epoch_override.setPrefix("Epochs: ")
        overrides_form.addWidget(self.epoch_override)

        self.trials_override = QSpinBox()
        self.trials_override.setRange(1, 1000)
        self.trials_override.setValue(10)  # Default placeholder
        self.trials_override.setPrefix("Trials: ")
        overrides_form.addWidget(self.trials_override)

        self.repeats_override = QSpinBox()
        self.repeats_override.setRange(1, 10)
        self.repeats_override.setValue(1)
        self.repeats_override.setPrefix("Repeats: ")
        overrides_form.addWidget(self.repeats_override)

        settings_layout.addWidget(overrides_group)
        settings_layout.addStretch()

        controls_layout.addLayout(settings_layout, 1)

        # Actions
        actions_layout = QVBoxLayout()
        self.start_btn = QPushButton("🚀 Run Experiment")
        self.start_btn.clicked.connect(self._start_experiment)

        self.stop_btn = QPushButton("🛑 Stop")
        self.stop_btn.clicked.connect(self._stop_experiment)
        self.stop_btn.setEnabled(False)

        self.leaderboard_btn = QPushButton("🏆 Leaderboard")
        self.leaderboard_btn.clicked.connect(self._open_leaderboard)

        actions_layout.addWidget(self.start_btn)
        actions_layout.addWidget(self.stop_btn)
        actions_layout.addWidget(self.leaderboard_btn)
        actions_layout.addStretch()
        controls_layout.addLayout(actions_layout, 1)

        layout.addWidget(controls_group)

        # --- Visualization Area ---
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Radar View
        self.radar_view = RadarView()
        self.radar_view.request_training.connect(self._on_radar_train_request)
        splitter.addWidget(self.radar_view)

        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            ["Trial", "Task", "Model", "Accuracy", "Loss", "Params"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        splitter.addWidget(self.results_table)

        # Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        splitter.addWidget(self.log_output)

        splitter.setSizes([300, 200, 100])
        layout.addWidget(splitter)

        self.worker = None

    def _start_experiment(self):
        # 1. Gather Selections
        selected_models = []
        for i in range(self.algo_list.count()):
            item = self.algo_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_models.append(item.text())

        selected_tasks = []
        for i in range(self.task_list.count()):
            item = self.task_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_tasks.append(item.data(Qt.ItemDataRole.UserRole))

        if not selected_models:
            QMessageBox.warning(
                self, "No Models", "Please select at least one algorithm."
            )
            return
        if not selected_tasks:
            QMessageBox.warning(self, "No Tasks", "Please select at least one task.")
            return

        # 2. Get Config
        tier_str = self.tier_combo.currentText()
        if "Smoke" in tier_str:
            tier = PatientLevel.SMOKE
        elif "Shallow" in tier_str:
            tier = PatientLevel.SHALLOW
        elif "Standard" in tier_str:
            tier = PatientLevel.STANDARD
        elif "Deep" in tier_str:
            tier = PatientLevel.DEEP
        else:
            tier = PatientLevel.SHALLOW

        overrides = {
            "epochs": self.epoch_override.value(),
            "trials": self.trials_override.value(),
            "repeats": self.repeats_override.value(),
        }

        # 3. UI State
        self.log_output.append(
            f"Starting Experiment: {len(selected_models)} Models x {len(selected_tasks)} Tasks"
        )
        self.log_output.append(
            f"Tier: {tier.name} | Repeats: {overrides['repeats']} | Overrides: {overrides}"
        )
        self.results_table.setRowCount(0)
        # self.radar_view.clear() # RadarView doesn't have clear(), but we can add logic if needed. Or just leave history.
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 4. Start Worker
        self.worker = ExperimentWorker(selected_tasks, selected_models, tier, overrides)
        self.worker.trial_finished.connect(self._on_trial_finished)
        self.worker.search_started.connect(
            lambda m, t: self.log_output.append(f"🔍 Running {m} on {t}...")
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _stop_experiment(self):
        if self.worker:
            self.worker.stop()
            self.worker.quit()
        self.log_output.append("Experiment stopped.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_finished(self):
        self.log_output.append("Experiment completed.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Generate Report
        try:
            from bioplausible.analysis.reporting import \
                generate_experiment_report

            tier_str = (
                self.tier_combo.currentText().split()[0].lower()
            )  # e.g. "Smoke" -> "smoke"
            report_path = f"experiment_report_{tier_str}.md"
            report_content = generate_experiment_report(
                "bioplausible.db", tier_str, report_path
            )

            self.log_output.append(f"📄 Report saved: {report_path}")

            # Show Report Dialog
            from PyQt6.QtWidgets import (QDialog, QPushButton, QTextEdit,
                                         QVBoxLayout)

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Experiment Report: {tier_str.title()}")
            dialog.resize(800, 600)
            layout = QVBoxLayout(dialog)

            text_edit = QTextEdit()
            text_edit.setMarkdown(report_content)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

        except Exception as e:
            self.log_output.append(f"Failed to generate report: {e}")
            print(f"Report generation error: {e}")

    def _on_trial_finished(self, result):
        params = result.get("params", {})
        acc = result.get("accuracy", 0.0)
        loss = result.get("loss", float("inf"))

        # 1. Rich Log Output
        model = result["model"]
        task = result["task"]

        # Icon based on task
        icon = "🧪"
        if "vision" in task.lower():
            icon = "👁️"
        elif "lm" in task.lower():
            icon = "📝"
        elif "rl" in task.lower():
            icon = "🎮"

        # Color based on accuracy
        color = "#e2e8f0"  # Default white
        if acc > 0.85:
            color = "#4ade80"  # Green
        elif acc > 0.70:
            color = "#60a5fa"  # Blue
        elif acc < 0.50:
            color = "#f87171"  # Red

        html_msg = (
            f"<div style='margin-bottom: 2px;'>"
            f"<span style='font-size: 14px;'>{icon} <b>{model}</b> on <i>{task}</i></span><br>"
            f"<span style='color: {color}; font-weight: bold;'>⟶ Accuracy: {acc:.2%}</span> "
            f"<span style='color: #94a3b8;'> (Loss: {loss:.4f})</span>"
            f"</div>"
        )
        self.log_output.append(html_msg)

        # 2. Radar Update
        self.radar_view.add_result(result)

        # 3. Table Update
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        # Add items with centering
        def item(txt):
            it = QTableWidgetItem(str(txt))
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it

        self.results_table.setItem(row, 0, item(result.get("trial_number")))
        self.results_table.setItem(row, 1, item(task))
        self.results_table.setItem(row, 2, item(model))
        self.results_table.setItem(row, 3, item(f"{acc:.4f}"))
        self.results_table.setItem(row, 4, item(f"{loss:.4f}"))

        # Params string
        p_str = ", ".join(
            [f"{k}={v}" for k, v in params.items() if k not in ["epochs", "batch_size"]]
        )
        self.results_table.setItem(row, 5, QTableWidgetItem(p_str))

        # Scroll to bottom
        self.results_table.scrollToBottom()
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def _on_radar_train_request(self, result):
        """Called when user clicks 'Train This' in Radar Overlay."""
        config = {
            "task": result.get("task", "vision"),
            "dataset": "default",
            "model": result.get("model", "Unknown"),
            "hyperparams": result.get("params", {}),
        }
        self.transfer_config.emit(config)

    def _open_leaderboard(self):
        """Open the Leaderboard Window."""
        if self.leaderboard_window is None:
            self.leaderboard_window = LeaderboardWindow("bioplausible.db")
        self.leaderboard_window.show()
        self.leaderboard_window.raise_()
        self.leaderboard_window.activateWindow()
