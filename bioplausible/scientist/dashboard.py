"""
TUI Dashboard for AutoScientist.

This module provides a rich Terminal User Interface (TUI) for monitoring
the progress of the autonomous discovery process, including real-time
trial updates, resource usage, and historical performance logs.
"""

import datetime
import shutil
from typing import Any, Dict, List, Optional

import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

try:
    import torch
except ImportError:
    torch = None


class Dashboard:
    """
    TUI Dashboard for AutoScientist.

    Attributes:
        console (Console): The rich console instance.
        layout (Layout): The main layout structure.
        progress (Progress): Progress bar instance.
        epoch_task (TaskID): ID for the epoch progress task.
        status_log (List[str]): Log of recent status messages.
        recent_trials (List[Dict[str, Any]]): History of recent trials.
        current_trial_info (Dict[str, Any]): Details of the currently running trial.
        best_model (Optional[Dict[str, Any]]):
            Information about the best performing model.
        insight_text (str): Current scientific insight message.
        live (Live): The live display manager.
    """

    def __init__(self) -> None:
        """Initialize the dashboard components."""
        self.console = Console(stderr=True)
        self.layout = Layout()
        self._init_layout()

        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.epoch_task = self.progress.add_task("Epoch", total=100)

        self.status_log: List[str] = []
        self.recent_trials: List[Dict[str, Any]] = []
        self.current_trial_info: Dict[str, Any] = {}
        self.best_model: Optional[Dict[str, Any]] = None
        self.insight_text: str = "Initializing analysis modules..."

        self.system_status: str = "Initializing"
        self.system_status_style: str = "dim"

        self.live = Live(self.layout, refresh_per_second=4, console=self.console)

    def _init_layout(self) -> None:
        """Set up the TUI layout grid."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )
        self.layout["left"].split(
            Layout(name="current_trial", size=12),
            Layout(name="history", ratio=1),
        )
        self.layout["right"].split(
            Layout(name="stats", size=10),
            Layout(name="log", ratio=1),
        )
        self.layout["stats"].split_row(
            Layout(name="best_model", ratio=1),
            Layout(name="system", ratio=1),
        )

    def start(self) -> None:
        """Start the live dashboard display."""
        self.live.start()

    def stop(self) -> None:
        """Stop the live dashboard display."""
        self.live.stop()

    def update(self) -> None:
        """Refresh all dashboard components with current data."""
        # Update Header
        self.layout["header"].update(
            Panel(
                Text(
                    "AutoScientist: Bio-Plausible Algorithm Discovery",
                    justify="center",
                    style="bold green",
                ),
                style="green",
            )
        )

        # Update Current Trial
        trial_text = Text()
        if self.current_trial_info:
            info = self.current_trial_info
            trial_text.append(f"Trial ID: {info.get('id', 'N/A')}  ", style="bold cyan")
            trial_text.append(
                f"Model: {info.get('model', 'N/A')}  ", style="bold yellow"
            )
            trial_text.append(
                f"Task: {info.get('task', 'N/A')}\n", style="bold magenta"
            )
            trial_text.append(f"Tier: {info.get('tier', 'N/A')} | ", style="white")

            # Metrics
            if "metrics" in info:
                m = info["metrics"]
                loss = m.get("loss", 0.0)
                acc = m.get("accuracy", 0.0)
                trial_text.append(
                    f"Loss: {loss:.4f} | Acc: {acc:.2%}",
                    style="green",
                )

            trial_text.append(
                f"\nParams: {str(info.get('params', {}))[:80]}...", style="dim"
            )

        self.layout["current_trial"].update(
            Panel(self.progress, title="🔬 Current Experiment", subtitle=trial_text)
        )

        # Update History Table
        table = Table(expand=True, box=None)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Model", style="yellow")
        table.add_column("Task", style="magenta")
        table.add_column("Acc", justify="right", style="bold green")
        table.add_column("Rob.", justify="right", style="blue")
        table.add_column("Status", justify="center")

        for t in reversed(self.recent_trials[-12:]):
            status_style = "green" if t["status"] == "completed" else "red"

            # Robustness Score
            rob_val = "-"
            if t.get("metrics") and "robustness_score" in t["metrics"]:
                rob = t["metrics"]["robustness_score"]
                rob_val = f"{rob:.2%}"

            table.add_row(
                str(t["id"]),
                t["model"],
                t["task"],
                f"{t['accuracy']:.2%}",
                rob_val,
                f"[{status_style}]{t['status']}[/]",
            )

        self.layout["history"].update(Panel(table, title="🧪 Experiment History"))

        # Best Model
        best_text = Text()
        if self.best_model:
            best_text.append(f"{self.best_model['model']}\n", style="bold yellow")
            best_text.append(f"{self.best_model['accuracy']:.2%} ", style="bold green")
            best_text.append(f"on {self.best_model['task']}", style="magenta")
        else:
            best_text.append("No data yet...", style="dim")
        self.layout["best_model"].update(Panel(best_text, title="🏆 SOTA Model"))

        # Update System
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        sys_text = Text()
        sys_text.append("Status: ", style="bold")
        sys_text.append(f"{self.system_status}\n", style=self.system_status_style)
        sys_text.append(f"CPU: {cpu}%\n")
        sys_text.append(f"RAM: {mem}%\n")

        # GPU
        if torch and torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                used_ratio = (total - free) / total * 100.0
                sys_text.append(f"GPU: {used_ratio:.1f}%\n")
            except Exception:
                pass

        # Disk
        total_disk, used_disk, free_disk = shutil.disk_usage(".")
        disk_percent = (used_disk / total_disk) * 100.0
        sys_text.append(f"DSK: {disk_percent:.1f}%\n")

        self.layout["system"].update(Panel(sys_text, title="💻 System"))

        # Update Log & Insight
        log_text = Text()
        for msg in self.status_log[-15:]:
            log_text.append(msg + "\n")

        # Footer with insight
        self.layout["footer"].update(
            Panel(
                Text(f"🧠 Insight: {self.insight_text}", style="italic cyan"),
                style="blue",
            )
        )

        self.layout["log"].update(Panel(log_text, title="📜 Event Log"))

    def log(self, message: str, style: str = "") -> None:
        """
        Add a message to the dashboard log.

        Args:
            message (str): The message text.
            style (str): Optional rich style tag (e.g., 'bold red').
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        if style:
            formatted = f"[{style}]{formatted}[/]"
        self.status_log.append(formatted)
        self.update()

    def set_trial(
        self,
        trial_id: str,
        model: str,
        task: str,
        tier: str,
        params: Dict[str, Any],
    ) -> None:
        """
        Set the details for the currently running trial.

        Args:
            trial_id: Unique identifier for the trial.
            model: Name of the model being trained.
            task: Name of the task/dataset.
            tier: Experiment tier (e.g., SMOKE, STANDARD).
            params: Dictionary of hyperparameters.
        """
        self.current_trial_info = {
            "id": trial_id,
            "model": model,
            "task": task,
            "tier": tier,
            "params": params,
        }
        self.progress.reset(self.epoch_task)
        self.update()

    def update_progress(
        self, epoch: int, total_epochs: int, metrics: Dict[str, float]
    ) -> None:
        """
        Update the progress bar and metrics for the current trial.

        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs planned.
            metrics: Current training metrics (loss, accuracy).
        """
        self.progress.update(self.epoch_task, completed=epoch, total=total_epochs)
        self.current_trial_info["metrics"] = metrics
        self.update()

    def complete_trial(self, status: str, metrics: Dict[str, Any]) -> None:
        """
        Mark the current trial as completed and update history.

        Args:
            status: Completion status ('completed' or 'failed').
            metrics: Dictionary of result metrics (must contain 'accuracy').
        """
        if self.current_trial_info:
            accuracy = metrics.get("accuracy", 0.0)
            trial_data = {
                "id": self.current_trial_info["id"],
                "model": self.current_trial_info["model"],
                "task": self.current_trial_info["task"],
                "accuracy": accuracy,
                "metrics": metrics,
                "status": status,
            }
            self.recent_trials.append(trial_data)

            # Update Best Model
            if status == "completed":
                if self.best_model is None or accuracy > self.best_model["accuracy"]:
                    self.best_model = trial_data
                    self.log(
                        f"New SOTA: {accuracy:.2%} ({trial_data['model']})",
                        style="bold yellow",
                    )

            self.current_trial_info = {}
        self.update()

    def set_insight(self, text: str) -> None:
        """
        Update the scientific insight message in the footer.

        Args:
            text: Insight message string.
        """
        self.insight_text = text
        self.update()

    def set_system_status(self, status: str, style: str = "white") -> None:
        """
        Update the system status message.

        Args:
            status: Status message (e.g., 'Active', 'Paused').
            style: Rich style tag (e.g., 'bold green').
        """
        self.system_status = status
        self.system_status_style = style
        self.update()


# Global Instance
DASHBOARD = Dashboard()
