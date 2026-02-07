import datetime
from typing import Optional, Dict, Any
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text
import psutil
import os

class Dashboard:
    """
    TUI Dashboard for AutoScientist.
    """
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self._init_layout()

        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.epoch_task = self.progress.add_task("Epoch", total=100)

        self.status_log = []
        self.recent_trials = []
        self.current_trial_info = {}

        self.live = Live(self.layout, refresh_per_second=4, console=self.console)

    def _init_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        self.layout["left"].split(
            Layout(name="current_trial", size=10),
            Layout(name="history", ratio=1),
        )
        self.layout["right"].split(
            Layout(name="system", size=8),
            Layout(name="log", ratio=1),
        )

    def start(self):
        self.live.start()

    def stop(self):
        self.live.stop()

    def update(self):
        # Update Header
        self.layout["header"].update(
            Panel(Text("AutoScientist: Bio-Plausible Algorithm Discovery", justify="center", style="bold green"))
        )

        # Update Current Trial
        trial_text = Text()
        if self.current_trial_info:
            info = self.current_trial_info
            trial_text.append(f"Trial ID: {info.get('id', 'N/A')}\n", style="bold cyan")
            trial_text.append(f"Model: {info.get('model', 'N/A')}\n", style="yellow")
            trial_text.append(f"Task: {info.get('task', 'N/A')}\n", style="magenta")
            trial_text.append(f"Tier: {info.get('tier', 'N/A')}\n")
            trial_text.append(f"Params: {info.get('params', {})}\n", style="dim")

        self.layout["current_trial"].update(
            Panel(
                self.progress,
                title=f"Current Trial: {self.current_trial_info.get('model', 'Idle')}",
                subtitle=trial_text.plain
            )
        )

        # Update History Table
        table = Table(title="Recent Trials", expand=True)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Model", style="yellow")
        table.add_column("Task", style="magenta")
        table.add_column("Acc", justify="right", style="green")
        table.add_column("Status", justify="center")

        for t in reversed(self.recent_trials[-10:]):
            status_style = "green" if t['status'] == "completed" else "red"
            table.add_row(
                str(t['id']),
                t['model'],
                t['task'],
                f"{t['accuracy']:.2%}",
                f"[{status_style}]{t['status']}[/]"
            )

        self.layout["history"].update(Panel(table))

        # Update System
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        sys_text = Text()
        sys_text.append(f"CPU: {cpu}%\n")
        sys_text.append(f"RAM: {mem}%\n")
        self.layout["system"].update(Panel(sys_text, title="System Resources"))

        # Update Log
        log_text = Text()
        for msg in self.status_log[-15:]:
            log_text.append(msg + "\n")
        self.layout["log"].update(Panel(log_text, title="Scientist Log"))

    def log(self, message: str, style: str = ""):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        if style:
            formatted = f"[{style}]{formatted}[/]"
        self.status_log.append(formatted)
        self.update()

    def set_trial(self, trial_id, model, task, tier, params):
        self.current_trial_info = {
            "id": trial_id,
            "model": model,
            "task": task,
            "tier": tier,
            "params": params
        }
        self.progress.reset(self.epoch_task)
        self.update()

    def update_progress(self, epoch, total_epochs, metrics):
        self.progress.update(self.epoch_task, completed=epoch, total=total_epochs)
        self.current_trial_info["metrics"] = metrics
        self.update()

    def complete_trial(self, status, accuracy):
        if self.current_trial_info:
            self.recent_trials.append({
                "id": self.current_trial_info["id"],
                "model": self.current_trial_info["model"],
                "task": self.current_trial_info["task"],
                "accuracy": accuracy,
                "status": status
            })
            self.current_trial_info = {}
        self.update()

# Global Instance
DASHBOARD = Dashboard()
