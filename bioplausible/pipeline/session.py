import dataclasses
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn as nn

from bioplausible.hyperopt.tasks import BaseTask, create_task
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec
from bioplausible.pipeline.config import TrainingConfig
from bioplausible.pipeline.events import (CompletedEvent, Event, PausedEvent,
                                          ProgressEvent)
from bioplausible.pipeline.results import ResultsManager
from bioplausible.training.base import BaseTrainer


class SessionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class TrainingSession:
    """Headless training orchestrator (no UI dependencies)."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = SessionState.IDLE
        self.task: Optional[BaseTask] = None
        self.model: Optional[nn.Module] = None
        self.trainer: Optional[BaseTrainer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Unique Run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}_{str(uuid.uuid4())[:8]}"

    def start(self) -> Generator[Event, None, None]:
        """Start training, yield events."""
        self.state = SessionState.RUNNING

        try:
            # 1. Setup Task
            self.task = create_task(self.config.dataset, device=self.device)

            self.task.setup()

            # 2. Setup Model
            spec = get_model_spec(self.config.model)

            self.model = create_model(
                spec=spec,
                input_dim=self.task.input_dim,
                output_dim=self.task.output_dim,
                device=self.device,
                task_type=self.task.task_type,
                **self.config.hyperparams,
            )

            # 3. Create Trainer
            self.trainer = self.task.create_trainer(
                self.model,
                batches_per_epoch=100,
                eval_batches=20,
                epochs=self.config.epochs,
                lr=self.config.learning_rate,
                **self.config.hyperparams,
            )

            metrics = {}
            history = []

            # 4. Training Loop
            for epoch in range(self.config.epochs):
                if self.state == SessionState.STOPPED:
                    break

                while self.state == SessionState.PAUSED:
                    yield PausedEvent()
                    import time

                    time.sleep(0.1)

                if self.state == SessionState.STOPPED:
                    break

                metrics = self.trainer.train_epoch()
                metrics["epoch"] = epoch
                history.append(metrics)

                yield ProgressEvent(epoch=epoch, metrics=metrics)

            if self.state != SessionState.STOPPED:
                self.state = SessionState.COMPLETED

                # Save results
                # Save full history + final metrics
                final_results = {"final_metrics": metrics, "history": history}
                self._save_results(final_results)

                yield CompletedEvent(final_metrics=metrics)

        except Exception as e:
            self.state = SessionState.ERROR
            import traceback

            traceback.print_exc()
            raise e

    def _save_results(self, metrics):
        """Save results and weights using ResultsManager."""
        try:
            mgr = ResultsManager()
            # Convert config dataclass to dict
            config_dict = dataclasses.asdict(self.config)
            mgr.save_run(self.run_id, config_dict, metrics)

            # Save weights
            if self.model:
                run_dir = os.path.join(mgr.BASE_DIR, self.run_id)
                torch.save(self.model.state_dict(), os.path.join(run_dir, "model.pt"))

        except Exception as e:
            print(f"Failed to save results: {e}")

    def pause(self):
        if self.state == SessionState.RUNNING:
            self.state = SessionState.PAUSED

    def resume(self):
        if self.state == SessionState.PAUSED:
            self.state = SessionState.RUNNING

    def stop(self):
        self.state = SessionState.STOPPED
