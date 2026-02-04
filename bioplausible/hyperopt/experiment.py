"""
Experiment Runner

Executes hyperparameter optimization trials and collects metrics.
"""

import numpy as np
import torch

from bioplausible.config import GLOBAL_CONFIG
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.hyperopt.tasks import create_task
from bioplausible.models.factory import create_model, load_weights
from bioplausible.models.registry import get_model_spec
from bioplausible.scientist.archiver import ExperimentArchiver
from bioplausible.scientist.monitoring import InterferenceMonitor
from bioplausible.tracking import ExperimentTracker


class TrialRunner:
    """Runs individual hyperparameter optimization trials."""

    def __init__(
        self,
        storage: HyperoptStorage = None,
        device: str = "auto",
        task: str = "shakespeare",
        quick_mode: bool = True,
    ):
        self.storage = storage or HyperoptStorage()
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.task_name = task
        self.quick_mode = quick_mode
        self.epochs = GLOBAL_CONFIG.epochs

        # Initialize Task abstraction
        self.task_obj = create_task(task, self.device, quick_mode)
        self.task_obj.setup()

        self.input_dim = self.task_obj.input_dim
        self.output_dim = self.task_obj.output_dim

    def _load_transfer_weights(self, transfer_from: int, model: torch.nn.Module, config: dict):
        """Helper to find and load weights from a previous trial."""
        from pathlib import Path
        import tempfile
        import zipfile

        artifact_dir = Path("artifacts")
        if not artifact_dir.exists():
            print("⚠️ Warning: Artifacts directory not found.")
            return

        try:
            # Find matching zip or dir
            for item in artifact_dir.iterdir():
                # Expected format: trial_{id}_{model_name}
                if item.name.startswith(f"trial_{transfer_from}_"):
                    if item.is_dir():
                        found_path = item / "model.pt"
                        if found_path.exists():
                            load_weights(
                                model,
                                str(found_path),
                                device=self.device,
                                strict=False,
                                freeze_layers=config.get("freeze_layers", False),
                            )
                            return
                    elif item.suffix == ".zip":
                        # Unzip to temp context
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            with zipfile.ZipFile(item, "r") as zip_ref:
                                zip_ref.extract("model.pt", temp_path)
                                found_path = temp_path / "model.pt"
                                if found_path.exists():
                                    load_weights(
                                        model,
                                        str(found_path),
                                        device=self.device,
                                        strict=False,
                                        freeze_layers=config.get(
                                            "freeze_layers", False
                                        ),
                                    )
                                    return
                    break

            print(f"⚠️ Warning: Could not find artifact for trial {transfer_from}")

        except Exception as e:
            print(f"❌ Error loading transfer weights: {e}")

    def run_trial(self, trial_id: int, pruning_callback=None) -> bool:
        """Run a single trial and record results."""
        # Get trial
        trial = self.storage.get_trial(trial_id)
        if not trial:
            print(f"Trial {trial_id} not found")
            return False

        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: {trial.model_name}")
        print(f"Config: {trial.config}")
        print(f"{'='*60}\n")

        self.storage.update_trial(trial_id, status="running")

        tracker = ExperimentTracker(
            project="bioplausible",
            name=f"trial_{trial_id}_{trial.model_name}",
            config=trial.config,
        )

        try:
            # Create model using factory directly
            spec = get_model_spec(trial.model_name)
            config = trial.config
            hidden_dim = config.get("hidden_dim", 128)
            num_layers = config.get("num_layers", 4)

            model = create_model(
                spec=spec,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=self.device,
                task_type=self.task_obj.task_type,
            )

            # Transfer Learning Logic
            transfer_from = config.get("transfer_from")
            if transfer_from:
                print(
                    f"🔄 Initializing Transfer Learning from Trial {transfer_from}..."
                )
                self._load_transfer_weights(transfer_from, model, config)

            # Apply hyperparameters
            lr = config.get("lr", spec.default_lr)
            beta = config.get("beta")  # None if not in config
            steps = config.get("steps")

            # Additional params (for Hebbian, etc.)
            # If the model or trainer needs these, we should pass them.
            # Trainer takes **kwargs.

            # Create Trainer via Task
            # We pass all config items as kwargs to the trainer
            trainer_kwargs = config.copy()
            # Remove keys that are passed explicitly to avoid conflicts
            for key in [
                "lr",
                "steps",
                "batches_per_epoch",
                "eval_batches",
                "model",
                "task",
                "tier",
                "job_id",
                "fold",
                "is_verification",
                "verified_trial_id",
            ]:
                if key in trainer_kwargs:
                    del trainer_kwargs[key]

            trainer = self.task_obj.create_trainer(
                model,
                lr=lr,
                steps=steps if steps else 20,
                batches_per_epoch=200 if not GLOBAL_CONFIG.quick_mode else 100,
                eval_batches=50 if not GLOBAL_CONFIG.quick_mode else 20,
                tracker=tracker,
                **trainer_kwargs,
            )

            # Manually set beta on model if provided in config
            # Models that use beta will have it in their config or as an attribute

            if beta is not None and hasattr(model, "config"):
                # BioModel based
                model.config.beta = beta
            if beta is not None and hasattr(model, "beta"):
                model.beta = beta

            # Monitoring
            monitor = InterferenceMonitor(threshold_cpu=20.0, sustain_duration=5.0)
            monitor.start()

            # Training Loop
            epoch_times = []

            for epoch in range(self.epochs):
                metrics = trainer.train_epoch()

                # Log
                self.storage.log_epoch(
                    trial_id,
                    epoch,
                    metrics["loss"],
                    metrics.get("accuracy", 0.0),
                    metrics.get("perplexity", 0.0),
                    metrics["time"],
                )

                epoch_times.append(metrics["time"])

                print(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"acc={metrics.get('accuracy', 0.0):.4f}, "
                    f"ppl={metrics.get('perplexity', 0.0):.2f}, "
                    f"time={metrics['time']:.1f}s"
                )

                # Pruning
                if pruning_callback:
                    if pruning_callback(trial_id, epoch + 1, metrics):
                        print(f"✂️ Trial {trial_id} PRUNED at epoch {epoch+1}")
                        self.storage.update_trial(trial_id, status="pruned")
                        monitor.stop()
                        return False

            monitor.stop()

            if monitor.check_interference():
                print("⚠️ INTERFERENCE DETECTED: Rejecting trial results.")
                self.storage.update_trial(trial_id, status="failed")
                return False

            # Final Stats
            avg_iter_time = np.mean(epoch_times) / (
                trainer.batches_per_epoch
                if hasattr(trainer, "batches_per_epoch")
                else 1
            )  # Fallback for RL
            if hasattr(trainer, "episodes_per_epoch"):  # RL
                avg_iter_time = np.mean(epoch_times) / trainer.episodes_per_epoch

            param_count = sum(p.numel() for p in model.parameters())
            param_count_millions = param_count / 1e6

            self.storage.update_trial(
                trial_id,
                status="completed",
                epochs_completed=self.epochs,
                final_loss=metrics["loss"],
                accuracy=metrics.get("accuracy", 0.0),
                perplexity=metrics.get("perplexity", 0.0),
                iteration_time=avg_iter_time,
                param_count=param_count_millions,
            )

            # Check if we should archive
            if config.get("save_artifacts"):
                print("📦 Archiving artifacts...")
                archiver = ExperimentArchiver()
                archiver.archive_trial(
                    trial_id=trial_id, model=model, config=config, metrics=metrics
                )

            print(f"\n✅ Trial {trial_id} completed successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Trial {trial_id} failed: {e}")
            import traceback

            traceback.print_exc()
            self.storage.update_trial(trial_id, status="failed")
            return False
        finally:
            if "monitor" in locals():
                monitor.stop()
            tracker.finish()
