"""
Hyperopt Trial Execution Helper.
"""

import contextlib
import io
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from bioplausible.hyperopt.experiment import TrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage


def run_single_trial_task(
    task: str,
    model_name: str,
    config: Dict[str, Any],
    storage_path: Optional[str] = None,
    quick_mode: bool = True,
    verbose: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Run a single trial and return metrics.

    Args:
        task: Task name (e.g. 'mnist')
        model_name: Model architecture name
        config: Hyperparameter dictionary
        storage_path: Path to SQLite DB. If None, uses a temporary DB.
        quick_mode: If True, uses fewer data/iterations (default True).
        verbose: If True, show training output
    """
    temp_dir = None

    if storage_path is None:
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "worker_temp.db"
    else:
        db_path = Path(storage_path)

    storage = None
    try:
        storage = HyperoptStorage(str(db_path))

        # Create trial entry
        trial_id = storage.create_trial(model_name, config)

        # Create runner
        runner = TrialRunner(
            storage=storage, device="auto", task=task, quick_mode=quick_mode
        )

        # Override epochs if present
        if "epochs" in config:
            runner.epochs = int(config["epochs"])

        # Run training
        if verbose:
            success = runner.run_trial(trial_id)
        else:
            # Suppress output but keep stderr for errors
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                success = runner.run_trial(trial_id)

        if success:
            trial = storage.get_trial(trial_id)
            metrics = {
                "trial_id": trial_id, # DB PK
                "accuracy": trial.accuracy,
                "loss": trial.final_loss,
                "perplexity": trial.perplexity,
                "time": trial.iteration_time,
                "param_count": trial.param_count,  # In millions
            }
            return metrics
        else:
            if verbose:
                print(f"Trial {trial_id} returned success=False")
            return None

    except Exception as e:
        print(f"Execution Error: {e}")
        if verbose:
            traceback.print_exc()
        return None
    finally:
        if storage:
            storage.close()
        if temp_dir:
            shutil.rmtree(temp_dir)
