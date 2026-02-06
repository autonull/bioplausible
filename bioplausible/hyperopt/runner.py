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

from bioplausible.scientist.failure_tracker import FailureTracker, FailureRecord
from datetime import datetime
from bioplausible.hyperopt.experiment import TrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage

# ... imports ...

def run_single_trial_task(
    task: str,
    model_name: str,
    config: Dict[str, Any],
    storage_path: Optional[str] = None,
    quick_mode: bool = True,
    verbose: bool = False,
) -> Optional[Dict[str, float]]:
    # ... docstring ...
    temp_dir = None

    if storage_path is None:
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "worker_temp.db"
    else:
        db_path = Path(storage_path)

    storage = None
    failure_tracker = FailureTracker(str(db_path))
    
    try:
        storage = HyperoptStorage(str(db_path))

        # Create trial entry
        trial_id = storage.create_trial(model_name, config)

        # Create runner
        runner = TrialRunner(
            storage=storage, 
            device="auto", 
            task=task, 
            quick_mode=quick_mode,
            checkpoint_db_path=str(db_path)
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
                "trial_id": trial_id,  # DB PK
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
            
            # Log logical failure (e.g. NaN, divergence)
            failure_tracker.log_failure(FailureRecord(
                timestamp=datetime.now().isoformat(),
                model_name=model_name,
                task_name=task,
                tier=config.get("tier", "unknown"),
                trial_id=trial_id,
                failure_type="training_failed",
                failure_epoch=config.get("epochs", 0), # approx
                failure_batch=None,
                config=config,
                last_metrics={}
            ))
            return None

    except Exception as e:
        print(f"Execution Error: {e}")
        if verbose:
            traceback.print_exc()
            
        # Log exception failure
        failure_tracker.log_failure(FailureRecord(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            task_name=task,
            tier=config.get("tier", "unknown"),
            trial_id=config.get("job_id"), # might be None
            failure_type="exception",
            failure_epoch=None,
            failure_batch=None,
            config=config,
            last_metrics={},
            stack_trace=traceback.format_exc()
        ))
        return None
    finally:
        if storage:
            storage.close()
            
        # Cleanup
        if verbose:
            print("Cleaning up trial resources...")
        
        # Explicitly break references
        if 'runner' in locals():
            del runner
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if verbose:
            print("Cleanup complete.")
            
        if temp_dir:
            shutil.rmtree(temp_dir)
