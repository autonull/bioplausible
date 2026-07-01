import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional

from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.scientist.task import ExperimentTask


def _worker_process_task(args: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Worker function to process a single task.
    Args are passed as a dict to be picklable and extensible.
    """
    # Configure worker logger
    worker_id = os.getpid()
    logging.basicConfig(
        format=f"%(asctime)s [Worker-{worker_id}] %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(f"Worker-{worker_id}")

    # Extract args
    # Note: args keys must match what is packed in run_batch
    task_data = args.get("task_obj") or args.get("task")
    config = args.get("config", {})
    db_path = args["db_path"]

    try:
        task: ExperimentTask = task_data

        # Ensure config has minimal fields if not already populated
        if not config and task.fixed_config:
            config = task.fixed_config.copy()

        config["tier"] = task.tier.value
        config["task"] = task.task_name
        config["model"] = task.model_name

        logger.info(
            f"Starting trial for {task.model_name} on {task.task_name}"
            f" (Tier: {task.tier.name})"
        )

        metrics = run_single_trial_task(
            task=task.task_name,
            model_name=task.model_name,
            config=config,
            storage_path=db_path,
            quick_mode=(task.tier.name == "SMOKE"),
            verbose=False,
        )

        if metrics:
            logger.info(f"Trial completed. Acc: {metrics.get('accuracy', 0.0):.2%}")
        else:
            logger.warning("Trial returned no metrics (Failed).")

        return metrics

    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        return None


class ParallelTrialRunner:
    """
    Executes a batch of experiments in parallel using multiprocessing.
    """

    def __init__(self, num_workers: int, db_path: str):
        self.num_workers = num_workers
        self.db_path = db_path

    def run_batch(
        self, tasks: List[ExperimentTask], configs: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, float]]]:
        """
        Run a batch of tasks.

        Args:
            tasks: List of ExperimentTask objects.
            configs: List of resolved configuration dictionaries corresponding to
                tasks. (Must be resolved in main process to avoid DB write
                contention on 'ask')

        Returns:
            List of result metrics (or None for failures).
        """
        if not tasks:
            return []

        # Prepare arguments for workers
        worker_args = []
        for task, config in zip(tasks, configs):
            args = {
                "task_obj": task,  # Pass for metadata
                "config": config,
                "db_path": self.db_path,
            }
            worker_args.append(args)

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._wrapped_worker, worker_args)

        return results

    @staticmethod
    def _wrapped_worker(args):
        """
        Static wrapper to unpack args and call the logic.
        Redefined here to ensure visibility or call the global one.
        """
        task = args["task_obj"]
        config = args["config"]
        db_path = args["db_path"]

        return run_single_trial_task(
            task=task.task_name,
            model_name=task.model_name,
            config=config,
            storage_path=db_path,
            quick_mode=(task.tier.name == "SMOKE"),
            verbose=False,
        )
