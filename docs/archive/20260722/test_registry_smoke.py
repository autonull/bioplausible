import logging
import unittest
import warnings

import torch

from bioplausible.execution.strategy import ExecutionStrategy
from bioplausible.hyperopt.experiment import run_single_trial_task

# Suppress warnings and logging for cleaner output
logging.getLogger("AutoScientist").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class TestRegistrySmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n>>> Starting comprehensive Registry Smoke Tests (Quick Mode)...")


def make_test(model_spec, task_name):
    def test(self):
        # print(f"Testing {model_spec.name} on {task_name}")

        config = {
            "tier": "smoke",
            "epochs": 1,
            "batch_size": 32,
            "lr": model_spec.default_lr,
            "hidden_dim": 16,  # Small for speed
            "num_layers": 2,
            "save_artifacts": False,
            "job_id": f"test_{task_name}_{model_spec.model_type}",
        }

        try:
            metrics = run_single_trial_task(
                task=task_name,
                model_name=model_spec.name,
                config=config,
                storage_path=None,
                quick_mode=True,
                verbose=False,
            )

            self.assertIsNotNone(
                metrics, f"Model {model_spec.name} returned None metrics on {task_name}"
            )
            self.assertFalse(torch.isnan(torch.tensor(metrics["loss"])), "Loss is NaN")

        except Exception as e:
            if "kmnist" in task_name and "download" in str(e).lower():
                self.skipTest("KMNIST download failed, skipping test.")
            else:
                self.fail(
                    f"Model {model_spec.name} on {task_name} failed with exception: {e}"
                )

    return test


# Generate tests
vision_tasks = ExecutionStrategy.TASK_GROUPS["vision"]
lm_tasks = ExecutionStrategy.TASK_GROUPS["lm"]
rl_tasks = ExecutionStrategy.TASK_GROUPS["rl"]

all_tasks = set(vision_tasks + lm_tasks + rl_tasks)

# We want to test every model on every compatible task
for spec in MODEL_REGISTRY:
    compatible_tasks = []
    if spec.task_compat:
        for t in spec.task_compat:
            if t == "vision":
                compatible_tasks.extend(vision_tasks)
            elif t == "lm":
                compatible_tasks.extend(lm_tasks)
            elif t == "rl":
                compatible_tasks.extend(rl_tasks)
            elif t in all_tasks:
                compatible_tasks.append(t)
    else:
        # If None, compatible with all applicable?
        # We assume vision + rl based on most models
        compatible_tasks = vision_tasks + rl_tasks

    # Dedup
    compatible_tasks = sorted(list(set(compatible_tasks)))

    for task_name in compatible_tasks:
        # Create a test method name
        safe_model_name = (
            spec.model_type
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        safe_task_name = task_name.replace(" ", "_").replace("-", "_")
        test_name = f"test_{safe_model_name}_on_{safe_task_name}"

        setattr(TestRegistrySmoke, test_name, make_test(spec, task_name))

if __name__ == "__main__":
    unittest.main()
