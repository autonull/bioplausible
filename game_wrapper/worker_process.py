import json
import os
import sys

# Modify sys.path to ensure we can import bioplausible
sys.path.append(os.getcwd())

from bioplausible.hyperopt.runner import run_single_trial_task


def run_job(model_name, config):
    print(f"Worker starting: {model_name}")

    # Check if task is specified in config, else default to vision
    task = config.get("task", "vision")

    # Ensure config has minimal required fields if they are missing
    # But usually the game generator provides them.

    # We use the existing robust runner
    try:
        # We need to adapt the config to what run_single_trial_task expects?
        # Actually run_single_trial_task takes (task, model_name, config, ...)
        # It handles dataset loading internally.

        # NOTE: run_single_trial_task logs to storage itself if storage_path is provided!
        # The game main loop reads from "results/hyperopt.db".
        # So we just pass that path.

        metrics = run_single_trial_task(
            task=task,
            model_name=model_name,
            config=config,
            storage_path="results/hyperopt.db",
            quick_mode=True,  # Keep it fast for the game
            verbose=True,
        )

        if metrics:
            print("Worker finished successfully.")
        else:
            print("Worker finished with no metrics (failure?).")

    except Exception as e:
        print(f"Worker failed: {e}")
        # We might want to manually log failure if runner doesn't catch it
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: worker_process.py <model_name> <json_config>")
        sys.exit(1)

    model_name = sys.argv[1]
    config = json.loads(sys.argv[2])

    run_job(model_name, config)
