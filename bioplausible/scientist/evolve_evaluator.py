#!/usr/bin/env python3
"""
Bridge between ASI-Evolve's code generation and AutoScientist's evaluation.
This script is executed by the wrapper script run by ASI-Evolve's Engineer.
It loads a candidate model proposed by the AI and evaluates it using AutoScientist's rigorous pipeline.
"""
import argparse
import importlib.util
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path

from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.models.registry import register_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evolve_evaluator")

def load_candidate_model(code_path: str):
    """Dynamically load the candidate code from the given file."""
    spec = importlib.util.spec_from_file_location("candidate_model", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {code_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # We expect the AI to define a class that inherits from bioplausible.models.BioModel
    # and provide a function `get_model_class()` or we scan for it.
    if hasattr(mod, "get_model_class"):
        return mod.get_model_class()

    # Fallback: scan for any class inheriting from torch.nn.Module or BioModel
    import torch
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj.__module__ == "candidate_model":
            return obj

    raise ValueError("No PyTorch model class found in candidate code.")

def evaluate_candidate(code_path: str, task: str = "digits", model_name: str = "EvolvedModel"):
    """Evaluate the generated model and return metrics."""
    try:
        # Load the model class from the file
        model_cls = load_candidate_model(code_path)

        # Register it dynamically so create_model can find it
        register_model(model_name)(model_cls)

        # We need a basic configuration for the model.
        # For simplicity in evaluation, we use a fixed smoke-test style configuration.
        config = {
            "batch_size": 32,
            "epochs": 1,
            "hidden_dim": 64,
            "optimizer": "adam",
            "lr": 0.01,
            "early_stopping_patience": 3,
        }

        # Use AutoScientist's execution framework
        metrics = run_single_trial_task(
            task=task,
            model_name=model_name,
            config=config,
            storage_path=":memory:",  # Don't clutter the main DB with ephemeral candidates
            quick_mode=True
        )

        if metrics is None:
            raise RuntimeError("Evaluation failed to produce metrics.")

        return {
            "eval_score": metrics.get("accuracy", 0.0), # primary metric for ASI-Evolve, named eval_score
            "metrics": metrics,
            "success": True
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        traceback.print_exc()
        return {
            "eval_score": 0.0,
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("code_path", help="Path to the python file to evaluate")
    parser.add_argument("--task", default="digits", help="Dataset task to evaluate on")

    args = parser.parse_args()

    result = evaluate_candidate(args.code_path, task=args.task)

    # Write to results.json in the current working directory, which is the experiment_dir
    with open("results.json", "w") as f:
        json.dump(result, f)
