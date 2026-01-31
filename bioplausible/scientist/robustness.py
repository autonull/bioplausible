"""
Robustness Testing Suite for AutoScientist.

Headless wrapper for the existing RobustnessTool logic.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional

from bioplausible.models.registry import get_model_spec, MODEL_REGISTRY
# Note: bioplausible_ui is separate from bioplausible core in some contexts
# We import only if available, or mock if running in headless core environment
try:
    from bioplausible_ui.lab.tools.robustness import RobustnessWorker
except ImportError:
    RobustnessWorker = None

logger = logging.getLogger("Robustness")

def run_robustness_check(model_name: str, task: str, config: Dict[str, Any], weights_path: str = None) -> float:
    """
    Runs a suite of robustness tests (Noise, FGSM, Dropout) on a trained model.
    Returns a unified 'Robustness Score' (0.0 - 1.0).
    """
    # Note: In a real implementation, we would load the weights from 'weights_path'.
    # Since run_single_trial_task doesn't strictly save weights to disk in a known location
    # unless configured, we might be simulating this or needing to load from the 'trial artifact'.
    # For now, we instantiate a FRESH model with the same config to test *architectural* robustness
    # or assume weights are passed.

    # Limitation: Robustness usually requires a TRAINED model.
    # AutoScientist currently runs trials but doesn't persist checkpoints for every trial by default.
    # However, for the "Robustness Tier", we arguably want to TRAIN then TEST.
    # Or, we assume the run_single_trial_task returns a model? It doesn't.

    # Strategy: We will treat this as a "Train & Stress Test" loop.
    # We train quickly (or load if we had checkpointing), then attack.

    # To keep it compatible with the existing pipeline which expects a float score,
    # we return a score.

    # Attempt to use real RobustnessWorker
    if RobustnessWorker:
        try:
            # We assume RobustnessWorker has a synchronous run_headless() method
            # If not, we fail gracefully rather than faking it.
            worker = RobustnessWorker(model_name, task, config, weights_path)
            if hasattr(worker, 'run_headless'):
                return worker.run_headless()
            else:
                logger.error("RobustnessWorker exists but lacks run_headless() method.")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to run RobustnessWorker: {e}")
            return 0.0

    logger.error("RobustnessWorker not available in this environment.")
    return 0.0
