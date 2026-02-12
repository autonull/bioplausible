"""
Task promotion logic.

Defines the criteria and logic for promoting a model to a higher tier of
difficulty or complexity within the experimental curriculum.
"""

from typing import Any, Dict

# Minimum success criteria for each task
PROMOTION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "char_ngram": {"accuracy": 0.95},  # Should be trivial
    "digits": {"accuracy": 0.90},  # Tiny, should be easy
    "usps": {"accuracy": 0.85},
    "kmnist": {"accuracy": 0.80},
    "mnist": {"accuracy": 0.85},  # Good baseline
    "fashion_mnist": {"accuracy": 0.75},
    "svhn": {"accuracy": 0.60},  # Noisier than F-MNIST
    "cifar10": {"accuracy": 0.45},  # Harder
    "cifar100": {"accuracy": 0.20},  # Very Hard (100 classes)
    "pendulum": {"reward": -200.0},  # "Solved" is roughly -200
    "cartpole": {"reward": 100.0},  # Basic balancing
    "acrobot": {"reward": -100.0},
}


class PromotionGate:
    """Checks if model performance warrants promotion."""

    @staticmethod
    def check_promotion(task_name: str, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics satisfy promotion criteria for task.

        Args:
            task_name: The name of the task.
            metrics: Dictionary of performance metrics (e.g., {'accuracy': 0.95}).

        Returns:
            bool: True if promotion criteria are met, False otherwise.
        """
        thresholds = PROMOTION_THRESHOLDS.get(task_name)
        if not thresholds:
            return True  # No barrier

        acc = metrics.get("accuracy")
        rew = metrics.get("reward")

        # Check Accuracy
        if "accuracy" in thresholds:
            if acc is None or acc < thresholds["accuracy"]:
                return False

        # Check Reward
        if "reward" in thresholds:
            # Note: reward might be missing in metrics if not logged properly.
            # Assume fail if missing but required.
            if rew is None or rew < thresholds["reward"]:
                return False

        # Check Efficiency (if available)
        # If model is extremely slow (e.g. 100x slower than expected), fail promotion
        # unless it's the "Deep" tier where we care less about speed.
        if "time" in metrics and metrics["time"] > 0:
            # Simple heuristic: If accuracy is low but time is massive, don't promote.
            # But usually we promote based on success.
            # Let's enforce a minimum efficiency for lighter tasks.
            if task_name in ["digits", "mnist"] and metrics["time"] > 600.0: # > 10 mins for MNIST is bad
                 return False

        return True

    @staticmethod
    def get_threshold_desc(task_name: str) -> str:
        """
        Get human readable description of promotion thresholds.

        Args:
            task_name: The task name.

        Returns:
            str: Description of thresholds (e.g., "accuracy > 0.95").
        """
        t = PROMOTION_THRESHOLDS.get(task_name, {})
        return ", ".join([f"{k} > {v}" for k, v in t.items()])
