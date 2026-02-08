"""
Task promotion logic.
Defines when a model is ready to move to the next task difficulty.
"""

from typing import Dict, Any

# Minimum success criteria for each task
PROMOTION_THRESHOLDS = {
    "char_ngram": {"accuracy": 0.95},  # Should be trivial
    "digits": {"accuracy": 0.90},    # Tiny, should be easy
    "usps": {"accuracy": 0.85},
    "kmnist": {"accuracy": 0.80},
    "mnist": {"accuracy": 0.85},     # Good baseline
    "fashion_mnist": {"accuracy": 0.75},
    "svhn": {"accuracy": 0.60},      # Noisier than F-MNIST
    "cifar10": {"accuracy": 0.45},   # Harder
    "cifar100": {"accuracy": 0.20},  # Very Hard (100 classes)
    "pendulum": {"reward": -200.0},  # "Solved" is roughly -200
    "cartpole": {"reward": 100.0},   # Basic balancing
    "acrobot": {"reward": -100.0},
}


class PromotionGate:
    """Checks if model performance warrants promotion."""

    @staticmethod
    def check_promotion(task_name: str, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics satisfy promotion criteria for task.
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

        return True

    @staticmethod
    def get_threshold_desc(task_name: str) -> str:
        """Get human readable description."""
        t = PROMOTION_THRESHOLDS.get(task_name, {})
        return ", ".join([f"{k} > {v}" for k, v in t.items()])
