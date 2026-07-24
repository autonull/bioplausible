"""
Curriculum Learning Manager.

Manages the progression of tasks from simple to complex, allowing models
to build upon simpler skills before tackling harder problems.
"""

from typing import Dict
from typing import List
from typing import Optional


class CurriculumManager:
    """
    Defines task tracks and progressions.

    Attributes:
        TRACKS (Dict[str, List[str]]): Mapping of track names to ordered task lists.
    """

    # Define tracks
    TRACKS: Dict[str, List[str]] = {
        # Vision: Start small (digits) -> Scaling (mnist) -> Complexity (cifar)
        "vision": [
            "digits",
            "usps",
            "kmnist",
            "mnist",
            "fashion_mnist",
            "svhn",
            "cifar10",
            "cifar100",
        ],
        "lm": ["char_ngram", "tiny_shakespeare"],
        # Pendulum is arguably harder than cartpole balance
        "rl": ["cartpole", "pendulum", "acrobot"],
    }

    def __init__(self) -> None:
        """Initialize the Curriculum Manager."""
        pass

    def get_next_task(
        self, model_family: str, current_task: str, success: bool
    ) -> Optional[str]:
        """
        Suggest next task based on current outcome.

        Args:
            model_family: The family of the model (unused currently but reserved).
            current_task: The name of the task just completed.
            success: Whether the task was completed successfully.

        Returns:
            The name of the next task, or None if no progression is available.
        """
        # Identify track
        track = None
        for t_name, t_list in self.TRACKS.items():
            if current_task in t_list:
                track = t_list
                break

        if not track:
            return None  # Unknown track

        try:
            curr_idx = track.index(current_task)
        except ValueError:
            return None

        if success:
            # Promote
            if curr_idx + 1 < len(track):
                return track[curr_idx + 1]
            else:
                return "completed_track"
        else:
            # Demote or Retry logic handled externally
            pass

        return None

    def get_initial_task(self, model_family: str) -> str:
        """
        Get starting task for a model family.

        Args:
            model_family: The type of model (e.g., 'transformer', 'mlp').

        Returns:
            The name of the initial task for this model type.
        """
        # Heuristics based on model type
        family = model_family.lower()
        if "transformer" in family or "lm" in family or "language" in family:
            return "char_ngram"
        elif "rl" in family or "control" in family:
            return "cartpole"
        else:
            return "digits"  # Start with smallest possible task for rapid iteration
