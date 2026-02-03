"""
Task Registry

Centralized registry for Experiment Tasks.
"""

from typing import Callable, Dict, Optional, Type

from bioplausible.hyperopt.tasks import BaseTask, LMTask, RLTask, VisionTask


class TaskRegistry:
    """Registry for task classes."""

    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str, task_cls: Type[BaseTask]):
        """Register a task class."""
        cls._tasks[name] = task_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseTask]:
        """Get a task class by name."""
        if name not in cls._tasks:
            raise ValueError(f"Task '{name}' not found in registry.")
        return cls._tasks[name]

    @classmethod
    def list_tasks(cls):
        """List registered tasks."""
        return list(cls._tasks.keys())


# Register core tasks
# We map generic names to classes. Specific instantiation logic (like parsing 'mnist_01')
# might still need a factory, or we can make the factory use this registry.

# For now, we register base types and let the factory decide which one to instantiate based on string analysis.
# Or better: The factory `create_task` logic should eventually move here or use this.

# Let's start simple: Register the classes so they can be looked up if needed explicitly.
TaskRegistry.register("lm", LMTask)
TaskRegistry.register("vision", VisionTask)
TaskRegistry.register("rl", RLTask)
