"""Factory for sampler implementations."""


from .base import BaseSampler
from .greedy import GreedySampler
from .island import IslandSampler
from .random import RandomSampler
from .ucb1 import UCB1Sampler


def get_sampler(algorithm: str, **kwargs) -> BaseSampler:
    """
    Construct a sampler by name.

    Args:
        algorithm: Algorithm name (`ucb1`, `random`, `greedy`, or `island`).
        **kwargs: Algorithm-specific parameters.

    Returns:
        A sampler instance.

    Raises:
        ValueError: If the algorithm name is unknown.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    samplers: dict[str, type[BaseSampler]] = {
        "ucb1": UCB1Sampler,
        "random": RandomSampler,
        "greedy": GreedySampler,
        "island": IslandSampler,
    }

    if algorithm not in samplers:
        raise ValueError(
            f"Unknown sampling algorithm: {algorithm}. "
            f"Available: {list(samplers.keys())}"
        )

    return samplers[algorithm](**kwargs)
