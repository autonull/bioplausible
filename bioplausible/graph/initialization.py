# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.

"""Parameter initialization for GraphStructure.

Calls each node's initialize_params with a shared RNG.
"""

from __future__ import annotations

import torch

from bioplausible.graph.topology import GraphStructure


def initialize_params(
    structure: GraphStructure,
    rng_key: torch.Generator | int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    """Initialize all parameters in a GraphStructure.

    Returns {node_name: {param_name: tensor}} by calling each node's
    ``initialize_params(rng_key)`` method.

    Args:
        structure: The graph whose nodes need parameter initialization.
        rng_key: A torch.Generator or integer seed for deterministic init.

    Returns:
        Nested dict: node_name -> param_name -> tensor.
    """
    if isinstance(rng_key, int):
        gen = torch.Generator()
        gen.manual_seed(rng_key)
    else:
        gen = rng_key

    result: dict[str, dict[str, torch.Tensor]] = {}
    for node in structure.nodes:
        result[node.name] = node.initialize_params(gen)
    return result
