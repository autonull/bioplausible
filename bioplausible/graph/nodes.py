# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.

"""Graph node abstractions for the FabricPC-inspired topology system.

Defines Slot, NodeBase, and built-in node types (Linear, ReLU, Tanh).
Node forward methods are pure functions: no in-place mutation, no side effects.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch
import torch.nn.functional as F


class Slot:
    """Named input port on a node.

    Analogous to FabricPC's slot concept — each incoming edge targets a specific slot.
    """

    def __init__(self, name: str, owner: NodeBase) -> None:
        self.name = name
        self.owner = owner

    def __repr__(self) -> str:
        return f"Slot({self.owner.name}.{self.name})"


class NodeBase(ABC):
    """Abstract base for all graph nodes in the topology system.

    Subclasses must implement:
        - forward(**slot_inputs) -> torch.Tensor  (pure function)
        - get_slots() -> dict[str, Slot]
        - slot(name: str) -> Slot
        - initialize_params(rng_key) -> dict[str, torch.Tensor]
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def forward(self, **slot_inputs: torch.Tensor) -> torch.Tensor:
        """Compute forward pass from slot inputs. MUST be a pure function."""

    @abstractmethod
    def get_slots(self) -> dict[str, Slot]:
        """Return dict of {slot_name: Slot} for this node."""

    def slot(self, name: str) -> Slot:
        """Get a specific Slot by name."""
        slots = self.get_slots()
        if name not in slots:
            raise KeyError(
                f"Node '{self.name}' has no slot '{name}'. "
                f"Available slots: {list(slots.keys())}"
            )
        return slots[name]

    @abstractmethod
    def initialize_params(self, rng_key: torch.Generator) -> dict[str, torch.Tensor]:
        """Initialize trainable parameters for this node.

        Args:
            rng_key: A torch.Generator for deterministic initialization.

        Returns:
            Dict mapping param_name -> tensor. Empty dict if no params.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


class Linear(NodeBase):
    """Linear (dense) node with weight and bias parameters.

    Slots: {"input"}
    Params: {"weight", "bias"}
    Forward: F.linear(x, weight, bias)

    FabricPC equivalent: Linear node with learnable affine transform.
    """

    def __init__(self, shape: tuple[int, int], name: str) -> None:
        super().__init__(name)
        self.shape = shape  # (in_features, out_features)
        self._slots: dict[str, Slot] = {"input": Slot("input", self)}

    def get_slots(self) -> dict[str, Slot]:
        return dict(self._slots)

    def forward(self, **slot_inputs: torch.Tensor) -> torch.Tensor:
        in_t = slot_inputs.get("input")
        if in_t is None:
            raise ValueError(f"Linear node '{self.name}' missing 'input' slot")
        weight = slot_inputs.get("weight")
        bias = slot_inputs.get("bias")
        return F.linear(in_t, weight, bias)

    def initialize_params(self, rng_key: torch.Generator) -> dict[str, torch.Tensor]:
        in_features, out_features = self.shape
        weight = torch.empty(out_features, in_features)
        weight = torch.nn.init.kaiming_uniform_(weight, a=0, generator=rng_key)
        bias = torch.empty(out_features)
        bias = torch.nn.init.uniform_(bias, -0.1, 0.1, generator=rng_key)
        return {"weight": weight, "bias": bias}


class ReLU(NodeBase):
    """ReLU activation node. No trainable parameters.

    Slots: {"input"}
    Params: none
    Forward: F.relu(x)
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._slots: dict[str, Slot] = {"input": Slot("input", self)}

    def get_slots(self) -> dict[str, Slot]:
        return dict(self._slots)

    def forward(self, **slot_inputs: torch.Tensor) -> torch.Tensor:
        in_t = slot_inputs.get("input")
        if in_t is None:
            raise ValueError(f"ReLU node '{self.name}' missing 'input' slot")
        return F.relu(in_t)

    def initialize_params(self, rng_key: torch.Generator) -> dict[str, torch.Tensor]:
        return {}


class Tanh(NodeBase):
    """Tanh activation node. No trainable parameters.

    Slots: {"input"}
    Params: none
    Forward: torch.tanh(x)
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._slots: dict[str, Slot] = {"input": Slot("input", self)}

    def get_slots(self) -> dict[str, Slot]:
        return dict(self._slots)

    def forward(self, **slot_inputs: torch.Tensor) -> torch.Tensor:
        in_t = slot_inputs.get("input")
        if in_t is None:
            raise ValueError(f"Tanh node '{self.name}' missing 'input' slot")
        return torch.tanh(in_t)

    def initialize_params(self, rng_key: torch.Generator) -> dict[str, torch.Tensor]:
        return {}
