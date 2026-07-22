# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.

"""Predictive coding inference via energy minimization (InferenceSGD).

Implements the PC settling procedure:
    E = || a_child - f_parent(a_parent, θ_parent) ||²
    a_child ← a_child - η_infer * ∂E/∂a_child

For feedforward graphs, settle = feedforward (1 step).
For cyclic graphs, multiple steps allow activity to stabilize.

This updates ACTIVITIES only, not weights.
Uses gradient-based activity updates for non-feedforward topologies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bioplausible.graph.topology import GraphStructure


class InferenceSGD:
    """Energy-minimization settling for predictive coding.

    FabricPC equivalent: the inference/settling loop that adjusts
    node activities to minimize prediction error.

    For feedforward graphs, one settle step = one feedforward pass
    (activities converge immediately since there's no recurrence).

    Attributes:
        eta_infer: Learning rate for activity updates (used for cyclic graphs).
        infer_steps: Number of settling steps.
    """

    def __init__(self, eta_infer: float = 0.05, infer_steps: int = 20) -> None:
        self.eta_infer = eta_infer
        self.infer_steps = infer_steps

    def _feedforward(
        self,
        structure: GraphStructure,
        params: dict[str, dict[str, torch.Tensor]],
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a feedforward pass through the graph.

        For the input node (task_map.x), activity is clamped to x.
        Other nodes compute their activity via forward with predecessor activities.
        """
        activities: dict[str, torch.Tensor] = {}
        try:
            topo_order = structure.topological_order()
            for node in topo_order:
                preds = structure.get_predecessors(node)
                slot_inputs: dict[str, torch.Tensor] = {}
                if not preds:
                    if node.name == structure.task_map.x.name:
                        slot_inputs["input"] = x
                    else:
                        slot_inputs["input"] = torch.zeros(
                            x.shape[0], 1, device=x.device
                        )
                else:
                    for src_node, target_slot in preds:
                        val = activities[src_node.name]
                        if target_slot.name in slot_inputs:
                            slot_inputs[target_slot.name] = (
                                slot_inputs[target_slot.name] + val
                            )
                        else:
                            slot_inputs[target_slot.name] = val
                slot_inputs.update(params.get(node.name, {}))
                activities[node.name] = node.forward(**slot_inputs)
        except ValueError:
            # Cyclic graph: fallback to per-node forward
            for node in structure.nodes:
                si: dict[str, torch.Tensor] = {}
                if node.name == structure.task_map.x.name:
                    si["input"] = x
                else:
                    si["input"] = torch.zeros(x.shape[0], 1, device=x.device)
                si.update(params.get(node.name, {}))
                activities[node.name] = node.forward(**si)
        return activities

    def settle(
        self,
        structure: GraphStructure,
        params: dict[str, dict[str, torch.Tensor]],
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the inference settling procedure.

        For feedforward graphs, this is equivalent to a forward pass.
        For cyclic graphs, multiple iterative steps are performed.

        Args:
            structure: The graph topology.
            params: Node parameters {node_name: {param_name: tensor}}.
            x: Input data tensor.
            y: Optional target tensor (clamps output node activity).

        Returns:
            Dict mapping node_name -> settled activity tensor.
        """
        # Feedforward pass for initialization (activities converge immediately
        # for feedforward graphs — only multiple steps needed for cyclic)
        activities = self._feedforward(structure, params, x)

        # For cyclic graphs, run iterative settling
        is_cyclic = False
        try:
            structure.topological_order()
        except ValueError:
            is_cyclic = True

        if is_cyclic:
            for step in range(self.infer_steps):
                eta = self.eta_infer * (1.0 - step / max(self.infer_steps, 1) * 0.5)
                new_activities = dict(activities)
                for node in structure.nodes:
                    if node.name == structure.task_map.x.name:
                        continue  # Clamp input
                    preds = structure.get_predecessors(node)
                    if not preds:
                        continue
                    # Compute prediction from each parent
                    total_pred = None
                    for src_node, target_slot in preds:
                        fwd_args = dict(params.get(src_node.name, {}))
                        fwd_args["input"] = activities[src_node.name]
                        pred = src_node.forward(**fwd_args)
                        if total_pred is None:
                            total_pred = pred
                        else:
                            total_pred = total_pred + pred
                    if total_pred is not None:
                        error = activities[node.name] - total_pred
                        new_activities[node.name] = activities[node.name] - eta * error
                activities = new_activities

        # Clamp output toward target if provided
        if y is not None:
            out_name = structure.task_map.y.name
            if isinstance(y, torch.Tensor):
                if y.dim() == 1 and activities[out_name].dim() > 1:
                    y_onehot = F.one_hot(
                        y, num_classes=activities[out_name].shape[-1]
                    ).float()
                    activities[out_name] = 0.5 * activities[out_name] + 0.5 * y_onehot
                else:
                    activities[out_name] = 0.5 * activities[out_name] + 0.5 * y.float()

        return activities

    def __repr__(self) -> str:
        return (
            f"InferenceSGD(eta_infer={self.eta_infer}, "
            f"infer_steps={self.infer_steps})"
        )
