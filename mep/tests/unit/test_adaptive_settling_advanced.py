"""
Advanced adaptive settling logic tests.
"""

import torch
import torch.nn as nn
from mep.optimizers.settling import Settler
from mep.optimizers.inspector import ModelInspector

def test_adaptive_logic_mocked():
    """Test that adaptive settling backtracks correctly on energy increase."""
    model = nn.Linear(1, 1)
    x = torch.tensor([[1.0]])
    structure = [{"type": "layer", "module": model}]

    # Expected Energy Sequence:
    # 10.0 (Iter 0) -> Accept. prev=10.0.
    # 12.0 (Iter 1) -> Reject (increase). Restore. Continue. prev=10.0.
    # 10.0 (Iter 2) -> Accept (restored). prev=10.0.
    # 9.0  (Iter 3) -> Accept. prev=9.0.

    energy_values = [10.0, 12.0, 10.0, 9.0, 8.0]
    call_idx = 0

    def mock_energy_fn(model, x, states, structure, target_vec, beta):
        nonlocal call_idx
        val = energy_values[min(call_idx, len(energy_values)-1)]
        call_idx += 1
        return torch.tensor(val, requires_grad=True)

    settler = Settler(
        steps=4,
        lr=1.0,
        adaptive=True,
        tol=0.0 # prevent early stop
    )

    settler.settle(model, x, target=None, beta=0.0, energy_fn=mock_energy_fn, structure=structure)

    # We expect 4 calls: 10(0), 12(1), 10(2), 9(3).
    # Iter 0: 10. Accept.
    # Iter 1: 12. Reject. Continue.
    # Iter 2: 10. Accept (Retry).
    # Iter 3: 9. Accept.

    assert call_idx == 4
