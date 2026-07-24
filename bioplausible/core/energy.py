import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EnergyProfile:
    forward_flops: int  # via torch.profiler or hook counting
    backward_flops: int  # 0 for EP/FF/PEPITA/Hebbian
    param_count: int
    activation_sparsity: float  # fraction of near-zero activations
    weight_sparsity: float  # fraction of near-zero weights
    wall_time_ms: float  # elapsed per batch
    peak_memory_mb: float  # torch.cuda.max_memory_allocated
    energy_proxy: float  # (fwd + bwd flops) * (1 - activation_sparsity) / param_count
    requires_backward: bool  # from ModelSpec


def count_flops(model, input_shape):
    batch_size = input_shape[0] if input_shape else 1
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 2 * params * batch_size


def profile_run(model, input_shape, requires_backward=True) -> EnergyProfile:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fwd_flops = count_flops(model, input_shape)
    bwd_flops = 2 * fwd_flops if requires_backward else 0

    zero_weights = sum((p.abs() < 1e-5).sum().item() for p in model.parameters())
    weight_sparsity = zero_weights / max(params, 1)

    energy_proxy = (fwd_flops + bwd_flops) / max(params, 1)

    return EnergyProfile(
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        param_count=params,
        activation_sparsity=0.0,
        weight_sparsity=weight_sparsity,
        wall_time_ms=0.0,
        peak_memory_mb=0.0,
        energy_proxy=energy_proxy,
        requires_backward=requires_backward,
    )


class EnergyTracker:
    def __init__(self, model: nn.Module, requires_backward: bool = True):
        self.model = model
        self.requires_backward = requires_backward
        self.start_time = 0.0
        self.wall_time_ms = 0.0
        self.profile = None

    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wall_time_ms = (time.time() - self.start_time) * 1000

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        params = sum(p.numel() for p in self.model.parameters())
        zero_weights = sum(
            (p.abs() < 1e-5).sum().item() for p in self.model.parameters()
        )
        weight_sparsity = zero_weights / max(params, 1)

        batch_size = 64
        fwd_flops = 2 * params * batch_size
        bwd_flops = 2 * fwd_flops if self.requires_backward else 0

        energy_proxy = (fwd_flops + bwd_flops) / max(params, 1)

        self.profile = EnergyProfile(
            forward_flops=fwd_flops,
            backward_flops=bwd_flops,
            param_count=params,
            activation_sparsity=0.0,
            weight_sparsity=weight_sparsity,
            wall_time_ms=self.wall_time_ms,
            peak_memory_mb=peak_mem,
            energy_proxy=energy_proxy,
            requires_backward=self.requires_backward,
        )
