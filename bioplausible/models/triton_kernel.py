"""
Triton Kernels for EqProp Acceleration (Priority 2)

Provides fused kernels for Equilibrium Propagation dynamics to maximize
GPU throughput by reducing memory bandwidth usage.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

# Try to import CuPy for type checking/pointer access if available
try:
    import cupy as cp

    # Robust check
    if hasattr(cp, "cuda") and cp.cuda.is_available():
        with cp.cuda.Device(0):
            _ = cp.array([1.0])
            _ = cp.random.rand(1)
        HAS_CUPY = True
    else:
        HAS_CUPY = False
        cp = None
except (ImportError, Exception):
    cp = None
    HAS_CUPY = False

if HAS_TRITON:

    @triton.jit
    def _eqprop_step_kernel(
        h_ptr,  # Current hidden state
        pre_act_ptr,  # Linear projection (Wx + Wh)
        out_ptr,  # Output pointer
        alpha,  # Nudge factor
        n_elements,  # Total elements
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel for: h_new = (1 - alpha) * h + alpha * tanh(pre_act)
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        h = tl.load(h_ptr + offsets, mask=mask)
        pre = tl.load(pre_act_ptr + offsets, mask=mask)

        val = tl.tanh(pre)
        out = (1.0 - alpha) * h + alpha * val

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _eqprop_step_kernel_with_bias(
        h_ptr,  # Current hidden state
        pre_act_ptr,  # Linear projection (Wx + Wh)
        bias_ptr,  # Bias vector
        out_ptr,  # Output pointer
        alpha,  # Nudge factor
        n_rows,  # Batch size
        n_cols,  # Hidden dim
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel including bias addition:
        h_new = (1 - alpha) * h + alpha * tanh(pre_act + bias)
        """
        # Program ID covers the flattened array
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        n_elements = n_rows * n_cols
        mask = offsets < n_elements

        # Map linear offset to row/col
        # row = offsets // n_cols  # Not needed
        col = offsets % n_cols

        # Load
        h = tl.load(h_ptr + offsets, mask=mask)
        pre = tl.load(pre_act_ptr + offsets, mask=mask)
        b = tl.load(bias_ptr + col, mask=mask)  # Broadcast bias

        val = tl.tanh(pre + b)
        out = (1.0 - alpha) * h + alpha * val

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _eqprop_step_linear_kernel(
        h_ptr,  # Current hidden state
        target_ptr,  # Target state (e.g. h + ffn_out + x)
        out_ptr,  # Output pointer
        alpha,  # Nudge factor
        n_elements,  # Total elements
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel for linear relaxation: h_new = (1 - alpha) * h + alpha * target
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        h = tl.load(h_ptr + offsets, mask=mask)
        target = tl.load(target_ptr + offsets, mask=mask)

        out = (1.0 - alpha) * h + alpha * target

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _neural_cube_update_kernel(
        h_ptr,  # [batch, n_neurons]
        w_ptr,  # [n_neurons, 27]
        out_ptr,  # [batch, n_neurons]
        cube_size,  # int
        n_neurons,  # int (cube_size**3)
        n_elements,  # total elements (batch * n_neurons)
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel for NeuralCube 3D local update.
        Computes weighted sum of 26 neighbors for each neuron.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Current global index -> batch_idx, neuron_idx
        batch_idx = offsets // n_neurons
        neuron_idx = offsets % n_neurons

        # Decompose neuron_idx into z, y, x
        s2 = cube_size * cube_size
        z = neuron_idx // s2
        rem = neuron_idx % s2
        y = rem // cube_size
        x = rem % cube_size

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # Loop over 27 neighbors (3x3x3)
        # Using constant loops for unrolling
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nz = z + dz
                    ny = y + dy
                    nx = x + dx

                    # Check bounds
                    in_bounds = (
                        (nz >= 0)
                        & (nz < cube_size)
                        & (ny >= 0)
                        & (ny < cube_size)
                        & (nx >= 0)
                        & (nx < cube_size)
                    )

                    # Neighbor neuron index
                    n_neuron_idx = nz * s2 + ny * cube_size + nx

                    # Global h index: same batch
                    h_idx = batch_idx * n_neurons + n_neuron_idx

                    # Weight index: specific to this neuron and this neighbor offset
                    # w_offset maps (-1,-1,-1) -> 0 ... (1,1,1) -> 26
                    w_offset = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)
                    w_idx = neuron_idx * 27 + w_offset

                    # Masked load
                    # If out of bounds, we load from index 0 (or anywhere valid) but mask ensures result is 0
                    # However, to be safe and avoid OOB memory access, we clamp index or use safe mask
                    # If !in_bounds, h_idx might be invalid if we didn't clamp?
                    # Actually, if !in_bounds, h_idx calculation might produce something weird but still within range?
                    # No, n_neuron_idx could be anything if we don't check.
                    # But we use `mask & in_bounds` for loading.
                    # Does triton load safe with false mask? Yes.

                    h_val = tl.load(h_ptr + h_idx, mask=mask & in_bounds, other=0.0)
                    w_val = tl.load(
                        w_ptr + w_idx, mask=mask
                    )  # Weight exists for all neurons

                    acc += h_val * w_val

        tl.store(out_ptr + offsets, acc, mask=mask)

else:
    # Dummy definitions to prevent NameErrors if HAS_TRITON is False
    _eqprop_step_kernel = None
    _eqprop_step_kernel_with_bias = None
    _eqprop_step_linear_kernel = None
    _neural_cube_update_kernel = None


class TritonEqPropOps:
    """Interface for Triton kernels."""

    _triton_functioning = True

    @staticmethod
    def is_available():
        if not TritonEqPropOps._triton_functioning:
            return False
        return HAS_TRITON and torch.cuda.is_available()

    @staticmethod
    def step(
        h: torch.Tensor,
        pre_act: torch.Tensor,
        alpha: float,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one EqProp step: h <- (1-a)h + a*tanh(pre_act + bias)
        """
        if not TritonEqPropOps.is_available():
            # Fallback
            if bias is not None:
                return torch.lerp(h, torch.tanh(pre_act + bias), alpha)
            return torch.lerp(h, torch.tanh(pre_act), alpha)

        # Ensure contiguity for safe pointer access
        if not h.is_contiguous():
            h = h.contiguous()
        if not pre_act.is_contiguous():
            pre_act = pre_act.contiguous()
        if bias is not None and not bias.is_contiguous():
            bias = bias.contiguous()

        try:
            # Prepare output
            out = torch.empty_like(h)

            n_elements = h.numel()

            # Heuristic for block size
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            if bias is not None:
                # Check shapes
                assert bias.dim() == 1
                assert h.dim() == 2
                n_rows, n_cols = h.shape
                assert bias.shape[0] == n_cols

                _eqprop_step_kernel_with_bias[grid](
                    h, pre_act, bias, out, alpha, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
                )
            else:
                _eqprop_step_kernel[grid](
                    h, pre_act, out, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
                )
            return out
        except Exception:
            # Disable Triton for future calls
            TritonEqPropOps._triton_functioning = False
            # Fallback
            if bias is not None:
                return torch.lerp(h, torch.tanh(pre_act + bias), alpha)
            return torch.lerp(h, torch.tanh(pre_act), alpha)

    @staticmethod
    def step_cupy(h, pre_act, alpha: float):
        """
        Perform one EqProp step using CuPy arrays: h <- (1-a)h + a*tanh(pre_act)
        """
        # Fallback if no Triton or no CuPy
        if not HAS_TRITON or not HAS_CUPY:
            return (1 - alpha) * h + alpha * cp.tanh(pre_act)

        # Ensure we are dealing with CuPy arrays on GPU
        if not isinstance(h, cp.ndarray) or not isinstance(pre_act, cp.ndarray):
            return (1 - alpha) * h + alpha * cp.tanh(pre_act)

        try:
            # Ensure contiguity
            if not h.flags.c_contiguous:
                h = cp.ascontiguousarray(h)
            if not pre_act.flags.c_contiguous:
                pre_act = cp.ascontiguousarray(pre_act)

            out = cp.empty_like(h)
            n_elements = h.size
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            _eqprop_step_kernel[grid](
                h.data.ptr,  # h_ptr
                pre_act.data.ptr,  # pre_act_ptr
                out.data.ptr,  # out_ptr
                alpha,  # alpha
                n_elements,  # n_elements
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out
        except Exception:
            TritonEqPropOps._triton_functioning = False
            return (1 - alpha) * h + alpha * cp.tanh(pre_act)

    @staticmethod
    def neural_cube_update(
        h: torch.Tensor, w_local: torch.Tensor, cube_size: int
    ) -> torch.Tensor:
        """
        Perform 3D local update for NeuralCube.

        Args:
            h: [batch, n_neurons]
            w_local: [n_neurons, 27]
            cube_size: Dimension of the cube

        Returns:
            Weighted sum [batch, n_neurons]
        """
        if not TritonEqPropOps.is_available():
            raise RuntimeError("Triton not available")

        # Ensure contiguity
        if not h.is_contiguous():
            h = h.contiguous()
        if not w_local.is_contiguous():
            w_local = w_local.contiguous()

        try:
            batch_size, n_neurons = h.shape
            n_elements = batch_size * n_neurons

            out = torch.empty_like(h)

            BLOCK_SIZE = 512  # Smaller block size due to heavy computation per thread
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            _neural_cube_update_kernel[grid](
                h, w_local, out, cube_size, n_neurons, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )

            return out
        except Exception:
            TritonEqPropOps._triton_functioning = False
            raise RuntimeError(
                "Triton failed and no fallback available for neural_cube_update (yet)"
            )

    @staticmethod
    def step_linear(
        h: torch.Tensor, target: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        Perform one EqProp linear step: h <- (1-a)h + a*target
        """
        if not TritonEqPropOps.is_available():
            # Fallback
            return torch.lerp(h, target, alpha)

        # Ensure contiguity
        if not h.is_contiguous():
            h = h.contiguous()
        if not target.is_contiguous():
            target = target.contiguous()

        try:
            out = torch.empty_like(h)
            n_elements = h.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            _eqprop_step_linear_kernel[grid](
                h, target, out, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )

            return out
        except Exception:
            TritonEqPropOps._triton_functioning = False
            return torch.lerp(h, target, alpha)

    @staticmethod
    def step_linear_cupy(h, target, alpha: float):
        """
        Perform one EqProp linear step using CuPy arrays: h <- (1-a)h + a*target
        """
        # Fallback if no Triton or no CuPy
        if not HAS_TRITON or not HAS_CUPY:
            # This fallback assumes standard NumPy/CuPy broadcasting
            return (1 - alpha) * h + alpha * target

        # Ensure we are dealing with CuPy arrays on GPU
        if not isinstance(h, cp.ndarray) or not isinstance(target, cp.ndarray):
            return (1 - alpha) * h + alpha * target

        try:
            # Ensure contiguity (Triton requires contiguous memory)
            if not h.flags.c_contiguous:
                h = cp.ascontiguousarray(h)
            if not target.flags.c_contiguous:
                target = cp.ascontiguousarray(target)

            out = cp.empty_like(h)
            n_elements = h.size
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            # Launch kernel using raw pointers
            _eqprop_step_linear_kernel[grid](
                h.data.ptr,  # h_ptr
                target.data.ptr,  # target_ptr
                out.data.ptr,  # out_ptr
                alpha,  # alpha
                n_elements,  # n_elements
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out
        except Exception:
            TritonEqPropOps._triton_functioning = False
            return (1 - alpha) * h + alpha * target
