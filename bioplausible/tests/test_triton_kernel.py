import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from bioplausible.models.triton_kernel import HAS_TRITON, TritonEqPropOps

# Define skip condition
skip_if_no_triton = pytest.mark.skipif(
    not TritonEqPropOps.is_available(), reason="Triton or CUDA not available"
)


class TestTritonKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.batch_size = 32
        self.hidden_dim = 128
        self.alpha = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        self.pre_act = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        self.bias = torch.randn(self.hidden_dim, device=self.device)

    def test_fallback_cpu(self):
        """Test the fallback logic on CPU."""
        if self.device.type != "cpu":
            pytest.skip("Test specific to CPU fallback")

        out = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, self.bias)
        expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(
            self.pre_act + self.bias
        )
        assert torch.allclose(out, expected, atol=1e-5)

    @skip_if_no_triton
    def test_triton_match(self):
        """Test that Triton kernel matches PyTorch implementation."""
        # Run Triton
        out_triton = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, self.bias)
        # Run PyTorch
        expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(
            self.pre_act + self.bias
        )
        assert torch.allclose(out_triton, expected, atol=1e-5)

    @skip_if_no_triton
    def test_no_bias_triton(self):
        """Test without bias (Triton)."""
        out_triton = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, None)
        expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(self.pre_act)
        assert torch.allclose(out_triton, expected, atol=1e-5)

    def test_no_bias_cpu(self):
        if self.device.type != "cpu":
            pytest.skip("Test specific to CPU fallback")

        out = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, None)
        expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(self.pre_act)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_linear_fallback_cpu(self):
        if self.device.type != "cpu":
            pytest.skip("Test specific to CPU fallback")

        h_target = self.h + self.pre_act
        out = TritonEqPropOps.step_linear(self.h, h_target, self.alpha)
        expected = (1 - self.alpha) * self.h + self.alpha * h_target
        assert torch.allclose(out, expected, atol=1e-5)

    @skip_if_no_triton
    def test_linear_triton_match(self):
        h_target = self.h + self.pre_act
        out_triton = TritonEqPropOps.step_linear(self.h, h_target, self.alpha)
        expected = (1 - self.alpha) * self.h + self.alpha * h_target
        assert torch.allclose(out_triton, expected, atol=1e-5)

    @skip_if_no_triton
    def test_cupy_integration(self):
        pytest.importorskip("cupy")
        import cupy as cp

        try:
            with cp.cuda.Device(0):
                _ = cp.array([1.0])
        except Exception:
            pytest.skip("CuPy runtime failure")

        # Create CuPy arrays
        h_cp = cp.random.randn(self.batch_size, self.hidden_dim, dtype=cp.float32)
        target_cp = cp.random.randn(self.batch_size, self.hidden_dim, dtype=cp.float32)

        # Run Triton CuPy kernel
        out_cp = TritonEqPropOps.step_linear_cupy(h_cp, target_cp, self.alpha)

        # Run NumPy/CPU baseline
        h_np = cp.asnumpy(h_cp)
        target_np = cp.asnumpy(target_cp)
        expected_np = (1 - self.alpha) * h_np + self.alpha * target_np

        assert np.allclose(cp.asnumpy(out_cp), expected_np, atol=1e-5)
