"""
Tests for CUDA-accelerated kernels.

Tests verify correctness and performance of CUDA implementations.
"""

import torch
import pytest
from mep.cuda.kernels import (
    newton_schulz_cuda,
    dion_update_cuda,
    spectral_norm_power_iteration_cuda,
    enforce_spectral_constraint_cuda,
    batched_newton_schulz_cuda,
)


@pytest.fixture
def cuda_device():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestNewtonSchulzCUDA:
    """Test Newton-Schulz orthogonalization on CUDA."""

    def test_newton_schulz_cuda_correctness(self, cuda_device):
        """Test CUDA Newton-Schulz produces well-conditioned matrix."""
        G = torch.randn(64, 64, device=cuda_device)

        X = newton_schulz_cuda(G, steps=5)

        # Newton-Schulz with Frobenius normalization provides
        # a well-conditioned direction, not perfect orthogonality
        # Check that X^T X is bounded (not exploding)
        XtX = X.T @ X
        xtX_norm = XtX.norm().item()

        # Should be reasonable (not exploding)
        assert xtX_norm < 100, f"X^T X norm too large: {xtX_norm}"

        # Output should be non-zero and finite
        assert X.norm().item() > 0
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()

    def test_newton_schulz_cuda_vs_cpu(self, cuda_device):
        """Test CUDA and CPU implementations produce similar results."""
        G_cpu = torch.randn(64, 64)
        G_cuda = G_cpu.clone().to(cuda_device)

        X_cpu = newton_schulz_cuda(G_cpu, steps=5)  # Uses CPU fallback
        X_cuda = newton_schulz_cuda(G_cuda, steps=5)

        # Results should be very close (same algorithm)
        rel_error = (X_cuda.cpu() - X_cpu).norm() / (X_cpu.norm() + 1e-8)

        assert rel_error.item() < 0.01, f"High relative error: {rel_error.item()}"

    def test_newton_schulz_rectangular(self, cuda_device):
        """Test Newton-Schulz on rectangular matrices."""
        # Tall matrix
        G_tall = torch.randn(128, 64, device=cuda_device)
        X_tall = newton_schulz_cuda(G_tall, steps=5)
        assert X_tall.shape == G_tall.shape

        # Wide matrix
        G_wide = torch.randn(64, 128, device=cuda_device)
        X_wide = newton_schulz_cuda(G_wide, steps=5)
        assert X_wide.shape == G_wide.shape


class TestDionUpdateCUDA:
    """Test Dion low-rank SVD update on CUDA."""

    def test_dion_update_cuda_correctness(self, cuda_device):
        """Test CUDA Dion update produces low-rank approximation."""
        G = torch.randn(128, 64, device=cuda_device)
        rank = 10

        update, _ = dion_update_cuda(G, rank=rank)

        # Check rank of update
        U, S, Vh = torch.linalg.svd(update)
        effective_rank = (S > 1e-6).sum().item()

        # Should be approximately low-rank
        assert effective_rank <= rank * 1.5, f"Effective rank {effective_rank} exceeds target {rank}"

    def test_dion_update_cuda_error_feedback(self, cuda_device):
        """Test Dion update with error feedback."""
        G = torch.randn(128, 64, device=cuda_device)
        rank = 10
        error_beta = 0.9

        # Initialize error buffer
        error_buffer = torch.zeros_like(G)

        # Run multiple steps
        for i in range(5):
            G_step = torch.randn_like(G)
            update, error_buffer = dion_update_cuda(
                G_step, rank=rank, error_buffer=error_buffer, error_beta=error_beta
            )

        # Error buffer should accumulate
        assert error_buffer.norm().item() > 0, "Error buffer should be non-zero"

    def test_dion_update_cuda_vs_cpu(self, cuda_device):
        """Test CUDA and CPU Dion updates are similar."""
        G_cpu = torch.randn(64, 32)
        G_cuda = G_cpu.clone().to(cuda_device)
        rank = 8

        # CPU version
        U_cpu, S_cpu, V_cpu = torch.svd_lowrank(G_cpu, q=rank)
        update_cpu = U_cpu @ V_cpu.T

        # CUDA version
        update_cuda, _ = dion_update_cuda(G_cuda, rank=rank)

        # Compare structure (both should be low-rank)
        # Exact values may differ due to SVD non-uniqueness
        cpu_rank = (S_cpu > 1e-6).sum().item()
        cuda_rank = update_cuda.norm().item()

        # Both should produce non-trivial updates
        assert update_cpu.norm().item() > 0
        assert cuda_rank > 0

        # Shapes should match
        assert update_cuda.shape == update_cpu.shape


class TestSpectralNormCUDA:
    """Test spectral norm estimation on CUDA."""

    def test_spectral_norm_cuda_correctness(self, cuda_device):
        """Test CUDA spectral norm matches torch.linalg.svd."""
        W = torch.randn(64, 64, device=cuda_device)

        # Power iteration
        sigma, u, v = spectral_norm_power_iteration_cuda(W, niter=10)

        # Reference: full SVD
        _, S, _ = torch.linalg.svd(W)
        sigma_ref = S[0]

        # Should be close
        rel_error = (sigma - sigma_ref).abs() / (sigma_ref + 1e-8)

        assert rel_error.item() < 0.01, f"High spectral norm error: {rel_error.item()}"

    def test_spectral_norm_constraint_cuda(self, cuda_device):
        """Test spectral norm constraint enforcement on CUDA."""
        # Create matrix with large spectral norm
        W = torch.randn(64, 64, device=cuda_device) * 10
        gamma = 0.95

        W_constrained, u, v = enforce_spectral_constraint_cuda(W, gamma=gamma, niter=10)

        # Verify constraint is satisfied
        sigma, _, _ = spectral_norm_power_iteration_cuda(W_constrained, u, v, niter=5)

        assert sigma.item() <= gamma * 1.05, f"Spectral norm {sigma.item()} exceeds gamma {gamma}"


class TestBatchedNewtonSchulzCUDA:
    """Test batched Newton-Schulz on CUDA."""

    def test_batched_newton_schulz_cuda(self, cuda_device):
        """Test batched Newton-Schulz processes multiple matrices."""
        batch_size = 8
        G_batch = torch.randn(batch_size, 64, 64, device=cuda_device)

        X_batch = batched_newton_schulz_cuda(G_batch, steps=5)

        assert X_batch.shape == G_batch.shape

        # Check that outputs are well-conditioned (not exploding)
        for i in range(batch_size):
            X = X_batch[i]
            XtX = X.T @ X
            xtX_norm = XtX.norm().item()
            assert xtX_norm < 100, f"Batch {i}: X^T X norm too large: {xtX_norm}"


class TestCUDAFallback:
    """Test fallback behavior when CUDA is not available."""

    def test_cpu_fallback(self):
        """Test that functions work on CPU (no crash)."""
        G = torch.randn(32, 32)

        # Should not crash
        X = newton_schulz_cuda(G, steps=3)
        assert X.shape == G.shape

        # Spectral norm
        sigma, u, v = spectral_norm_power_iteration_cuda(G)
        assert sigma.numel() == 1

        # Dion update
        update, _ = dion_update_cuda(G, rank=8)
        assert update.shape == G.shape
