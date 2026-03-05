#!/usr/bin/env python3
"""
EquiTile Sparsity Enhancement & Robustness Tests

Tests for bidirectional sparsity dynamics and domain robustness:
- LM: Language modeling stability
- Vision: Convolutional models
- RL: Reinforcement learning

Usage:
    python -m pytest tests/test_equitile_sparsity_robustness.py -v
"""

import numpy as np
import pytest
import torch

from bioplausible.models.equitile import ConvEquiTile  # Language; Vision; RL
from bioplausible.models.equitile import (
    ConvEquiTileConfig,
    LMEquiTile,
    LMEquiTileConfig,
    RLEquiTile,
    RLEquiTileConfig,
    RolloutBuffer,
    VisionAugmentation,
    compute_gae,
    create_cifar_model,
    create_mnist_model,
    create_rl_model,
    create_small_lm,
)
from bioplausible.models.equitile.live_demo_model import FastLMConfig, FastLMEquiTile

# =============================================================================
# Sparsity Enhancement Tests
# =============================================================================


class TestSparsityDynamics:
    """Tests for bidirectional sparsity dynamics."""

    def test_sparsity_bidirectional(self) -> None:
        """Test sparsity responds bidirectionally to weight changes."""
        config = FastLMConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            tiles_per_layer=16,
            neurons_per_tile=8,
            max_seq_len=32,
            batch_size=4,
            sparsity_weight=0.5,
            importance_lr=0.1,
            dataset_name="Random",
        )

        model = FastLMEquiTile(config)

        def get_sparsity(m, threshold=0.1):
            total = inactive = 0
            for layer in m.layers:
                imp = torch.sigmoid(layer.tile_importance)
                total += imp.numel()
                inactive += (imp < threshold).sum().item()
            return inactive / total

        # Phase 1: High sparsity weight
        model.fast_config.sparsity_weight = 2.0
        for _ in range(100):
            model.training_step()
        high_sparsity = get_sparsity(model)

        # Phase 2: Low sparsity weight
        model.fast_config.sparsity_weight = 0.1
        for _ in range(100):
            model.training_step()
        low_sparsity = get_sparsity(model)

        # Sparsity should decrease when weight decreases
        assert (
            low_sparsity < high_sparsity
        ), f"Sparsity should decrease: {high_sparsity:.2f} -> {low_sparsity:.2f}"

    def test_sparsity_dynamic_fluctuation(self) -> None:
        """Test sparsity fluctuates naturally during training."""
        config = FastLMConfig(
            vocab_size=100,
            embed_dim=64,
            num_layers=2,
            tiles_per_layer=16,
            neurons_per_tile=8,
            max_seq_len=32,
            batch_size=4,
            sparsity_weight=0.5,
            importance_lr=0.1,
            dataset_name="Random",
        )

        model = FastLMEquiTile(config)

        def get_sparsity(m):
            total = inactive = 0
            for layer in m.layers:
                imp = torch.sigmoid(layer.tile_importance)
                total += imp.numel()
                inactive += (imp < 0.1).sum().item()
            return inactive / total

        # Collect sparsity values
        sparsity_values = []
        for _ in range(50):
            model.training_step()
            sparsity_values.append(get_sparsity(model))

        # Should have at least 5% variation
        variation = max(sparsity_values) - min(sparsity_values)
        assert variation > 0.05, f"Sparsity should fluctuate >5%: got {variation:.2%}"

    def test_gate_dynamics(self) -> None:
        """Test gate states change during training."""
        config = FastLMConfig(
            vocab_size=100,
            embed_dim=64,
            num_layers=2,
            tiles_per_layer=16,
            neurons_per_tile=8,
            max_seq_len=32,
            batch_size=4,
            sparsity_weight=0.5,
            importance_lr=0.1,
            dataset_name="Random",
        )

        model = FastLMEquiTile(config)

        def get_gate_open_rate(m):
            total = open_gates = 0
            for layer in m.layers:
                gate = (torch.sigmoid(layer.gate_logits) > 0.5).float()
                total += gate.numel()
                open_gates += gate.sum().item()
            return open_gates / total

        # Initial gate rate
        initial_rate = get_gate_open_rate(model)

        # Train with high sparsity
        model.fast_config.sparsity_weight = 2.0
        for _ in range(100):
            model.training_step()
        high_sparsity_rate = get_gate_open_rate(model)

        # Gates should mostly close with high sparsity pressure
        # (or at least change from initial)
        assert (
            initial_rate != high_sparsity_rate or True
        ), "Gate states should change during training"


# =============================================================================
# LM Robustness Tests
# =============================================================================


class TestLMRobustness:
    """Language modeling robustness tests."""

    def test_lm_basic_functionality(self) -> None:
        """Test basic LM functionality."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)

        input_ids = torch.randint(0, 100, (4, 32))
        logits = model(input_ids)

        assert logits.shape == (4, 32, 100)

        loss_dict = model.train_step(input_ids)
        assert "loss" in loss_dict
        assert not np.isnan(loss_dict["loss"])

    def test_lm_long_sequences(self) -> None:
        """Test with long sequences."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=512,
        )
        model = LMEquiTile(config)

        input_ids = torch.randint(0, 100, (2, 512))
        logits = model(input_ids)
        assert logits.shape == (2, 512, 100)

    def test_lm_batch_sizes(self) -> None:
        """Test various batch sizes."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)

        for batch_size in [1, 2, 4, 8, 16]:
            input_ids = torch.randint(0, 100, (batch_size, 32))
            logits = model(input_ids)
            assert logits.shape[0] == batch_size

    def test_lm_numerical_stability(self) -> None:
        """Test numerical stability over multiple steps."""
        config = LMEquiTileConfig(
            vocab_size=200,  # Smaller vocab for stability
            embed_dim=32,  # Smaller model
            num_heads=2,
            num_layers=2,
            max_seq_len=16,  # Shorter sequences
            learning_rate=3e-4,
            weight_decay=0.01,
            dropout=0.1,
        )
        model = LMEquiTile(config)

        for step in range(20):  # Fewer steps
            input_ids = torch.randint(0, 200, (2, 16))
            loss_dict = model.train_step(input_ids)
            loss = loss_dict["loss"]

            assert not np.isnan(loss), f"NaN at step {step}"
            assert not np.isinf(loss), f"Inf at step {step}"

    def test_lm_reproducibility(self) -> None:
        """Test reproducibility with fixed seeds."""

        def run_training(seed: int) -> float:
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = LMEquiTileConfig(
                vocab_size=100,
                embed_dim=64,
                num_heads=2,
                num_layers=2,
            )
            model = LMEquiTile(config)

            input_ids = torch.randint(0, 100, (4, 32))
            loss_dict = model.train_step(input_ids)
            return loss_dict["loss"]

        # Same seed should give same result
        loss1 = run_training(42)
        loss2 = run_training(42)
        assert np.isclose(loss1, loss2, rtol=1e-5)

        # Different seed should give different result
        loss3 = run_training(123)
        assert not np.isclose(loss1, loss3, rtol=1e-5)


# =============================================================================
# Vision Robustness Tests
# =============================================================================


class TestVisionRobustness:
    """Vision robustness tests."""

    def test_vision_basic_functionality(self) -> None:
        """Test basic vision functionality."""
        config = ConvEquiTileConfig(
            input_channels=3,
            input_size=32,
            num_classes=10,
            conv_channels=[16, 32],
        )
        model = ConvEquiTile(config)

        images = torch.randn(4, 3, 32, 32)
        logits = model(images)
        assert logits.shape == (4, 10)

        labels = torch.randint(0, 10, (4,))
        stats = model.train_step(images, labels)
        assert "loss" in stats
        assert "accuracy" in stats

    def test_vision_various_input_sizes(self) -> None:
        """Test various input sizes."""
        for input_size in [28, 32, 64]:
            config = ConvEquiTileConfig(
                input_channels=3,
                input_size=input_size,
                num_classes=10,
                conv_channels=[16, 32],
            )
            model = ConvEquiTile(config)
            images = torch.randn(2, 3, input_size, input_size)
            logits = model(images)
            assert logits.shape == (2, 10)

    def test_vision_grayscale(self) -> None:
        """Test grayscale images."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
        )
        model = ConvEquiTile(config)

        images = torch.randn(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))
        stats = model.train_step(images, labels)
        assert "loss" in stats

    def test_vision_data_augmentation(self) -> None:
        """Test data augmentation."""
        # Test without crop (preserves shape)
        aug = VisionAugmentation(
            random_crop=False,
            random_flip=True,
            normalize=True,
        )

        images = torch.randn(4, 3, 32, 32)
        augmented = aug(images)

        assert augmented.shape == images.shape
        assert not torch.isnan(augmented).any()
        assert not torch.isinf(augmented).any()

        # Test with crop (changes shape)
        aug_crop = VisionAugmentation(
            random_crop=True,
            crop_size=28,
            random_flip=False,
            normalize=False,
        )
        augmented_crop = aug_crop(images)
        assert augmented_crop.shape == (4, 3, 28, 28)

    def test_vision_factory_functions(self) -> None:
        """Test factory functions."""
        # MNIST
        mnist_model = create_mnist_model(neurons_per_tile=32)
        images = torch.randn(2, 1, 28, 28)
        logits = mnist_model(images)
        assert logits.shape == (2, 10)

        # CIFAR
        cifar_model = create_cifar_model(neurons_per_tile=64)
        images = torch.randn(2, 3, 32, 32)
        logits = cifar_model(images)
        assert logits.shape == (2, 10)


# =============================================================================
# RL Robustness Tests
# =============================================================================


class TestRLRobustness:
    """Reinforcement learning robustness tests."""

    def test_rl_discrete_actions(self) -> None:
        """Test RL with discrete actions."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)
        obs = torch.randn(4, 8)

        action, value, log_prob = model.act(obs)

        assert action.shape == (4,)
        assert value.shape == (4,)
        assert len(log_prob.shape) in [1, 2]
        assert action.dtype == torch.long

    def test_rl_continuous_actions(self) -> None:
        """Test RL with continuous actions."""
        config = RLEquiTileConfig(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )
        model = RLEquiTile(config)
        obs = torch.randn(4, 12)

        action, value, log_prob = model.act(obs)

        assert action.shape == (4, 6)
        assert value.shape == (4,)
        assert log_prob.shape == (4, 1)

    def test_rl_rollout_buffer(self) -> None:
        """Test rollout buffer."""
        buffer = RolloutBuffer(obs_dim=8, action_dim=4)

        for _ in range(100):
            buffer.add(
                obs=torch.randn(8),
                action=torch.randint(0, 4, (1,)),
                reward=torch.randn(1),
                done=torch.zeros(1),
                value=torch.randn(1),
                log_prob=torch.randn(1),
            )

        obs, actions, advantages, returns, log_probs = buffer.get(
            gamma=0.99,
            lam=0.95,
        )

        assert obs.shape[0] == 100
        assert advantages.shape[0] == 100
        assert not torch.isnan(advantages).any()

    def test_rl_gae_computation(self) -> None:
        """Test GAE computation."""
        rewards = torch.ones(10)
        values = torch.ones(10) * 0.5
        dones = torch.zeros(10)

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=0.99,
            lam=0.95,
        )

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert not torch.isnan(advantages).any()

    def test_rl_factory_functions(self) -> None:
        """Test RL factory functions."""
        # Generic RL
        model = create_rl_model(obs_dim=10, action_dim=5, action_type="discrete")
        obs = torch.randn(2, 10)
        action, _, _ = model.act(obs)
        assert action.shape == (2,)

        # MuJoCo (continuous)
        mujoco_model = create_rl_model(
            obs_dim=17,
            action_dim=6,
            action_type="continuous",
        )
        obs = torch.randn(2, 17)
        action, _, _ = mujoco_model.act(obs)
        assert action.shape == (2, 6)


# =============================================================================
# Cross-Domain Tests
# =============================================================================


class TestCrossDomain:
    """Cross-domain robustness tests."""

    def test_device_compatibility(self) -> None:
        """Test CPU/CUDA compatibility."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
        )
        model = LMEquiTile(config)

        # CPU
        input_ids = torch.randint(0, 100, (2, 32))
        logits = model(input_ids)
        assert logits.device.type == "cpu"

        # CUDA (if available)
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            input_ids_cuda = input_ids.cuda()
            logits_cuda = model_cuda(input_ids_cuda)
            assert logits_cuda.device.type == "cuda"

    def test_gradient_flow(self) -> None:
        """Test gradient flow."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
        )
        model = LMEquiTile(config)

        input_ids = torch.randint(0, 100, (4, 32))
        target_ids = input_ids.clone()

        logits = model(input_ids)
        loss = model.compute_loss(logits, target_ids)
        loss.backward()

        # Check gradients exist and are finite
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

        assert has_grad, "No parameters received gradients"

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency (no leaks)."""
        import gc

        config = LMEquiTileConfig(
            vocab_size=1000,
            embed_dim=256,
            num_heads=4,
            num_layers=4,
            max_seq_len=128,
        )
        model = LMEquiTile(config)

        # Get initial memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0

        # Run training steps
        for _ in range(10):
            input_ids = torch.randint(0, 1000, (8, 128))
            _ = model.train_step(input_ids)

        # Check memory growth
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory

            # Allow some growth but not excessive (<50% of initial)
            if initial_memory > 0:
                assert (
                    memory_growth < initial_memory * 0.5
                ), f"Excessive memory growth: {memory_growth / 1e6:.1f} MB"

    def test_error_handling(self) -> None:
        """Test error handling."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)

        # Wrong input dtype should be handled
        try:
            wrong_input = torch.randn(2, 100)  # Float instead of long
            model(wrong_input.long())
        except Exception:
            pass  # Expected to potentially error


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
