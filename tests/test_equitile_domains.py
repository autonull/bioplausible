#!/usr/bin/env python3
"""
Domain-Specific Tests for EquiTile

Tests for:
- Vision (ConvEquiTile)
- Language Modeling (LMEquiTile)
- Reinforcement Learning (RLEquiTile)

Usage:
    python -m pytest tests/test_equitile_domains.py -v
"""

import pytest
import torch

from bioplausible.equitile import (
    ConvEquiTile,  # Vision; Language; RL
    ConvEquiTileConfig,
    LMEquiTile,
    LMEquiTileConfig,
    RecurrentRLEquiTile,
    RLEquiTile,
    RLEquiTileConfig,
    RolloutBuffer,
    SimpleTokenizer,
    VisionAugmentation,
    compute_gae,
    create_cifar_model,
    create_mnist_model,
    create_rl_model,
    create_small_lm,
)

# =============================================================================
# Vision Tests
# =============================================================================


class TestVision:
    """Tests for ConvEquiTile vision module."""

    def test_conv_equitile_config(self) -> None:
        """Test ConvEquiTileConfig."""
        config = ConvEquiTileConfig(
            input_channels=3,
            input_size=32,
            num_classes=10,
        )

        assert config.input_channels == 3
        assert config.input_size == 32
        assert config.num_classes == 10
        assert len(config.conv_channels) == 3

    def test_conv_equitile_creation(self) -> None:
        """Test ConvEquiTile creation."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[16, 32],
        )
        model = ConvEquiTile(config)

        assert model is not None
        assert model.config.num_classes == 10

    def test_conv_equitile_forward(self) -> None:
        """Test ConvEquiTile forward pass."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[16, 32],
        )
        model = ConvEquiTile(config)
        model.eval()

        # Create batch of images
        images = torch.randn(4, 1, 28, 28)

        with torch.no_grad():
            logits = model(images)

        assert logits.shape == (4, 10)

    def test_conv_equitile_train_step(self) -> None:
        """Test ConvEquiTile training step."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[16, 32],
        )
        model = ConvEquiTile(config)

        images = torch.randn(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))

        stats = model.train_step(images, labels)

        assert "loss" in stats
        assert "accuracy" in stats
        assert stats["loss"] > 0

    def test_create_mnist_model(self) -> None:
        """Test MNIST model factory."""
        model = create_mnist_model(neurons_per_tile=32)

        assert model is not None
        assert model.config.num_classes == 10

    def test_create_cifar_model(self) -> None:
        """Test CIFAR model factory."""
        model = create_cifar_model(neurons_per_tile=64)

        assert model is not None
        assert model.config.input_channels == 3
        assert model.config.input_size == 32

    def test_vision_augmentation(self) -> None:
        """Test VisionAugmentation."""
        # Test without crop (preserves shape)
        aug = VisionAugmentation(
            random_crop=False,
            random_flip=True,
            normalize=False,
        )

        images = torch.rand(4, 3, 32, 32)
        augmented = aug(images)

        assert augmented.shape == images.shape
        assert augmented.min() >= 0
        assert augmented.max() <= 1

        # Test with crop (changes shape)
        aug_crop = VisionAugmentation(
            random_crop=True,
            crop_size=28,
            random_flip=False,
            normalize=False,
        )

        augmented_crop = aug_crop(images)
        assert augmented_crop.shape == (4, 3, 28, 28)


# =============================================================================
# Language Tests
# =============================================================================


class TestLanguage:
    """Tests for LMEquiTile language module."""

    def test_lm_equitile_config(self) -> None:
        """Test LMEquiTileConfig."""
        config = LMEquiTileConfig(
            vocab_size=1000,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
        )

        assert config.vocab_size == 1000
        assert config.embed_dim == 128
        assert config.num_heads == 4

    def test_lm_equitile_creation(self) -> None:
        """Test LMEquiTile creation."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)

        assert model is not None
        assert model.config.vocab_size == 100

    def test_lm_equitile_forward(self) -> None:
        """Test LMEquiTile forward pass."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)
        model.eval()

        # Create batch of token IDs
        input_ids = torch.randint(0, 100, (4, 16))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (4, 16, 100)

    def test_lm_equitile_train_step(self) -> None:
        """Test LMEquiTile training step."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)

        input_ids = torch.randint(0, 100, (4, 16))
        target_ids = torch.randint(0, 100, (4, 16))

        stats = model.train_step(input_ids, target_ids)

        assert "loss" in stats
        assert "perplexity" in stats
        assert stats["loss"] > 0

    def test_simple_tokenizer(self) -> None:
        """Test SimpleTokenizer."""
        tokenizer = SimpleTokenizer()

        # Test encoding
        text = "hello world"
        encoded = tokenizer.encode(text)
        assert len(encoded) > 0

        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert len(decoded) > 0

        # Test batch encoding
        texts = ["hello", "world"]
        batch = tokenizer.batch_encode(texts, max_length=10, padding=True)
        assert batch.shape == (2, 10)

    def test_create_small_lm(self) -> None:
        """Test small LM factory."""
        model = create_small_lm(vocab_size=500)

        assert model is not None
        assert model.config.embed_dim == 128
        assert model.config.num_layers == 2

    def test_lm_generate(self) -> None:
        """Test LMEquiTile generation."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            max_seq_len=32,
        )
        model = LMEquiTile(config)
        model.eval()

        input_ids = torch.randint(1, 50, (1, 8))  # Start with some tokens

        with torch.no_grad():
            generated = model.generate(input_ids, max_length=16)

        assert generated.shape[1] == 16
        assert generated.shape[0] == 1


# =============================================================================
# RL Tests
# =============================================================================


class TestRL:
    """Tests for RLEquiTile RL module."""

    def test_rl_equitile_config(self) -> None:
        """Test RLEquiTileConfig."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )

        assert config.obs_dim == 8
        assert config.action_dim == 4
        assert config.action_type == "discrete"

    def test_rl_equitile_discrete_creation(self) -> None:
        """Test RLEquiTile with discrete actions."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)

        assert model is not None
        assert model.config.action_type == "discrete"

    def test_rl_equitile_continuous_creation(self) -> None:
        """Test RLEquiTile with continuous actions."""
        config = RLEquiTileConfig(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )
        model = RLEquiTile(config)

        assert model is not None
        assert model.config.action_type == "continuous"

    def test_rl_equitile_act_discrete(self) -> None:
        """Test RLEquiTile action selection (discrete)."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)
        model.eval()

        obs = torch.randn(1, 8)

        with torch.no_grad():
            action, value, log_prob = model.act(obs)

        assert action.shape == (1,)
        assert value.shape == (1,)
        # Log prob can be [1] or [1, 1] depending on torch version/distribution
        assert log_prob.reshape(-1).shape == (1,)
        assert action.item() in range(4)

    def test_rl_equitile_act_continuous(self) -> None:
        """Test RLEquiTile action selection (continuous)."""
        config = RLEquiTileConfig(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )
        model = RLEquiTile(config)
        model.eval()

        obs = torch.randn(1, 12)

        with torch.no_grad():
            action, value, log_prob = model.act(obs)

        assert action.shape == (1, 6)
        assert value.shape == (1,)
        assert log_prob.shape == (
            1,
            1,
        )  # continuous: 2D action -> 2D log_prob (keepdim)

    def test_rl_equitile_evaluate_actions(self) -> None:
        """Test RLEquiTile action evaluation."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)

        obs = torch.randn(4, 8)
        actions = torch.randint(0, 4, (4,))

        log_prob, entropy, value = model.evaluate_actions(obs, actions)

        # For discrete: log_prob and entropy are per-sample (no sum)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_rl_equitile_train_step(self) -> None:
        """Test RLEquiTile training step."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)

        obs = torch.randn(4, 8)
        actions = torch.randint(0, 4, (4,))
        advantages = torch.randn(4, 1)
        returns = torch.randn(4)
        old_log_probs = torch.randn(4, 1)

        stats = model.train_step(obs, actions, advantages, returns, old_log_probs)

        assert "total_loss" in stats
        assert "policy_loss" in stats
        assert "value_loss" in stats

    def test_recurrent_rl_equitile(self) -> None:
        """Test RecurrentRLEquiTile."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RecurrentRLEquiTile(config, rnn_hidden_dim=64)

        assert model is not None
        assert model.rnn_hidden_dim == 64

        # Test hidden state reset
        model.reset_hidden(batch_size=4, device=torch.device("cpu"))
        assert model._hidden_state is not None

    def test_rollout_buffer(self) -> None:
        """Test RolloutBuffer."""
        buffer = RolloutBuffer(obs_dim=8, action_dim=4)

        # Add some transitions
        for _ in range(10):
            buffer.add(
                obs=torch.randn(8),
                action=torch.tensor(1),
                reward=torch.tensor(1.0),
                done=torch.tensor(0),
                value=torch.tensor(0.5),
                log_prob=torch.tensor(-1.0),
            )

        assert len(buffer) == 10

        # Get data with GAE
        obs, actions, advantages, returns, log_probs = buffer.get(
            gamma=0.99,
            lam=0.95,
            last_value=0.0,
        )

        assert obs.shape[0] == 10
        assert advantages.shape[0] == 10
        assert returns.shape[0] == 10

        # Buffer should be cleared
        assert len(buffer) == 0

    def test_compute_gae(self) -> None:
        """Test GAE computation."""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        dones = torch.tensor([0, 0, 0, 0])

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=0.99,
            lam=0.95,
            last_value=0.0,
        )

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert advantages.mean() > 0  # Should be positive for positive rewards

    def test_create_rl_model_discrete(self) -> None:
        """Test RL model factory (discrete)."""
        model = create_rl_model(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )

        assert model is not None
        assert model.config.action_type == "discrete"

    def test_create_rl_model_continuous(self) -> None:
        """Test RL model factory (continuous)."""
        model = create_rl_model(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )

        assert model is not None
        assert model.config.action_type == "continuous"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDomainIntegration:
    """Integration tests across domains."""

    def test_vision_to_rl_pipeline(self) -> None:
        """Test vision features can feed into RL."""
        # Create vision model
        vision_config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[16, 32],
        )
        vision_model = ConvEquiTile(vision_config)

        # Get feature dimension from vision model
        feature_dim = vision_model.feature_extractor.output_size

        # Create RL model with matching obs_dim
        rl_config = RLEquiTileConfig(
            obs_dim=feature_dim,
            action_dim=4,
            action_type="discrete",
        )
        rl_model = RLEquiTile(rl_config)

        # Process image through vision model
        images = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            vision_output = vision_model.extract_features(images)

        # Use vision output as RL observation
        obs = vision_output
        with torch.no_grad():
            action, value, log_prob = rl_model.act(obs)

        assert action.shape[0] == 4

    def test_language_to_rl_pipeline(self) -> None:
        """Test language features can feed into RL."""
        # Create language model
        lm_config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
        )
        lm_model = LMEquiTile(lm_config)

        # Create RL model
        rl_config = RLEquiTileConfig(
            obs_dim=64,  # Use LM embed dim
            action_dim=4,
            action_type="discrete",
        )
        rl_model = RLEquiTile(rl_config)

        # Process text through language model
        input_ids = torch.randint(0, 100, (4, 16))
        with torch.no_grad():
            hidden = lm_model.get_hidden_states(input_ids)

        # Use last hidden state as RL observation
        obs = hidden[:, -1, :]  # Last token
        with torch.no_grad():
            action, value, log_prob = rl_model.act(obs)

        assert action.shape[0] == 4


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
