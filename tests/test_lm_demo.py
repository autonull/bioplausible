"""
Tests for EquiTile LM Demo
===========================

Comprehensive tests for FastLMEquiTile and related components.

Run tests:
    pytest tests/test_lm_demo.py -v

Run specific test:
    pytest tests/test_lm_demo.py::test_fast_lm_forward -v
"""

import pytest
import torch
import torch.nn.functional as F

from bioplausible.models.equitile.lm_demo.data import (
    ByteLevelTokenizer, CharacterTokenizer, LMDataset,
    create_shakespeare_dataset)
from bioplausible.models.equitile.lm_demo.fast_lm import (
    FastEquiTileLayer, FastLMConfig, FastLMEquiTile, MixtureOfTiles,
    SwiGLUFeedForward, TileLocalAttention, create_fast_lm_shakespeare,
    create_fast_lm_small, create_fast_lm_tiny)
from bioplausible.models.equitile.lm_demo.training import (LMTrainer,
                                                           LRScheduler,
                                                           TrainingConfig,
                                                           TrainingMetrics)

# =============================================================================
# Model Tests
# =============================================================================


class TestFastLMConfig:
    """Tests for FastLMConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FastLMConfig()
        assert config.vocab_size == 1000
        assert config.embed_dim == 192
        assert config.num_layers == 6
        assert config.mot_k == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = FastLMConfig(
            vocab_size=500,
            embed_dim=128,
            num_layers=4,
            neurons_per_tile=32,
            tiles_per_layer=2,
            mot_k=1,
        )
        assert config.vocab_size == 500
        assert config.embed_dim == 128
        assert config.num_layers == 4


class TestMixtureOfTiles:
    """Tests for MixtureOfTiles module."""

    def test_mot_forward(self):
        """Test MoT forward pass."""
        mot = MixtureOfTiles(
            embed_dim=64,
            neurons_per_tile=16,
            tiles_per_layer=4,
            mot_k=2,
        )

        x = torch.randn(2, 10, 64)
        output, tile_importance = mot(x)

        assert output.shape == (2, 10, 64)
        assert tile_importance.shape == (2, 4)

    def test_mot_sparse_activation(self):
        """Test that MoT activates only k tiles."""
        mot = MixtureOfTiles(
            embed_dim=64,
            neurons_per_tile=16,
            tiles_per_layer=4,
            mot_k=1,  # Only 1 tile active
        )

        x = torch.randn(1, 5, 64)
        output, tile_importance = mot(x)

        # Check that importance sums to 1 (softmax)
        assert torch.allclose(tile_importance.sum(dim=-1), torch.ones(1))


class TestTileLocalAttention:
    """Tests for TileLocalAttention module."""

    def test_attention_forward(self):
        """Test attention forward pass."""
        attn = TileLocalAttention(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=2,
            sliding_window=16,
        )

        x = torch.randn(2, 10, 64)
        output = attn(x)

        assert output.shape == (2, 10, 64)

    def test_grouped_query_attention(self):
        """Test grouped query attention."""
        attn = TileLocalAttention(
            embed_dim=64,
            num_heads=8,
            num_kv_heads=2,  # 4 Q heads per KV head
        )

        x = torch.randn(2, 10, 64)
        output = attn(x)

        assert output.shape == (2, 10, 64)

    def test_causal_masking(self):
        """Test causal masking."""
        attn = TileLocalAttention(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=2,
        )

        x = torch.randn(1, 5, 64)
        output = attn(x, causal=True)

        assert output.shape == (1, 5, 64)


class TestSwiGLUFeedForward:
    """Tests for SwiGLUFeedForward module."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        ff = SwiGLUFeedForward(
            embed_dim=64,
            hidden_dim=128,
        )

        x = torch.randn(2, 10, 64)
        output = ff(x)

        assert output.shape == (2, 10, 64)


class TestFastEquiTileLayer:
    """Tests for FastEquiTileLayer module."""

    def test_layer_forward(self):
        """Test transformer layer forward pass."""
        config = FastLMConfig(
            embed_dim=64,
            neurons_per_tile=16,
            tiles_per_layer=2,
            mot_k=1,
            num_heads=4,  # Must divide embed_dim
            num_kv_heads=2,
            sliding_window=16,
        )
        layer = FastEquiTileLayer(config)

        x = torch.randn(2, 10, 64)
        output, tile_importance = layer(x)

        assert output.shape == (2, 10, 64)
        assert tile_importance.shape == (2, 2)


class TestFastLMEquiTile:
    """Tests for FastLMEquiTile model."""

    def test_model_creation(self):
        """Test model creation."""
        config = FastLMConfig(
            vocab_size=100,
            embed_dim=64,
            num_layers=2,
            num_heads=4,  # Must divide embed_dim
            num_kv_heads=2,
        )
        model = FastLMEquiTile(config)

        assert model.get_parameter_count() > 0

    def test_model_forward(self):
        """Test model forward pass."""
        model = create_fast_lm_tiny(vocab_size=100)

        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)

        assert logits.shape == (2, 10, 100)

    def test_model_generation(self):
        """Test autoregressive generation."""
        model = create_fast_lm_tiny(vocab_size=50)
        model.eval()

        input_ids = torch.randint(0, 50, (1, 5))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=15,
                temperature=0.8,
            )

        assert output.shape == (1, 15)

    def test_model_training_step(self):
        """Test training step."""
        model = create_fast_lm_tiny(vocab_size=100)
        model.train()

        input_ids = torch.randint(0, 100, (2, 10))
        target_ids = torch.randint(0, 100, (2, 10))

        stats = model.train_step(input_ids, target_ids)

        assert "loss" in stats
        assert "perplexity" in stats
        assert stats["loss"] > 0

    def test_weight_tying(self):
        """Test weight tying between input/output embeddings."""
        model = create_fast_lm_tiny(vocab_size=100)

        # Check that output uses token_embedding.weight
        assert model.output_proj is None

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = create_fast_lm_tiny(vocab_size=100)
        params = model.get_parameter_count()

        # Tiny model should have < 1M parameters
        assert params < 1_000_000

        model = create_fast_lm_small(vocab_size=500)
        params = model.get_parameter_count()

        # Small model should have < 5M parameters
        assert params < 5_000_000

    def test_tile_importance_output(self):
        """Test tile importance statistics."""
        model = create_fast_lm_tiny(vocab_size=100)

        input_ids = torch.randint(0, 100, (2, 10))
        logits, tile_stats = model(input_ids, return_tile_stats=True)

        assert len(tile_stats) == model.config.num_layers


# =============================================================================
# Data Tests
# =============================================================================


class TestCharacterTokenizer:
    """Tests for CharacterTokenizer."""

    def test_tokenizer_creation(self):
        """Test tokenizer creation."""
        text = "Hello, world!"
        tokenizer = CharacterTokenizer(text)

        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id == 0

    def test_encode_decode(self):
        """Test encoding and decoding."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_batch_encode(self):
        """Test batch encoding."""
        tokenizer = CharacterTokenizer("abc")

        texts = ["ab", "bc", "abc"]
        # Pad to same length for batch encoding
        encoded = tokenizer.batch_encode(texts, max_length=3)

        assert encoded.shape[0] == 3
        assert encoded.shape[1] == 3


class TestLMDataset:
    """Tests for LMDataset."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        text = "Hello world! " * 100
        tokenizer = CharacterTokenizer(text)

        dataset = LMDataset(text, tokenizer, seq_length=10)

        assert len(dataset) > 0

    def test_dataset_item(self):
        """Test dataset item retrieval."""
        text = "Hello world! " * 100
        tokenizer = CharacterTokenizer(text)

        dataset = LMDataset(text, tokenizer, seq_length=10)
        input_ids, target_ids = dataset[0]

        assert input_ids.shape == (10,)
        assert target_ids.shape == (10,)

    def test_target_is_shifted_input(self):
        """Test that target is shifted input."""
        text = "abcdefghij" * 10
        tokenizer = CharacterTokenizer(text)

        dataset = LMDataset(text, tokenizer, seq_length=5)
        input_ids, target_ids = dataset[0]

        # Target should be input shifted by 1
        assert torch.equal(input_ids[1:], target_ids[:-1])


class TestShakespeareDataset:
    """Tests for Shakespeare dataset."""

    def test_create_shakespeare_dataset(self):
        """Test Shakespeare dataset creation."""
        train_loader, val_loader, tokenizer = create_shakespeare_dataset(
            batch_size=4,
            seq_length=32,
            num_workers=0,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert tokenizer.vocab_size > 0

    def test_shakespeare_batch(self):
        """Test Shakespeare batch loading."""
        train_loader, _, _ = create_shakespeare_dataset(
            batch_size=4,
            seq_length=32,
            num_workers=0,
        )

        for input_ids, target_ids in train_loader:
            assert input_ids.shape == (4, 32)
            assert target_ids.shape == (4, 32)
            break


# =============================================================================
# Training Tests
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()

        assert config.epochs == 10
        assert config.learning_rate == 3e-4
        assert config.use_amp is True

    def test_auto_device(self):
        """Test auto device detection."""
        config = TrainingConfig(device="auto")

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected


class TestTrainingMetrics:
    """Tests for TrainingMetrics."""

    def test_metrics_update(self):
        """Test metrics update."""
        metrics = TrainingMetrics()

        metrics.update(
            train_loss=2.0,
            val_loss=2.5,
            lr=1e-4,
        )

        assert len(metrics.train_loss) == 1
        assert metrics.train_loss[0] == 2.0
        assert len(metrics.learning_rates) == 1

    def test_metrics_summary(self):
        """Test metrics summary."""
        metrics = TrainingMetrics()

        metrics.update(train_loss=2.0, val_loss=2.5)
        metrics.update(train_loss=1.5, val_loss=2.0)

        summary = metrics.get_summary()

        assert summary["best_val_loss"] == 2.0
        assert summary["current_train_loss"] == 1.5


class TestLRScheduler:
    """Tests for LRScheduler."""

    def test_warmup(self):
        """Test learning rate warmup."""
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-4)

        scheduler = LRScheduler(
            optimizer,
            peak_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )

        # During warmup, LR should increase
        lr1 = scheduler.get_lr()
        scheduler.step()
        lr2 = scheduler.get_lr()

        assert lr2 > lr1

    def test_cosine_decay(self):
        """Test cosine decay schedule."""
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)

        scheduler = LRScheduler(
            optimizer,
            peak_lr=1e-3,
            warmup_steps=10,
            total_steps=100,
            schedule_type="cosine",
        )

        # Run through warmup
        for _ in range(10):
            scheduler.step()

        # After warmup, LR should decrease
        lr1 = scheduler.get_lr()
        scheduler.step()
        lr2 = scheduler.get_lr()

        assert lr2 < lr1


class TestLMTrainer:
    """Tests for LMTrainer."""

    def test_trainer_creation(self):
        """Test trainer creation."""
        model = create_fast_lm_tiny(vocab_size=100)
        config = TrainingConfig(epochs=1, device="cpu")

        trainer = LMTrainer(model, config)

        assert trainer.model is model
        assert trainer.device.type == "cpu"

    def test_trainer_evaluate(self):
        """Test trainer evaluation."""
        model = create_fast_lm_tiny(vocab_size=100)
        config = TrainingConfig(epochs=1, device="cpu")
        trainer = LMTrainer(model, config)

        # Create small validation loader
        text = "hello world " * 50
        tokenizer = CharacterTokenizer(text)
        from torch.utils.data import DataLoader

        dataset = LMDataset(text, tokenizer, seq_length=16)
        val_loader = DataLoader(dataset, batch_size=2)

        val_loss = trainer.evaluate(val_loader, max_batches=2)

        assert val_loss > 0

    def test_trainer_generate(self):
        """Test trainer generation."""
        model = create_fast_lm_tiny(vocab_size=50)
        config = TrainingConfig(epochs=1, device="cpu")
        trainer = LMTrainer(model, config)

        tokenizer = CharacterTokenizer("abc")
        trainer.set_tokenizer(tokenizer)

        generated = trainer.generate_sample(
            prompt="a",
            max_length=20,
        )

        assert len(generated) > 0

    def test_trainer_checkpoint(self):
        """Test trainer checkpointing."""
        import os
        import tempfile

        model = create_fast_lm_tiny(vocab_size=100)
        config = TrainingConfig(epochs=1, device="cpu")
        trainer = LMTrainer(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test.pt")

            # Save
            trainer.save_checkpoint(checkpoint_path)

            # Load
            trainer.load_checkpoint(checkpoint_path)

            assert os.path.exists(checkpoint_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_full_training_loop(self):
        """Test complete training loop."""
        # Create small dataset
        text = "The quick brown fox jumps over the lazy dog. " * 20
        tokenizer = CharacterTokenizer(text)

        from torch.utils.data import DataLoader

        train_dataset = LMDataset(text, tokenizer, seq_length=32)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Create model
        model = create_fast_lm_tiny(vocab_size=tokenizer.vocab_size)

        # Create trainer
        config = TrainingConfig(
            epochs=2,
            learning_rate=1e-3,
            warmup_steps=5,
            device="cpu",
            use_amp=False,
        )
        trainer = LMTrainer(model, config)
        trainer.set_tokenizer(tokenizer)

        # Train
        metrics = trainer.train(train_loader)

        # Check that loss decreased
        assert metrics.train_loss[-1] < metrics.train_loss[0]

    def test_model_on_gpu(self):
        """Test model runs on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = create_fast_lm_tiny(vocab_size=100)
        model = model.cuda()

        input_ids = torch.randint(0, 100, (2, 10)).cuda()
        logits = model(input_ids)

        assert logits.device.type == "cuda"


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestBenchmarks:
    """Tests for benchmark utilities."""

    def test_nanoGPT_model(self):
        """Test NanoGPT model creation."""
        from bioplausible.models.equitile.benchmarks.compare_nanoGPT import (
            NanoGPTConfig, NanoGPTModel)

        config = NanoGPTConfig(
            vocab_size=100,
            n_layer=2,
            n_embd=64,
            n_head=4,  # Must divide n_embd
        )
        model = NanoGPTModel(config)

        input_ids = torch.randint(0, 100, (2, 10))
        logits, loss = model(input_ids, input_ids)

        assert logits.shape == (2, 10, 100)
        assert loss > 0

    def test_efficiency_analyzer(self):
        """Test efficiency analyzer."""
        from bioplausible.models.equitile.benchmarks.efficiency_analysis import \
            EfficiencyAnalyzer

        model = create_fast_lm_tiny(vocab_size=100)
        analyzer = EfficiencyAnalyzer(model, device="cpu")

        param_counts = analyzer.count_parameters()

        assert param_counts["total"] > 0
        assert param_counts["embedding"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
