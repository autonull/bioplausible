"""
Tests for continual learning benchmarks.
"""

import torch
import torch.nn as nn
import pytest
from mep.benchmarks.continual_learning import (
    PermutedMNIST,
    MLP,
    create_mep_optimizer,
    train_epoch,
    evaluate,
    run_permuted_mnist_benchmark,
    TaskResult,
    ContinualLearningResult,
)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model(device):
    """Simple MLP for testing."""
    return MLP(input_dim=64, hidden_dims=(32, 16), num_classes=5).to(device)


@pytest.fixture
def sample_data(device):
    """Sample data for testing."""
    x = torch.randn(16, 64, device=device)
    y = torch.randint(0, 5, (16,), device=device)
    return x, y


class TestPermutedMNIST:
    """Tests for PermutedMNIST benchmark."""
    
    def test_permutations_generated(self):
        """Test that permutations are generated correctly."""
        benchmark = PermutedMNIST(num_tasks=3, seed=42)
        
        assert len(benchmark.permutations) == 3
        for perm in benchmark.permutations:
            assert perm.shape == (784,)
            assert sorted(perm.tolist()) == list(range(784))
    
    def test_permutations_reproducible(self):
        """Test that permutations are reproducible with same seed."""
        b1 = PermutedMNIST(num_tasks=3, seed=123)
        b2 = PermutedMNIST(num_tasks=3, seed=123)
        
        for p1, p2 in zip(b1.permutations, b2.permutations):
            assert torch.equal(p1, p2)
    
    def test_different_seeds_different_permutations(self):
        """Test that different seeds produce different permutations."""
        b1 = PermutedMNIST(num_tasks=3, seed=42)
        b2 = PermutedMNIST(num_tasks=3, seed=999)
        
        # At least one permutation should be different
        different = any(not torch.equal(p1, p2) for p1, p2 in zip(b1.permutations, b2.permutations))
        assert different


class TestMLP:
    """Tests for MLP architecture."""
    
    def test_mlp_forward(self, simple_model, sample_data, device):
        """Test MLP forward pass."""
        x, _ = sample_data
        output = simple_model(x)
        
        assert output.shape == (16, 5)
    
    def test_mlp_custom_architecture(self, device):
        """Test MLP with custom architecture."""
        model = MLP(input_dim=100, hidden_dims=(64, 32, 16), num_classes=10).to(device)
        x = torch.randn(8, 100, device=device)
        output = model(x)
        
        assert output.shape == (8, 10)
    
    def test_mlp_no_dropout(self, device):
        """Test MLP without dropout."""
        model = MLP(input_dim=50, hidden_dims=(32,), num_classes=5, dropout=0.0).to(device)
        x = torch.randn(4, 50, device=device)
        output = model(x)
        
        assert output.shape == (4, 5)


class TestOptimizer:
    """Tests for MEP optimizer creation."""
    
    def test_create_mep_optimizer_ep(self, simple_model):
        """Test creating MEP optimizer in EP mode."""
        optimizer = create_mep_optimizer(simple_model, mode="ep", use_error_feedback=True)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) > 0
    
    def test_create_mep_optimizer_backprop(self, simple_model):
        """Test creating MEP optimizer in backprop mode."""
        optimizer = create_mep_optimizer(simple_model, mode="backprop", use_error_feedback=False)
        
        assert optimizer is not None
    
    def test_error_feedback_enabled(self, simple_model):
        """Test that error feedback is properly configured."""
        optimizer_ef = create_mep_optimizer(simple_model, use_error_feedback=True)
        optimizer_no_ef = create_mep_optimizer(simple_model, use_error_feedback=False)
        
        # Check that feedback strategies are different types
        from mep.optimizers.strategies.feedback import ErrorFeedback, NoFeedback
        assert isinstance(optimizer_ef.feedback, ErrorFeedback)
        assert isinstance(optimizer_no_ef.feedback, NoFeedback)


class TestTrainingLoop:
    """Tests for training and evaluation functions."""
    
    def test_train_epoch(self, simple_model, sample_data, device):
        """Test training for one epoch."""
        x, y = sample_data
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        optimizer = create_mep_optimizer(simple_model, mode="backprop")
        
        initial_params = [p.clone() for p in simple_model.parameters()]
        
        acc = train_epoch(simple_model, optimizer, loader, device, use_ep=False)
        
        # Accuracy should be a valid number
        assert 0 <= acc <= 1
        
        # Parameters should have changed
        changed = any(not torch.equal(p, ip) for p, ip in zip(simple_model.parameters(), initial_params))
        assert changed
    
    def test_evaluate(self, simple_model, sample_data, device):
        """Test evaluation function."""
        x, y = sample_data
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        acc = evaluate(simple_model, loader, device)
        
        assert 0 <= acc <= 1


class TestBenchmark:
    """Tests for benchmark execution."""
    
    def test_run_permuted_mnist_small(self, device):
        """Test running a small Permuted MNIST benchmark."""
        result = run_permuted_mnist_benchmark(
            num_tasks=2,
            epochs_per_task=1,
            lr=0.01,
            use_error_feedback=True,
            mode="backprop",  # Use backprop for faster testing
            device=device,
            seed=42
        )
        
        assert isinstance(result, ContinualLearningResult)
        assert result.num_tasks == 2
        assert len(result.task_results) == 2
        assert 0 <= result.average_accuracy <= 1
        assert result.benchmark_name == "Permuted MNIST"
    
    def test_task_result_structure(self, device):
        """Test that task results have correct structure."""
        result = run_permuted_mnist_benchmark(
            num_tasks=1,
            epochs_per_task=1,
            lr=0.01,
            use_error_feedback=False,
            mode="backprop",
            device=device,
            seed=42
        )
        
        assert len(result.task_results) == 1
        task_result = result.task_results[0]
        
        assert isinstance(task_result, TaskResult)
        assert task_result.task_id == 0
        assert hasattr(task_result, 'train_accuracy')
        assert hasattr(task_result, 'test_accuracy')
        assert hasattr(task_result, 'forgetting')


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_same_seed_same_results(self, device):
        """Test that same seed produces same results."""
        r1 = run_permuted_mnist_benchmark(
            num_tasks=1,
            epochs_per_task=1,
            lr=0.01,
            use_error_feedback=True,
            mode="backprop",
            device=device,
            seed=12345
        )
        
        r2 = run_permuted_mnist_benchmark(
            num_tasks=1,
            epochs_per_task=1,
            lr=0.01,
            use_error_feedback=True,
            mode="backprop",
            device=device,
            seed=12345
        )
        
        # Results should be identical with same seed
        assert r1.average_accuracy == r2.average_accuracy
        assert r1.average_forgetting == r2.average_forgetting
