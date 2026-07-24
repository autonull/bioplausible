"""Tests for the Domain abstraction layer."""

import torch

from bioplausible.domains.base import Batch
from bioplausible.domains.base import DomainSpec
from bioplausible.domains.base import DomainTask
from bioplausible.domains.base import DomainType
from bioplausible.domains.base import Metrics
from bioplausible.domains.base import TaskSplit


def test_metrics_to_dict():
    """Test Metrics serialization."""
    m = Metrics(loss=0.5, accuracy=0.8, perplexity=2.0)
    d = m.to_dict()
    assert d["loss"] == 0.5
    assert d["accuracy"] == 0.8
    assert d["perplexity"] == 2.0


def test_metrics_from_dict():
    """Test Metrics deserialization."""
    d = {"loss": 0.5, "accuracy": 0.8, "perplexity": 2.0, "custom_metric": 0.9}
    m = Metrics.from_dict(d)
    assert m.loss == 0.5
    assert m.accuracy == 0.8
    assert m.perplexity == 2.0
    assert m.custom["custom_metric"] == 0.9


def test_metrics_default():
    """Test Metrics default values."""
    m = Metrics(loss=0.0)
    assert m.accuracy is None
    assert m.perplexity is None


def test_batch_to_device():
    """Test Batch.to() moves tensors to device."""
    inputs = torch.tensor([1, 2, 3])
    targets = torch.tensor([0])
    batch = Batch(inputs=inputs, targets=targets)

    moved = batch.to(torch.device("cpu"))
    assert moved.inputs.device.type == "cpu"
    assert moved.targets.device.type == "cpu"


def test_batch_batch_size():
    """Test batch size property."""
    batch = Batch(inputs=torch.zeros(16, 10), targets=torch.zeros(16))
    assert batch.batch_size == 16


def test_domain_spec_defaults():
    """Test DomainSpec default values."""
    spec = DomainSpec(name="test", domain_type=DomainType.CUSTOM)
    assert spec.default_batch_size == 32
    assert spec.default_lr == 1e-3
    assert spec.requires_sequence is False
    assert spec.requires_spatial is False
    assert spec.tags == []


class TestDomainTask(DomainTask):
    """Concrete test task."""

    @property
    def domain_type(self) -> DomainType:
        return DomainType.CUSTOM

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.CUSTOM,
            description="Test task",
        )

    def setup(self) -> None:
        self._input_dim = 10
        self._output_dim = 2
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit):
        return None

    def evaluate(self, model, split=TaskSplit.VAL, max_batches=None):
        return Metrics(loss=0.0)


def test_domain_task_interface():
    """Test that DomainTask abstract interface works."""
    task = TestDomainTask(name="test_task", device="cpu")
    assert task.name == "test_task"
    assert task.device.type == "cpu"
    assert task.domain_type == DomainType.CUSTOM


def test_domain_task_input_output_dim():
    """Test input/output dimension properties."""
    task = TestDomainTask(name="test_task")
    assert task.input_dim == 10
    assert task.output_dim == 2


def test_domain_task_get_model_kwargs():
    """Test get_model_kwargs."""
    task = TestDomainTask(name="test_task")
    kwargs = task.get_model_kwargs()
    assert kwargs["input_dim"] == 10
    assert kwargs["output_dim"] == 2


def test_vision_task_creation():
    """Test creating a VisionTask."""
    from bioplausible.domains.vision import VisionTask

    task = VisionTask(name="test_vision", dataset_name="mnist", device="cpu")
    assert task.domain_type == DomainType.VISION
    assert task.name == "test_vision"


def test_lm_task_creation():
    """Test creating a LMTask."""
    from bioplausible.domains.lm import LMTask

    task = LMTask(
        name="test_lm", dataset_name="tiny_shakespeare", device="cpu", vocab_size=100
    )
    assert task.domain_type == DomainType.LM
    assert task.name == "test_lm"


def test_rl_task_creation():
    """Test creating an RLTask."""
    from bioplausible.domains.rl import RLTask

    task = RLTask(name="test_rl", env_id="CartPole-v1", device="cpu")
    assert task.domain_type == DomainType.RL
    assert task.name == "test_rl"


def test_domain_registry():
    """Test the domain registry via create_domain_task."""
    from bioplausible.domains import create_domain_task
    from bioplausible.domains import list_domains

    domains = list_domains()
    assert "vision" in domains
    assert "lm" in domains
    assert "rl" in domains

    # Test creating a task
    task = create_domain_task("vision", "mnist", device="cpu")
    assert task is not None


def test_metrics_computation():
    """Test compute_metrics."""
    task = TestDomainTask(name="test")
    outputs = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])

    metrics = task.compute_metrics(outputs, targets, 0.5)
    assert metrics.loss == 0.5
    assert metrics.accuracy == 1.0  # Both predictions correct

    # Test with wrong predictions
    outputs_wrong = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    metrics_wrong = task.compute_metrics(outputs_wrong, targets, 1.0)
    assert metrics_wrong.loss == 1.0
    assert metrics_wrong.accuracy == 0.0
