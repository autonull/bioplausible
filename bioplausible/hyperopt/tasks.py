"""
Task Abstraction for Hyperopt and Experiments

Encapsulates data loading, batch generation, and evaluation logic for different tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from bioplausible.datasets import get_lm_dataset, get_vision_dataset
from bioplausible.training.base import BaseTrainer

# Global dataset cache to avoid reloading for every trial
_DATASET_CACHE = {}


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, name: str, device: str = "cpu", quick_mode: bool = False):
        self.name = name
        self.device = device
        self.quick_mode = quick_mode
        self._input_dim = None
        self._output_dim = None

    @abstractmethod
    def setup(self):
        """Load datasets and prepare for training."""
        pass

    @abstractmethod
    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        pass

    @abstractmethod
    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        """Create a trainer specific to this task."""
        pass

    @property
    def input_dim(self) -> Optional[int]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return 'lm', 'vision', or 'rl'."""
        pass

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        return {"loss": loss}


class LMTask(BaseTask):
    """Language Modeling Task (Character level)."""

    def __init__(
        self,
        name: str = "tiny_shakespeare",
        device: str = "cpu",
        quick_mode: bool = False,
        seq_len: int = 64,
    ):
        super().__init__(name, device, quick_mode)
        self.seq_len = seq_len
        self.data_train = None
        self.data_val = None

    @property
    def task_type(self) -> str:
        return "lm"

    def setup(self):
        print(f"Loading LM dataset: {self.name}...")
        try:
            dataset = get_lm_dataset(self.name, seq_len=self.seq_len)
            data = dataset.data
            self._output_dim = dataset.vocab_size
            self._input_dim = None  # Uses embeddings

            # Split train/val
            n = int(0.9 * len(data))
            self.data_train = data[:n]
            self.data_val = data[n:]
            print(
                f"Dataset ready: {len(self.data_train)} train, "
                f"{len(self.data_val)} val tokens"
            )
        except Exception as e:
            print(f"Failed to load dataset {self.name}: {e}")
            raise

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_train is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        data = self.data_train if split == "train" else self.data_val
        idx = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + self.seq_len] for i in idx]).to(self.device)
        y = torch.stack([data[i + self.seq_len] for i in idx]).to(self.device)
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer

        return SupervisedTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> Dict[str, float]:
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        perplexity = np.exp(min(loss, 10))
        return {"loss": loss, "accuracy": acc, "perplexity": perplexity}


class VisionTask(BaseTask):
    """Vision Task (MNIST, CIFAR-10)."""

    def __init__(
        self,
        name: str = "mnist",
        device: str = "cpu",
        quick_mode: bool = False,
        included_classes: Optional[list] = None,
        augment: bool = False,
    ):
        super().__init__(name, device, quick_mode)
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.included_classes = included_classes
        self.augment = augment

    @property
    def task_type(self) -> str:
        return "vision"

    def setup(self):
        # Check cache first
        cache_key = (
            self.name,
            str(self.device),
            self.quick_mode,
            tuple(self.included_classes) if self.included_classes else None,
        )
        if cache_key in _DATASET_CACHE:
            cached = _DATASET_CACHE[cache_key]
            self.train_x = cached["train_x"]
            self.train_y = cached["train_y"]
            self.val_x = cached["val_x"]
            self.val_y = cached["val_y"]
            self._output_dim = cached["output_dim"]
            self._input_dim = cached["input_dim"]
            print(f"Using cached Vision dataset: {self.name}")
            return

        print(f"Loading Vision dataset: {self.name}...")
        try:
            dataset = get_vision_dataset(
                self.name,
                train=True,
                flatten=False,
                included_classes=self.included_classes,
                augment=self.augment,
            )
            test_dataset = get_vision_dataset(
                self.name,
                train=False,
                flatten=False,
                included_classes=self.included_classes,
            )

            # Optimized bulk loading - ONLY if not a Subset
            use_bulk = (
                self.included_classes is None
                and hasattr(dataset, "data")
                and hasattr(dataset, "targets")
            )

            if use_bulk:
                # MNIST/CIFAR style
                raw_x = dataset.data
                raw_y = dataset.targets

                # Handle list targets (CIFAR)
                if isinstance(raw_y, list):
                    raw_y = torch.tensor(raw_y)

                # Handle numpy data (CIFAR)
                if isinstance(raw_x, np.ndarray):
                    raw_x = torch.from_numpy(raw_x)

                # Preprocess X in bulk
                # 1. Convert to float and scale to [0, 1]
                if raw_x.dtype == torch.uint8 or raw_x.dtype == np.uint8:
                    raw_x = raw_x.float() / 255.0

                # 2. Handle dimensions (H, W) -> (1, H, W) or (H, W, C)
                if raw_x.dim() == 3:  # (N, H, W) e.g. MNIST
                    raw_x = raw_x.unsqueeze(1)
                elif raw_x.dim() == 4:  # (N, H, W, C) e.g. CIFAR
                    raw_x = raw_x.permute(0, 3, 1, 2)

                # 3. Normalize
                raw_x = (raw_x - 0.5) / 0.5

                self.train_x = raw_x.to(self.device)
                if not isinstance(raw_y, torch.Tensor):
                    raw_y = torch.tensor(raw_y)
                self.train_y = raw_y.to(self.device)
            else:
                # Fallback for generic datasets or Subsets (slow but safe)
                print("Using fallback loading due to Subset or missing attributes...")
                train_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=512, shuffle=False
                )
                xs, ys = [], []
                for x, y in train_loader:
                    xs.append(x)
                    ys.append(y)
                self.train_x = torch.cat(xs).to(self.device)
                self.train_y = torch.cat(ys).to(self.device)

            val_size = 1000 if self.quick_mode else 5000

            # Similar optimization for validation set
            use_bulk_val = (
                self.included_classes is None
                and hasattr(test_dataset, "data")
                and hasattr(test_dataset, "targets")
            )

            if use_bulk_val:
                raw_x = test_dataset.data
                raw_y = test_dataset.targets

                if isinstance(raw_y, list):
                    raw_y = torch.tensor(raw_y)
                if isinstance(raw_x, np.ndarray):
                    raw_x = torch.from_numpy(raw_x)

                if raw_x.dtype == torch.uint8 or raw_x.dtype == np.uint8:
                    raw_x = raw_x.float() / 255.0

                if raw_x.dim() == 3:
                    raw_x = raw_x.unsqueeze(1)
                elif raw_x.dim() == 4:
                    raw_x = raw_x.permute(0, 3, 1, 2)

                raw_x = (raw_x - 0.5) / 0.5

                # Slice first
                self.val_x = raw_x[: min(len(raw_x), val_size)].to(self.device)
                self.val_y = raw_y[: min(len(raw_y), val_size)].to(self.device)
            else:
                # Fallback
                val_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=512, shuffle=False
                )
                xs, ys = [], []
                total = 0
                for x, y in val_loader:
                    xs.append(x)
                    ys.append(y)
                    total += x.size(0)
                    if total >= val_size:
                        break

                if xs:
                    self.val_x = torch.cat(xs).to(self.device)
                    self.val_y = torch.cat(ys).to(self.device)
                    # Trim
                    self.val_x = self.val_x[:val_size]
                    self.val_y = self.val_y[:val_size]
                else:
                    # Empty dataset?
                    self.val_x = torch.empty(0).to(self.device)
                    self.val_y = torch.empty(0).to(self.device)

            if self.name == "mnist":
                self._output_dim = 10
                self._input_dim = 784
            elif self.name == "cifar10":
                self._output_dim = 10
                self._input_dim = 3072
            else:
                self._output_dim = 10
                self._input_dim = 784 if "mnist" in self.name else 3072

            # Cache for future trials
            _DATASET_CACHE[cache_key] = {
                "train_x": self.train_x,
                "train_y": self.train_y,
                "val_x": self.val_x,
                "val_y": self.val_y,
                "output_dim": self._output_dim,
                "input_dim": self._input_dim,
            }
            print(f"Cached dataset for future trials")
        except Exception as e:
            print(f"Failed to load dataset {self.name}: {e}")
            raise

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_x is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        if split == "train":
            dataset_x, dataset_y = self.train_x, self.train_y
        else:
            dataset_x, dataset_y = self.val_x, self.val_y

        idx = torch.randint(0, len(dataset_x), (batch_size,))
        x = dataset_x[idx]
        y = dataset_y[idx]
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer

        return SupervisedTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> Dict[str, float]:
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc, "perplexity": 0.0}


class RLTask(BaseTask):
    """Reinforcement Learning Task (CartPole)."""

    def __init__(
        self, name: str = "cartpole", device: str = "cpu", quick_mode: bool = False
    ):
        super().__init__(name, device, quick_mode)
        self.env_name = "CartPole-v1" if name == "cartpole" else name
        self.env = None

    @property
    def task_type(self) -> str:
        return "rl"

    def setup(self):
        import gymnasium as gym

        try:
            self.env = gym.make(self.env_name)
            self._output_dim = self.env.action_space.n
            self._input_dim = self.env.observation_space.shape[0]
        except Exception as e:
            print(f"Failed to load env {self.env_name}: {e}")
            raise

    def get_batch(self, split: str = "train", batch_size: int = 32):
        raise NotImplementedError(
            "RL Task does not support get_batch directly, use RLTrainer"
        )

    def create_trainer(self, model, **kwargs):
        from bioplausible.training.rl import RLTrainer

        # Filter relevant args for RLTrainer
        rl_args = {}
        valid_keys = ["episodes", "lr", "gamma", "max_steps", "tracker"]
        for k in valid_keys:
            if k in kwargs:
                rl_args[k] = kwargs[k]

        return RLTrainer(model, self.env_name, device=self.device, **rl_args)


def create_task(
    task_name: str, device: str = "cpu", quick_mode: bool = False
) -> BaseTask:
    """Factory function for tasks. Uses heuristics to map string names to Task classes."""
    if task_name == "char_ngram":
        from bioplausible.tasks.lm.char_ngram import CharNGramTask
        return CharNGramTask(name=task_name, device=device, quick_mode=quick_mode)
    
    if task_name == "pendulum":
        from bioplausible.tasks.rl.pendulum import PendulumTask
        return PendulumTask(name=task_name, device=device, quick_mode=quick_mode)
        
    if task_name in ["shakespeare", "tiny_shakespeare"]:
        return LMTask(task_name, device, quick_mode)

    # Check for Split/Continual Learning Tasks e.g. "mnist_01"
    included_classes = None
    base_name = task_name

    # Parse "mnist_01" -> [0, 1]
    # Match pattern like "mnist_01" or "cifar_0_1_2"
    if "_" in task_name and any(c.isdigit() for c in task_name):
        # Try to extract digits
        parts = task_name.split("_")
        digits = []
        clean_name_parts = []
        for p in parts:
            if p.isdigit():
                # "01" -> 0, 1
                for d in p:
                    digits.append(int(d))
            elif p == "split":
                continue  # ignore 'split' keyword
            else:
                clean_name_parts.append(p)

        if digits:
            included_classes = sorted(list(set(digits)))
            base_name = "_".join(clean_name_parts)
            # Handle special case where base name might be empty or partial
            if "mnist" in task_name and "mnist" not in base_name:
                base_name = "mnist"
            elif "cifar" in task_name and "cifar" not in base_name:
                base_name = "cifar10"

    if (
        "vision" in base_name
        or "mnist" in base_name
        or "cifar" in base_name
        or "fashion" in base_name
    ):
        # Normalize name
        if "cifar" in base_name:
            name = "cifar10"
        elif "fashion" in base_name:
            name = "fashion_mnist"
        elif "kmnist" in base_name or "kuzushiji" in base_name:
            name = "kmnist"
        else:
            name = "mnist"
        return VisionTask(name, device, quick_mode, included_classes=included_classes)
    elif task_name in ["cartpole", "rl"]:
        return RLTask("cartpole", device, quick_mode)
    else:
        # Default to LM
        print(f"Warning: Unknown task '{task_name}', defaulting to tiny_shakespeare LM")
        return LMTask("tiny_shakespeare", device, quick_mode)
