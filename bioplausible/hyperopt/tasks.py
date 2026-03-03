"""
Task Abstraction for Hyperopt and Experiments

Encapsulates data loading, batch generation, and evaluation logic for different tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

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
            # Quick Mode Truncation
            if self.quick_mode:
                n_quick = min(len(self.data_train), 1000)
                self.data_train = self.data_train[:n_quick].clone()
                self.data_val = self.data_val[: min(len(self.data_val), 1000)].clone()

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

        # Avoid duplicate device argument
        if "device" in kwargs:
            del kwargs["device"]

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
        fold: Optional[int] = None,
        num_folds: int = 5,
        data_fraction: Optional[float] = None,
    ):
        super().__init__(name, device, quick_mode)
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.included_classes = included_classes
        self.augment = augment
        self.fold = fold
        self.num_folds = num_folds
        self.data_fraction = data_fraction

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
            self.fold,
            self.num_folds,
            self.data_fraction,
        )
        if cache_key in _DATASET_CACHE:
            cached = _DATASET_CACHE[cache_key]
            self.train_x = cached["train_x"]
            self.train_y = cached["train_y"]
            self.val_x = cached["val_x"]
            self.val_y = cached["val_y"]
            self._output_dim = cached["output_dim"]
            self._input_dim = cached["input_dim"]
            print(
                f"Using cached Vision dataset: {self.name} "
                f"(Fold={self.fold}, Frac={self.data_fraction})"
            )
            return

        print(
            f"Loading Vision dataset: {self.name} (Fold={self.fold}, Frac={self.data_fraction})..."
        )
        try:
            # We first load the full training set (and test set)
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

            # --- Preprocessing and Loading Data to Tensors ---
            # Helper to process dataset into tensors
            def load_to_tensor(ds):
                # Check for various dataset formats
                has_data_targets = hasattr(ds, "data") and hasattr(ds, "targets")
                has_data_labels = hasattr(ds, "data") and hasattr(ds, "labels")  # SVHN
                has_tensors = hasattr(ds, "tensors")  # TensorDataset

                use_bulk = self.included_classes is None and (
                    has_data_targets or has_data_labels or has_tensors
                )

                if use_bulk:
                    if has_tensors:
                        raw_x, raw_y = ds.tensors
                    else:
                        raw_x = ds.data
                        raw_y = ds.targets if has_data_targets else ds.labels

                    if isinstance(raw_y, list):
                        raw_y = torch.tensor(raw_y)
                    if isinstance(raw_x, np.ndarray):
                        raw_x = torch.from_numpy(raw_x)

                    # Preprocess X in bulk
                    if raw_x.dtype == torch.uint8 or raw_x.dtype == np.uint8:
                        raw_x = raw_x.float() / 255.0
                    elif raw_x.dtype in [
                        torch.float32,
                        torch.float64,
                        np.float32,
                        np.float64,
                    ]:
                        # Check if data is unscaled (0-255) despite being float
                        if raw_x.max() > 1.0:
                            raw_x = raw_x / 255.0

                    if raw_x.dim() == 3:  # (N, H, W)
                        raw_x = raw_x.unsqueeze(1)
                    elif raw_x.dim() == 4:  # (N, H, W, C)
                        # Assume NCHW if channels are last (e.g. from NumPy),
                        # but only if not already NCHW. Heuristic: Check if
                        # channel dim is small (1 or 3) and not already in dim 1
                        is_nhwc = raw_x.shape[3] in [1, 3] and raw_x.shape[1] not in [
                            1,
                            3,
                        ]
                        # Also skip permutation if coming from TensorDataset (likely already NCHW)
                        if is_nhwc and not has_tensors:
                            raw_x = raw_x.permute(0, 3, 1, 2).contiguous()

                    # Normalize
                    raw_x = (raw_x - 0.5) / 0.5

                    # Ensure contiguous memory layout (critical for SVHN and others)
                    if not raw_x.is_contiguous():
                        raw_x = raw_x.contiguous()

                    if not isinstance(raw_y, torch.Tensor):
                        raw_y = torch.tensor(raw_y)

                    return raw_x.to(self.device), raw_y.to(self.device)
                else:
                    # Fallback
                    loader = torch.utils.data.DataLoader(
                        ds, batch_size=512, shuffle=False
                    )
                    xs, ys = [], []
                    for x, y in loader:
                        xs.append(x)
                        ys.append(y)
                    return torch.cat(xs).to(self.device), torch.cat(ys).to(self.device)

            full_train_x, full_train_y = load_to_tensor(dataset)
            full_test_x, full_test_y = load_to_tensor(test_dataset)

            # --- Splitting Logic ---
            if self.fold is not None:
                # K-Fold Cross Validation
                # We merge train and test (or just use train?)
                # Standard practice: Use Training Set for CV, keep Test Set hidden/separate.
                # Here we will perform CV on the TRAINING set.

                kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
                splits = list(kf.split(full_train_x))
                train_idx, val_idx = splits[self.fold]

                self.train_x = full_train_x[train_idx]
                self.train_y = full_train_y[train_idx]
                self.val_x = full_train_x[val_idx]
                self.val_y = full_train_y[val_idx]

                # We can also use full_test_x for final test if needed, but
                # for CV usually Val is the metric. However, our system logs
                # `accuracy` (val) and `final_loss` (train).

            else:
                # Standard Split
                self.train_x = full_train_x
                self.train_y = full_train_y

            # Quick Mode Truncation
            if self.quick_mode:
                n_quick = min(len(self.train_x), 1000)
                self.train_x = self.train_x[:n_quick].clone()
                self.train_y = self.train_y[:n_quick].clone()

                # Apply Data Fraction (Low Data Regime)
                if self.data_fraction is not None and 0.0 < self.data_fraction < 1.0:
                    n_samples = int(len(self.train_x) * self.data_fraction)
                    # Shuffle
                    perm = torch.randperm(len(self.train_x))[:n_samples]
                    self.train_x = self.train_x[perm]
                    self.train_y = self.train_y[perm]
                    print(
                        f"Subsampled dataset to {n_samples} samples ({self.data_fraction:.0%})"
                    )

                # Validation Set (Subset of Test Set for speed if quick_mode)
                val_size = 1000
                self.val_x = full_test_x[: min(len(full_test_x), val_size)]
                self.val_y = full_test_y[: min(len(full_test_y), val_size)]
            elif self.fold is None:
                # Standard Mode (Full Test Set for Validation)
                self.val_x = full_test_x
                self.val_y = full_test_y

            # Metadata
            if self.name == "mnist":
                self._output_dim = 10
                self._input_dim = (1, 28, 28)
            elif self.name == "cifar10":
                self._output_dim = 10
                self._input_dim = (3, 32, 32)
            else:
                # Fallback heuristics
                if self.train_x.dim() > 2:
                    self._input_dim = tuple(self.train_x.shape[1:])
                else:
                    self._input_dim = self.train_x.shape[1]

                if self.included_classes:
                    self._output_dim = len(self.included_classes)
                else:
                    self._output_dim = int(self.train_y.max().item() + 1)

            # Cache for future trials
            _DATASET_CACHE[cache_key] = {
                "train_x": self.train_x,
                "train_y": self.train_y,
                "val_x": self.val_x,
                "val_y": self.val_y,
                "output_dim": self._output_dim,
                "input_dim": self._input_dim,
            }
            print("Cached dataset for future trials")
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

        if len(dataset_x) == 0:
            return torch.empty(0).to(self.device), torch.empty(0).to(self.device)

        idx = torch.randint(0, len(dataset_x), (batch_size,))
        x = dataset_x[idx]
        y = dataset_y[idx]
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer

        if "device" in kwargs:
            del kwargs["device"]

        return SupervisedTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> Dict[str, float]:
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc, "perplexity": 0.0}


class CharNGramTask(BaseTask):
    """Synthetic task: Predict next character from previous N chars.

    Dataset: Deterministic repeating patterns or simple probabilistic grammar.
    Input: [B, SeqLen] (indices)
    Output: [B, VocabSize] (logits for last char)
    """

    def __init__(
        self,
        name: str = "char_ngram",
        device: str = "cpu",
        quick_mode: bool = False,
        vocab_size: int = 27,
        context_len: int = 3,
    ):
        super().__init__(name, device, quick_mode)
        self.vocab_size = vocab_size
        self.context_len = context_len
        self._input_dim = context_len # Since we flatten
        self._output_dim = vocab_size
        self.pattern = torch.arange(vocab_size)

    @property
    def task_type(self) -> str:
        return "lm"

    def setup(self):
        pass

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.vocab_size - self.context_len, (batch_size,))
        x_list = []
        y_list = []
        for s in starts:
            seq = (
                torch.arange(s.item(), s.item() + self.context_len + 1)
            ) % self.vocab_size
            x_list.append(seq[:-1])
            y_list.append(seq[-1])
        x = torch.stack(x_list).to(self.device).float().unsqueeze(2) # [B, L, 1]
        x = x.view(x.size(0), -1) # Flatten [B, L*1] -> [B, L]

        y = torch.stack(y_list).to(self.device).long()
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer

        if "device" in kwargs:
            del kwargs["device"]

        return SupervisedTrainer(model, self, device=self.device, **kwargs)


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

            # Determine Output Dim (Action Space)
            if hasattr(self.env.action_space, "n"):
                self._output_dim = self.env.action_space.n  # Discrete
            else:
                self._output_dim = self.env.action_space.shape[0]  # Continuous (Box)

            # Determine Input Dim (Observation Space)
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

        # Map batches_per_epoch to episodes_per_epoch for RL
        if "batches_per_epoch" in kwargs and "episodes_per_epoch" not in kwargs:
            kwargs["episodes_per_epoch"] = kwargs["batches_per_epoch"]

        valid_keys = [
            "episodes",
            "lr",
            "gamma",
            "max_steps",
            "tracker",
            "episodes_per_epoch",
        ]
        for k in valid_keys:
            if k in kwargs:
                rl_args[k] = kwargs[k]

        return RLTrainer(model, self.env_name, device=self.device, **rl_args)


def create_task(
    task_name: str, device: str = "cpu", quick_mode: bool = False, **kwargs
) -> BaseTask:
    """Factory function for tasks. Uses heuristics to map string names to Task classes."""
    if task_name == "char_ngram":
        return CharNGramTask(name=task_name, device=device, quick_mode=quick_mode)

    # RL Tasks
    if task_name == "pendulum":
        return RLTask("Pendulum-v1", device, quick_mode)
    if task_name == "acrobot":
        return RLTask("Acrobot-v1", device, quick_mode)
    if task_name in ["cartpole", "rl"]:
        return RLTask("CartPole-v1", device, quick_mode)

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
        or "digits" in base_name
        or "usps" in base_name
        or "svhn" in base_name
    ):
        # Normalize name
        name = base_name
        if "cifar" in base_name:
            if "100" in base_name:
                name = "cifar100"
            else:
                name = "cifar10"
        elif "fashion" in base_name:
            name = "fashion_mnist"
        elif "kmnist" in base_name or "kuzushiji" in base_name:
            name = "kmnist"
        elif "digits" in base_name:
            name = "digits"
        elif "usps" in base_name:
            name = "usps"
        elif "svhn" in base_name:
            name = "svhn"
        elif "mnist" in base_name:
            name = "mnist"

        # Extract fold and data_fraction from kwargs
        fold = kwargs.get("fold")
        data_fraction = kwargs.get("data_fraction")

        return VisionTask(
            name,
            device,
            quick_mode,
            included_classes=included_classes,
            fold=fold,
            data_fraction=data_fraction,
        )

    if base_name in ["cora", "pubmed", "citeseer"]:
        from bioplausible.hyperopt.graph_task import GraphTask
        return GraphTask(base_name, device, quick_mode)
        
    if base_name in ["breast_cancer", "california_housing"]:
        from bioplausible.hyperopt.tabular_task import TabularTask
        return TabularTask(base_name, device, quick_mode)

    # Default to LM
    print(f"Warning: Unknown task '{task_name}', defaulting to tiny_shakespeare LM")
    return LMTask("tiny_shakespeare", device, quick_mode)
