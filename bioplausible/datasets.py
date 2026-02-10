"""
EqProp-Torch Dataset Utilities

HuggingFace datasets and tokenizers integration for easy LM and vision dataset loading.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

# =============================================================================
# Vision Datasets
# =============================================================================


def get_vision_dataset(
    name: str = "mnist",
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    flatten: bool = False,
    included_classes: Optional[list] = None,
    augment: bool = False,
) -> Dataset:
    """
    Load a vision dataset with standard transforms.

    Args:
        name: Dataset name ('mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'kmnist', 'svhn', 'digits')
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        flatten: If True, flatten images to 1D
        included_classes: List of class indices to include (optional)
        augment: If True, apply data augmentation (RandomCrop, RandomHorizontalFlip) for training.

    Returns:
        PyTorch Dataset

    Example:
        >>> train_data = get_vision_dataset('mnist', train=True)
        >>> test_data = get_vision_dataset('mnist', train=False)
    """

    if name == "digits":
        return _load_sklearn_digits(train, flatten)

    from torchvision import datasets, transforms

    transform = _build_transforms(name, flatten, augment=augment and train)
    dataset_class = _get_dataset_class(name)

    dataset = None
    if name == "svhn":
        # SVHN uses 'split' instead of 'train'
        split = "train" if train else "test"
        dataset = dataset_class(root, split=split, download=download, transform=transform)
    else:
        dataset = dataset_class(root, train=train, download=download, transform=transform)

    if included_classes is not None:
        targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        indices = [i for i, t in enumerate(targets) if t in included_classes]

        from torch.utils.data import Subset
        return Subset(dataset, indices)

    return dataset


def _load_sklearn_digits(train: bool, flatten: bool) -> Dataset:
    """Load sklearn 8x8 digits dataset."""
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError(
            "scikit-learn required for 'digits' dataset. pip install scikit-learn"
        )

    digits = load_digits()
    X = digits.data.astype(np.float32)  # (1797, 64)
    y = digits.target.astype(np.int64)  # (1797,)

    # Normalize to [0, 1] (digits are 0-16)
    X /= 16.0

    # Simple split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_data = X_train if train else X_test
    y_data = y_train if train else y_test

    # Sklearn digits are already flattened (N, 64) by default in .data
    # If not flatten, reshape to (N, 1, 8, 8)
    if not flatten:
        X_data = X_data.reshape(-1, 1, 8, 8)

    return TensorDataset(torch.from_numpy(X_data), torch.from_numpy(y_data))


def _build_transforms(name: str, flatten: bool, augment: bool = False):
    """Build the appropriate transforms for the given dataset."""
    from torchvision import transforms

    transform_list = []

    if augment:
        if name in ["cifar10", "cifar100", "svhn"]:
            transform_list.append(transforms.RandomCrop(32, padding=4))
            transform_list.append(transforms.RandomHorizontalFlip())
        elif name in ["mnist", "fashion_mnist", "kmnist"]:
            # Slight augmentation for MNIST-like
            transform_list.append(transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)))

    transform_list.append(transforms.ToTensor())

    if name in ["mnist", "fashion_mnist", "kmnist", "usps"]:
        # Normalize grayscale to [-1, 1] range
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    elif name in ["cifar10", "cifar100", "svhn"]:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    return transforms.Compose(transform_list)


def _get_dataset_class(name: str) -> type:
    """Get the appropriate dataset class for the given name."""
    from torchvision import datasets

    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "kmnist": datasets.KMNIST,
        "svhn": datasets.SVHN,
        "usps": datasets.USPS,
    }

    if name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(dataset_map.keys())} + ['digits']"
        )

    return dataset_map[name]


# =============================================================================
# Language Modeling Datasets
# =============================================================================


class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, seq_len: int = 128) -> None:
        self.seq_len = seq_len

        # Build vocabulary
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        # Encode text
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

    def decode(self, indices: torch.Tensor) -> str:
        """
        Convert indices back to text.

        Args:
            indices: Tensor of character indices

        Returns:
            Decoded text string
        """
        return "".join(self.idx_to_char[i.item()] for i in indices)


def get_lm_dataset(
    name: str = "tiny_shakespeare",
    seq_len: int = 128,
    split: str = "train",
) -> CharDataset:
    """
    Load a language modeling dataset as a CharDataset.

    Args:
        name: Dataset name ('tiny_shakespeare', 'wikitext-2', 'ptb')
        seq_len: Sequence length for training
        split: 'train', 'validation', or 'test'

    Returns:
        CharDataset with vocab_size attribute

    Example:
        >>> dataset = get_lm_dataset('tiny_shakespeare', seq_len=128)
        >>> print(f"Vocab size: {dataset.vocab_size}")
        >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install datasets"
        )

    # Load dataset
    if name == "tiny_shakespeare":
        # Shakespeare from HuggingFace
        try:
            dataset = load_dataset("tiny_shakespeare")
            # tiny_shakespeare has 'train', 'validation', 'test' splits
            # Each row has a 'text' field containing a line
            split_data = dataset[split]

            # Collect all text from the split
            texts = []
            for item in split_data:
                if isinstance(item["text"], str):
                    texts.append(item["text"])

            text = "\n".join(texts)

            if not text or len(text) == 0:
                raise ValueError("Empty text after loading")

        except Exception as e:
            # Fallback: load from URL
            warnings.warn(f"HuggingFace dataset failed, using fallback: {e}")
            import urllib.request

            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with urllib.request.urlopen(url) as response:
                text = response.read().decode("utf-8")

    elif name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text = "\n".join(dataset[split]["text"])

    elif name == "ptb":
        dataset = load_dataset("ptb_text_only")
        split_name = (
            "train"
            if split == "train"
            else "validation" if split == "validation" else "test"
        )
        text = " ".join(dataset[split_name]["sentence"])

    else:
        raise ValueError(
            f"Unknown LM dataset: {name}. Available: tiny_shakespeare, wikitext-2, ptb"
        )

    # Create and return CharDataset (not DataLoader)
    char_dataset = CharDataset(text, seq_len=seq_len)

    return char_dataset


# =============================================================================
# Utility Functions
# =============================================================================


def create_data_loaders(
    dataset_name: str = "mnist",
    batch_size: int = 64,
    num_workers: int = 0,
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for a vision dataset.

    Args:
        dataset_name: Name of dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        flatten: Flatten images for MLP models

    Returns:
        (train_loader, test_loader)
    """
    train_data = get_vision_dataset(dataset_name, train=True, flatten=flatten)
    test_data = get_vision_dataset(dataset_name, train=False, flatten=flatten)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


__all__ = [
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "create_data_loaders",
]
