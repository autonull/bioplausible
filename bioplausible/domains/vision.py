"""
Vision Domain Tasks

Standard vision datasets (MNIST, CIFAR, ImageNet, etc.)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bioplausible.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)


class VisionTask(DomainTask):
    """Vision domain tasks (MNIST, CIFAR, ImageNet, etc.)."""

    def __init__(
        self,
        name: str = "mnist",
        dataset_name: str = "mnist",
        data_dir: str = "./data",
        train_transform=None,
        val_transform=None,
        download: bool = True,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.download = download

    @property
    def domain_type(self) -> DomainType:
        return DomainType.VISION

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.VISION,
            description=f"Vision task: {self.dataset_name}",
            default_metrics=["accuracy", "loss"],
            supported_tasks=["classification", "detection", "segmentation"],
            default_batch_size=64,
            default_lr=1e-3,
            requires_spatial=True,
            tags=["vision", "image"],
        )

    def setup(self) -> None:
        """Load vision datasets."""
        if self.train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        if self.val_transform is None:
            self.val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

        # Load dataset based on name
        if self.dataset_name.lower() == "mnist":
            train_ds = datasets.MNIST(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.MNIST(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (1, 28, 28)
            self._output_dim = 10
        elif self.dataset_name.lower() == "cifar10":
            train_ds = datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (3, 32, 32)
            self._output_dim = 10
        elif self.dataset_name.lower() == "cifar100":
            train_ds = datasets.CIFAR100(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.CIFAR100(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (3, 32, 32)
            self._output_dim = 100
        elif self.dataset_name.lower() == "fashion_mnist":
            train_ds = datasets.FashionMNIST(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.FashionMNIST(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (1, 28, 28)
            self._output_dim = 10
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self._train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self._val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        self._test_loader = self._val_loader
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> DataLoader:
        if not self._setup_done:
            self.setup()

        if split == TaskSplit.TRAIN:
            return self._train_loader
        elif split in (TaskSplit.VAL, TaskSplit.TEST):
            return self._val_loader
        else:
            return self._train_loader

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: int | None = None,
    ) -> Metrics:
        model.eval()
        loader = self.get_dataloader(split)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return Metrics(loss=avg_loss, accuracy=accuracy)
