"""Load datasets:
    - CIFAR10
    - CIFAR100
    - SVHN
    - MNIST
    - FashionMNIST
"""


from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST


class DATASET:
    def __init__(self, dataset: str, batch_size_train: int, batch_size_val: int):
        """Initialize DATASET class. This facilitates loading training and
        validation datasets.

        Args:
            dataset (str): Dataset name
            batch_size_train (int): Training batch size
            batch_size_val (int): Validation batch size
        """
        self.dataset = dataset
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

        self.transform()
        self.data_loaders()

    def transform(self):
        """Training and validation dataset transformation."""

        # Normalize
        if self.dataset == "CIFAR10":
            normalize = transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            )
        elif self.dataset == "CIFAR100":
            normalize = transforms.Normalize(
                mean=(0.507, 0.487, 0.441), std=(0.267, 0.256, 0.276)
            )
        elif self.dataset == "SVHN":
            normalize = transforms.Normalize(
                mean=(0.4376821, 0.4437697, 0.47280442),
                std=(0.19803012, 0.20101562, 0.19703614),
            )
        elif self.dataset == "MNIST":
            normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        elif self.dataset == "FashionMNIST":
            normalize = transforms.Normalize(mean=(0.2862,), std=(0.3299,))

        # Basic transform
        basic_transform = [transforms.ToTensor(), normalize]

        # Transform for training data
        if "MNIST" in self.dataset:
            transform_train = [transforms.Resize(32)]
        else:
            transform_train = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        self.transform_train = transforms.Compose(transform_train + basic_transform)

        # Transform for validation data
        if "MNIST" in self.dataset:
            self.transform_val = transforms.Compose(
                [transforms.Resize(32)] + basic_transform
            )
        else:
            self.transform_val = transforms.Compose(basic_transform)

    def data_loaders(self):
        """Training and validation datasets."""
        train_data = globals()[self.dataset](
            root="data",
            download=True,
            transform=self.transform_train,
            **self.kwargs("train")
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=2
        )

        # Validation data
        val_data = globals()[self.dataset](
            root="data",
            download=True,
            transform=self.transform_val,
            **self.kwargs("test")
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size_val, shuffle=False, num_workers=2
        )

    def kwargs(self, split: str) -> Dict:
        """Torchvision dataset keyword arguments.

        Args:
            split (str): Data split: "train" or "test"
        """
        return (
            {"split": split} if self.dataset == "SVHN" else {"train": split == "train"}
        )

    def load(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation datasets.

        Returns:
            DataLoader: Training dataloader
            DataLoader: Validation dataloader
        """
        return self.train_loader, self.val_loader

    @property
    def in_channels(self) -> int:
        """Number of data / input channels.

        Returns:
            int: Number of data / input channels
        """
        data_dimensions = self.train_loader.dataset.data.squeeze().shape
        return 1 if len(data_dimensions) == 3 else min(data_dimensions)

    @property
    def num_classes(self) -> int:
        """Number of classes.

        Returns:
            int: Number of classes
        """
        dataset = self.train_loader.dataset
        classes = (
            dataset.classes
            if hasattr(dataset, "classes")
            else np.unique(dataset.labels)
        )
        return len(classes)
