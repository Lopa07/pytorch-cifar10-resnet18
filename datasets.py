"""Load datasets:
    - CIFAR10
    - CIFAR100
    - SVHN
"""


from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN


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
        if self.dataset == "CIFAR10":
            normalize = transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )

        elif self.dataset == "CIFAR100":
            normalize = transforms.Normalize(
                mean=(0.507, 0.487, 0.441),
                std=(0.267, 0.256, 0.276),
            )

        elif self.dataset == "SVHN":
            normalize = transforms.Normalize(
                mean=(0.4376821, 0.4437697, 0.47280442),
                std=(0.19803012, 0.20101562, 0.19703614),
            )

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

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
        if self.dataset in {"CIFAR10", "CIFAR100"}:
            return {"train": split == "train"}

        if self.dataset == "SVHN":
            return {"split": split}

    def load(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation datasets.

        Returns:
            DataLoader: Training dataloader
            DataLoader: Validation dataloader
        """
        return self.train_loader, self.val_loader

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
