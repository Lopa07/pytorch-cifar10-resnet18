"""Load CIFAR datasets:
    - CIFAR10
"""


from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


def CIFAR10(
    batch_size_train: int, batch_size_val: int
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR10 training and validation datasets.

    Args:
        batch_size_train (int): Training batch size
        batch_size_val (int): Validation batch size

    Returns:
        DataLoader: Training dataloader
        DataLoader: Validation dataloader
    """

    # CIFAR10 classes are, 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
    # 'horse', 'ship', and 'truck'

    # Training data
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    train_data = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True, num_workers=2
    )

    # Validation data
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    val_data = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_val
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=100, shuffle=False, num_workers=2
    )

    return train_loader, val_loader
