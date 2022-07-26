"""Classify CIFAR10 dataset with ResNet18 model in PyTorch.
"""


import argparse
import logging
import os
from typing import Any, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from datasets import CIFAR10
from models import ResNet18


def initialize_logger():
    """Initialize logger.
    """
    logger = logging.getLogger('Classify CIFAR10 with ResNEt18')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join('log.txt'))
    logger.addHandler(fh)

    return logger

# Logger
logger = initialize_logger()


def get_args() -> argparse.Namespace:
    """This function parses the command-line arguments and returns necessary
    parameter values.

    Returns:
        argparse.Namespace: Required variables to classify CIFAR10 dataset with
                            ResNet18 model in PyTorch:
            learning_rate (float): Learning rate. Default 0.1
            num_epochs (int): Number of epochs. Default 200
            batch_size_train (int): Training batch size
            batch_size_val (int): Validation batch size
            resume (bool): Resume training from checkpoint. Default False
    """

    parser = argparse.ArgumentParser(
        'Classify CIFAR10 dataset with ResNet18 model in PyTorch.'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=0.1,
        help='Learning rate',
    )
    parser.add_argument(
        '-n',
        '--num_epochs',
        type=int,
        default=200,
        help='Number of epochs',
    )
    parser.add_argument(
        '-bt',
        '--batch_size_train',
        type=int,
        default=128,
        help='Training batch size',
    )
    parser.add_argument(
        '-bv',
        '--batch_size_val',
        type=int,
        default=100,
        help='Validation batch size',
    )
    parser.add_argument(
        '-r',
        '--resume',
        action='store_true',
        help='Resume training from checkpoint',
    )
    return parser.parse_args()


def main(
    learning_rate: float,
    num_epochs: int,
    batch_size_train: int,
    batch_size_val: int,
    resume: bool,
) -> None:
    """Classify CIFAR10 dataset with ResNet18 model in PyTorch.

    Args:
        learning_rate (float): Learning rate
        num_epochs (int): Number of epochs
        batch_size_train (int): Training batch size
        batch_size_val (int): Validation batch size
        resume (bool): Resume training from checkpoint
    """

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')

    # Dataset
    train_loader, val_loader = CIFAR10(batch_size_train, batch_size_val)
    logger.info('Training and validation datasets are loaded.')

    # Model
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    logger.info('Model loaded.')

    # Resume training
    net, best_acc, start_epoch = resume_training(net, resume)
    logger.info(f'Initial best accuracy: {best_acc}')
    logger.info(f'starting epoch: {start_epoch}')

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Classify
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train(epoch, net, train_loader, device, optimizer, criterion)
        val(epoch, net, val_loader, device, criterion, best_acc)
        scheduler.step()


def resume_training(net: Any, resume: bool) -> Tuple[Any, float, int]:
    """Resume training.

    Args:
        net (Any): Model to train
        resume (bool): Resume training from checkpoint

    Returns:
        Any: Checkpoint model
        float: Initial best accuray
        int: Starting epoch
    """

    # Initialize best accuracy and starting epoch
    best_acc = 0
    start_epoch = 0

    # If resuming training
    if resume:
        # Load checkpoint
        try:
            checkpoint = torch.load('checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            logger.info('Resuming model from checkpoint.')

        except FileNotFoundError:
            logger.error('Checkpoint path is not present!')

    return net, best_acc, start_epoch


def train(
    epoch: int,
    net: Any,
    train_loader: DataLoader,
    device: str,
    optimizer: Any,
    criterion: Any,
):
    """Training.

    Args:
        epoch (int): Epoch index
        net (Any): Model
        train_loader (DataLoader): Training dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (Any): Optimizer. Ex. SGD
        criterion (Any): Loss function to optimize. Ex. CrossEntropyLoss
    """

    logger.info(f'Training epoch: {epoch}')

    # Initialize
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # Train
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    logger.info(
        f'Training epoch: {epoch} | ' + 
        f'Loss: {train_loss / (batch_idx + 1)} | ' + 
        f'Acc: {100 * correct / total}%%, ({correct}/{total})'
    )


def val(
    epoch: int,
    net: Any,
    val_loader: DataLoader,
    device: str,
    criterion: Any,
    best_acc: float,
):
    """Validation.

    Args:
        epoch (int): Epoch index
        net (Any): Model
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (Any): Optimizer. Ex. SGD
        criterion (Any): Loss function to optimize. Ex. CrossEntropyLoss
    """

    logger.info(f'Validation epoch: {epoch}')

    # Initialize
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    # Validation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info(
        f'Validation epoch: {epoch} | ' + 
        f'Loss: {val_loss / (batch_idx + 1)} | ' + 
        f'Acc: {100 * correct / total}%%, ({correct}/{total})'
    )

    # Save checkpoint
    acc = 100 * correct / total
    if acc >= best_acc:
        logger.info(f'Saving checkpoint from epoch {epoch}.')
        logger.info(f'Best accuracy: was {best_acc}, now: {acc}.')

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == '__main__':
    # Get command-line arguments
    args = get_args()

    # Classify CIFAR10 dataset with ResNet18 model in PyTorch
    main(**args.__dict__)
