"""Train <model> with <dataset> in PyTorch.
<model>:
    - ResNet18

<dataset>:
    - CIFAR10
"""


import argparse
import datetime
import logging
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


def get_args() -> argparse.Namespace:
    """This function parses the command-line arguments and returns necessary
    parameter values.

    Returns:
        argparse.Namespace: Required variables to train <model> with <dataset>
                            in PyTorch:
            model (str): Model to train on. choices are 'ResNet18'. Default
                         'ResNet18'
            dataset (str): Dataset name to classify. choices are 'CIFAR10'.
                           Default 'CIFAR10'
            learning_rate (float): Learning rate. Default 0.1
            num_epochs (int): Number of epochs. Default 200
            batch_size_train (int): Training batch size. Default 128
            batch_size_val (int): Validation batch size. Default 100
            resume (bool): Resume training from checkpoint. Default False
            checkpoint_dir (str): Checkpoint directory. Default ''
            seed (int): Random seed. Default None
    """

    parser = argparse.ArgumentParser('Train <model> with <dataset> in PyTorch.')
    parser.add_argument(
        '-m',
        '--model',
        choices={'ResNet18'},
        default='ResNet18',
        help='Model to train on',
    )
    parser.add_argument(
        '-d',
        '--dataset',
        choices={'CIFAR10'},
        default='CIFAR10',
        help='Dataset to classify',
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
    parser.add_argument(
        '-cd',
        '--checkpoint_dir',
        type=str,
        default='',
        help='Checkpoint directory',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility',
    )
    return parser.parse_args()


def main(
    model: str,
    dataset: str,
    learning_rate: float,
    num_epochs: int,
    batch_size_train: int,
    batch_size_val: int,
    resume: bool,
    checkpoint_dir: str,
    seed: int,
) -> None:
    """Train <model> with <dataset> in PyTorch.

    Args:
        model (str): Model to train on. choices are 'ResNet18'. Default
                     'ResNet18'
        dataset (str): Dataset name to classify. choices are 'CIFAR10'. Default
                       'CIFAR10'
        learning_rate (float): Learning rate
        num_epochs (int): Number of epochs
        batch_size_train (int): Training batch size
        batch_size_val (int): Validation batch size
        resume (bool): Resume training from checkpoint
        checkpoint_dir (str): Checkpoint directory. Default None
        seed (int): Random seed for reproducibility
    """

    # Logger
    global logger
    global log_dir
    logger, log_dir = initialize_logger(model, dataset)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')

    # Set random seed
    set_seed(seed)

    # Model
    net = getattr(__import__('modelzoo'), model)()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    logger.info(f'{model} model loaded.')

    # Dataset
    train_loader, val_loader = getattr(__import__('datazoo'), dataset)(
        batch_size_train, batch_size_val)
    logger.info(f'{dataset} training and validation datasets are loaded.')

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    # Resume training
    net, optimizer, best_acc, start_epoch = resume_training(
        net, optimizer, resume, checkpoint_dir)
    logger.info(f'starting epoch: {start_epoch}')
    logger.info(f'Initial best accuracy: {best_acc}')

    # Train
    epochs, train_loss, train_acc, val_loss, val_acc = train(
        start_epoch,
        num_epochs, net,
        train_loader,
        val_loader,
        device,
        optimizer,
        criterion,
        best_acc,
        scheduler,
    )

    # Plot loss and accuracy over epochs
    plot_loss_accuracy_over_epochs(
        epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        fname=f'train_{model}_{dataset}',
    )


def initialize_logger(model: str, dataset: str) -> Tuple[logging.Logger, str]:
    """Initialize logger.
    """

    # Log directory
    log_dir = datetime.datetime.now().strftime('log-%d_%m_%Y-%H:%M:%S')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Logger
    logger = logging.getLogger(f'Train {model} with {dataset}')
    logger.setLevel(logging.DEBUG)
    logger.handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'log.txt')),
    ]

    return logger, log_dir


def set_seed(seed: int = None):
    """Initiate training with manual random seed for reproducibility.

    Args:
        seed (int): Random seed for reproducibility
    """

    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.warning(
            'Training with manual random seed. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'This may result in unexpected behavior when restarting from '
            'checkpoints.'
        )


def resume_training(
    net: Any,
    optimizer: Any,
    resume: bool,
    checkpoint_dir: str,
) -> Tuple[Any, Any, float, int]:
    """Resume training.

    Args:
        net (Any): Model to train
        optimizer (Any): Optimizer. Ex. SGD
        resume (bool): Resume training from checkpoint
        checkpoint_dir (str): Checkpoint directory. Default None

    Returns:
        Any: Checkpoint model
        Any: Checkpoint optimizer
        float: Initial best accuray
        int: Starting epoch
    """

    # Initialize best accuracy and starting epoch
    best_acc = 0
    start_epoch = 0

    # If resuming training
    if resume:
        # Load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pth')
        try:
            checkpoint = torch.load(checkpoint_path)

            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['acc']

            logger.info(
                f'Resuming training from epoch {start_epoch - 1} checkpoint in '
                f'{checkpoint_dir}.'
            )

        except FileNotFoundError:
            logger.error(f"Checkpoint path '{checkpoint_path}' is not present!")
            exit()

    return net, optimizer, best_acc, start_epoch


def train(
    start_epoch: int,
    num_epochs: int,
    net: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer: Any,
    criterion: Any,
    best_acc: float,
    scheduler: Any,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Train model.

    Args:
        start_epoch (int): Starting epoch
        num_epochs (int): Number of epochs
        net (Any): Model
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (Any): Optimizer. Ex. SGD
        criterion (Any): Loss function to optimize. Ex. CrossEntropyLoss
        best_acc (float): Initial best accuray
        scheduler (Any): Learning rate scheduler for optimizer

    Returns:
        List[int]: Training epochs, from start_epoch to start_epoch + num_epochs
        List[float]: Training losses over epochs
        List[float]: Training accuracies over epochs
        List[float]: Validation losses over epochs
        List[float]: Validation accuracies over epochs
    """
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epochs.append(epoch)

        # Training
        loss, acc = train_epoch(
            epoch, net, train_loader, device, optimizer, criterion)
        train_loss.append(loss)
        train_acc.append(acc)

        # Validation
        loss, acc, best_acc = val_epoch(
            epoch, net, val_loader, device, optimizer, criterion, best_acc)
        val_loss.append(loss)
        val_acc.append(acc)

        scheduler.step()

    return epochs, train_loss, train_acc, val_loss, val_acc


def train_epoch(
    epoch: int,
    net: Any,
    train_loader: DataLoader,
    device: str,
    optimizer: Any,
    criterion: Any,
) -> Tuple[float, float]:
    """Train this epoch.

    Args:
        epoch (int): Epoch index
        net (Any): Model
        train_loader (DataLoader): Training dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (Any): Optimizer. Ex. SGD
        criterion (Any): Loss function to optimize. Ex. CrossEntropyLoss

    Returns:
        float: Training loss this epoch
        float: Training accuracy this epoch
    """

    logger.info(f'Training epoch: {epoch}')

    # Initialize
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # Train
    for _, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Batch train loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = 100 * correct / total

    logger.info(f'Loss: {train_loss} | Acc: {train_acc}%, ({correct}/{total})')
    return train_loss, train_acc


def val_epoch(
    epoch: int,
    net: Any,
    val_loader: DataLoader,
    device: str,
    optimizer: Any,
    criterion: Any,
    best_acc: float,
) -> Tuple[float, float, float]:
    """Validate this epoch.

    Args:
        epoch (int): Epoch index
        net (Any): Model
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (Any): Optimizer. Ex. SGD
        criterion (Any): Loss function to optimize. Ex. CrossEntropyLoss
        best_acc (float): Best accuracy before this epoch

    Returns:
        float: Validation loss this epoch
        float: Validation accuracy this epoch
        float: Best accuracy after this epoch
    """

    logger.info(f'Validation epoch: {epoch}')

    # Initialize
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    # Validation
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # Batch validation loss and accuracy
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    logger.info(f'Loss: {val_loss} | Acc: {val_acc}%, ({correct}/{total})')

    # Save checkpoint
    if val_acc >= best_acc:
        logger.info(f'Saving checkpoint from epoch {epoch}.')
        logger.info(f'Best accuracy: was {best_acc}%, now: {val_acc}%.')

        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'acc': val_acc,
        }
        torch.save(state, os.path.join(log_dir, 'ckpt.pth'))
        best_acc = val_acc

    return val_loss, val_acc, best_acc


def plot_loss_accuracy_over_epochs(
    epochs: List[int],
    train_loss: List[float],
    train_acc: List[float],
    val_loss: List[float],
    val_acc: List[float],
    fname: str,
):
    """Plot training and validation loss and accuracy over epochs.

    Args:
        epochs (List[int]): Training epochs, from start_epoch to start_epoch + 
                            num_epochs
        train_loss (List[float]): Training losses over epochs
        train_acc (List[float]): Training accuracies over epochs
        val_loss (List[float]): Validation losses over epochs
        val_acc (List[float]): Validation accuracies over epochs
        fname (str): Png file name to save the plot
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    ax0.plot(epochs, train_loss, label='Train')
    ax0.plot(epochs, val_loss, label='Validation')
    ax0.grid(True)
    ax0.set_ylabel('Loss')

    ax1.plot(epochs, train_acc, label='Train')
    ax1.plot(epochs, val_acc, label='Validation')
    ax1.grid(True)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')

    lines, labels = ax0.get_legend_handles_labels()
    fig.legend(
        lines, labels, loc='upper right', bbox_to_anchor=(0.7, 0.45, 0.5, 0.5))

    fig.tight_layout()
    plt.savefig(os.path.join(log_dir, fname), bbox_inches='tight')


if __name__ == '__main__':
    # Get command-line arguments
    args = get_args()

    # train <model> with <dataset> in PyTorch
    main(**args.__dict__)
