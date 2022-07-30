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
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader


def get_args() -> argparse.Namespace:
    """This function parses the command-line arguments and returns necessary
    parameter values.

    Returns:
        argparse.Namespace: Configuration file with required parameters to train
                            <model> with <dataset> in PyTorch:
            config_file (str): Configuration yaml file path. Default
                               "config.example.yml"
    """

    parser = argparse.ArgumentParser("Train <model> with <dataset> in PyTorch.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.example.yml",
        help="Configuration yaml file path",
    )
    return parser.parse_args()


def main(config_file: str) -> None:
    """Train <model> with <dataset> in PyTorch.

    Args:
        config_file (str): Configuration yaml file path. Default
                           'config.example.yml'
    """

    # Load configuration
    config = yaml.safe_load(Path(config_file).read_text())

    # Model and dataset names
    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]

    # Logger
    global logger
    global log_dir
    logger, log_dir = initialize_logger(config_file, model_name, dataset_name)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Set random seed
    set_seed(config["seed"])

    # Model
    model = getattr(__import__("modelzoo"), model_name)()
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    logger.info(f"{model_name} model loaded.")

    # Dataset
    train_loader, val_loader = getattr(__import__("datazoo"), dataset_name)(
        config["training"]["batch_size"]["train"],
        config["training"]["batch_size"]["val"],
    )
    logger.info(f"{dataset_name} training and validation datasets are loaded.")

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config["training"]["optimizer"], model.parameters())
    scheduler = get_scheduler(
        config["training"]["scheduler"], optimizer, config["training"]["num_epochs"]
    )

    # Resume training
    model, optimizer, best_acc, start_epoch = resume_training(config, model, optimizer)
    logger.info(f"Starting epoch: {start_epoch}")
    logger.info(f"Initial best accuracy: {best_acc}")

    # Train
    epochs, train_loss, train_acc, val_loss, val_acc = train(
        start_epoch,
        config["training"]["num_epochs"],
        model,
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
        fname=f"train_{model_name}_{dataset_name}",
    )


def initialize_logger(
    config_file: str, model: str, dataset: str
) -> Tuple[logging.Logger, str]:
    """Initialize logger.

    Args:
        config_file (str): Configuration yaml file path
        model (str): Model name to train
        dataset (str): Dataset name to classify

    Returns:
        logging.Logger: Logger
        str: Log directory
    """

    # Log directory
    log_dir = datetime.datetime.now().strftime("log-%d_%m_%Y-%H:%M:%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save configuration file for this run
    shutil.copy2(config_file, os.path.join(log_dir, "config.yml"))

    # Logger
    logger = logging.getLogger(f"Train {model} with {dataset}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "log.txt")),
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
            "Training with manual random seed. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "This may result in unexpected behavior when restarting from "
            "checkpoints."
        )


def get_optimizer(
    optimizer_config: Dict, parameters: Iterator[Parameter]
) -> optim.Optimizer:
    """Optimizer.

    Args:
        optimizer_config (Dict): Dictionary with optimizer configurations.
            optimizer:
                name (str): 'sgd', 'nesterov_sgd', 'rmsprop', 'adagrad', or
                            'adam'
                learning_rate (float): Initial learning rate
                momentum (float): Momentum factor
                weight_decay (float): Weight decay (L2 penalty)
        parameters (Iterator[Parameter]): Model parameters

    Returns:
        optim.Optimizer: Optimizer
    """

    # Optimizer name, learning rate, momentum, and weight decay
    optimizer_name = optimizer_config["name"]
    lr = optimizer_config["learning_rate"]
    momentum = optimizer_config["momentum"]
    wd = optimizer_config["weight_decay"]

    # Optimizer
    if optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)

    if optimizer_name == "nesterov_sgd":
        return optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True
        )

    if optimizer_name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=wd)

    if optimizer_name == "adagrad":
        return optim.Adagrad(parameters, lr=lr, weight_decay=wd)

    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=wd)


def get_scheduler(
    scheduler_config: Dict, optimizer: optim.Optimizer, num_epochs: int
) -> _LRScheduler:
    """Learning rate scheduler.

    Args:
        scheduler_config (Dict): Dictionary with scheduler configurations.
            scheduler:
                name (str): 'constant', 'step', 'multistep', 'exponential', or
                            'cosine'
                kwargs (Dict): Scheduler specific key word arguments
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        num_epochs (int): Number of epochs

    Returns:
        _LRScheduler: Learning rate scheduler for optimizer
    """

    # Scheduler name and kwargs
    scheduler_name = scheduler_config["name"]
    kwargs = scheduler_config["kwargs"]

    # Scheduler
    if scheduler_name == "constant":
        return optim.lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)

    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, **kwargs)

    if scheduler_name == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 120, 200], gamma=0.1
        )

    if scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, (1e-3) ** (1 / num_epochs), **kwargs
        )

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1
        )


def resume_training(
    config: Dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[nn.Module, optim.Optimizer, float, int]:
    """Resume training.

    Args:
        config (Dict): Configuration with required parameters to train <model>
                       with <dataset>
        model (nn.Module): Model to train
        optimizer (optim.Optimizer): Optimizer. Ex. SGD

    Returns:
        nn.Module: Checkpoint model
        optim.Optimizer: Checkpoint optimizer
        float: Initial best accuray
        int: Starting epoch
    """

    # Initialize best accuracy and starting epoch
    best_acc = 0
    start_epoch = 0

    # Resume from checkpoint
    resume = config["training"]["resume"]["from_checkpoint"]
    if not resume:
        return model, optimizer, best_acc, start_epoch

    # Checkpoint directory
    checkpoint_dir = config["training"]["resume"]["checkpoint_dir"]

    # Check if basic configurations are same with the checkpoint and can resume
    # training
    config_checkpoint = Path(checkpoint_dir) / "config.yml"
    config_checkpoint = yaml.safe_load(config_checkpoint.read_text())

    if not compare_basic_configs(config, config_checkpoint):
        logger.error(
            "Different basic configuration from checkpoint. Cannot resume training!"
        )
        exit()

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pth")
    try:
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["acc"]

        logger.info(
            f"Resuming training from epoch {start_epoch - 1} checkpoint in "
            f"{checkpoint_dir}."
        )

    except FileNotFoundError:
        logger.error(f"Checkpoint path '{checkpoint_path}' is not present!")
        exit()

    return model, optimizer, best_acc, start_epoch


def compare_basic_configs(config: Dict, config_checkpoint: Dict) -> bool:
    """Compare basic configurations.

    Args:
        config (Dict): Configuration for this run
        config_checkpoint (Dict): Configuration for checkpoint run

    Returns:
        bool: Check if basic configurations are same with the checkpoint and can
              resume training
    """

    if config["model"] != config_checkpoint["model"]:
        logger.error("Different model configuration from checkpoint!")
        return False

    if config["dataset"] != config_checkpoint["dataset"]:
        logger.error("Different dataset configuration from checkpoint!")
        return False

    if (
        config["training"]["optimizer"]["name"]
        != config_checkpoint["training"]["optimizer"]["name"]
    ):
        logger.error("Different optimizer from checkpoint!")
        return False

    if config["training"]["scheduler"] != config_checkpoint["training"]["scheduler"]:
        logger.error("Different scheduler configuration from checkpoint!")
        return False

    return True


def train(
    start_epoch: int,
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    best_acc: float,
    scheduler: _LRScheduler,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Train model.

    Args:
        start_epoch (int): Starting epoch
        num_epochs (int): Number of epochs
        model (nn.Module): Model
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        criterion (_Loss): Loss function to optimize. Ex. CrossEntropyLoss
        best_acc (float): Initial best accuray
        scheduler (_LRScheduler): Learning rate scheduler for optimizer

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
            epoch, model, train_loader, device, optimizer, criterion
        )
        train_loss.append(loss)
        train_acc.append(acc)

        # Validation
        loss, acc, best_acc = val_epoch(
            epoch, model, val_loader, device, optimizer, criterion, best_acc
        )
        val_loss.append(loss)
        val_acc.append(acc)

        scheduler.step()

    return epochs, train_loss, train_acc, val_loss, val_acc


def train_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    criterion: _Loss,
) -> Tuple[float, float]:
    """Train this epoch.

    Args:
        epoch (int): Epoch index
        model (nn.Module): Model
        train_loader (DataLoader): Training dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        criterion (_Loss): Loss function to optimize. Ex. CrossEntropyLoss

    Returns:
        float: Training loss this epoch
        float: Training accuracy this epoch
    """

    logger.info(f"Training epoch: {epoch}")

    # Initialize
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # Train
    for _, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
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

    logger.info(f"Loss: {train_loss} | Acc: {train_acc}%, ({correct}/{total})")
    return train_loss, train_acc


def val_epoch(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    best_acc: float,
) -> Tuple[float, float, float]:
    """Validate this epoch.

    Args:
        epoch (int): Epoch index
        model (nn.Module): Model
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        criterion (_Loss): Loss function to optimize. Ex. CrossEntropyLoss
        best_acc (float): Best accuracy before this epoch

    Returns:
        float: Validation loss this epoch
        float: Validation accuracy this epoch
        float: Best accuracy after this epoch
    """

    logger.info(f"Validation epoch: {epoch}")

    # Initialize
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # Validation
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Batch validation loss and accuracy
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    logger.info(f"Loss: {val_loss} | Acc: {val_acc}%, ({correct}/{total})")

    # Save checkpoint
    if val_acc >= best_acc:
        logger.info(f"Saving checkpoint from epoch {epoch}.")
        logger.info(f"Best accuracy: was {best_acc}%, now: {val_acc}%.")

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "acc": val_acc,
        }
        torch.save(state, os.path.join(log_dir, "ckpt.pth"))
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

    ax0.plot(epochs, train_loss, label="Train")
    ax0.plot(epochs, val_loss, label="Validation")
    ax0.grid(True)
    ax0.set_ylabel("Loss")

    ax1.plot(epochs, train_acc, label="Train")
    ax1.plot(epochs, val_acc, label="Validation")
    ax1.grid(True)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    lines, labels = ax0.get_legend_handles_labels()
    fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(0.7, 0.45, 0.5, 0.5))

    fig.tight_layout()
    plt.savefig(os.path.join(log_dir, fname), bbox_inches="tight")


if __name__ == "__main__":
    # Get configuration yaml file path
    args = get_args()

    # train <model> with <dataset> in PyTorch
    main(**args.__dict__)
