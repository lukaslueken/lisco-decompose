"""Standalone training function for ApproxMPC."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.lisco.training.training_utils import TrainLogger, adaptive_clip_grad, create_optimizer, finalize_training


@dataclass
class ApproxMPCTrainConfig:
    """Training config for ApproxMPC (supervised mini-batch training)."""

    N_epochs: int = 1000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 0.0
    use_amsgrad: bool = True
    use_adaptive_clip_grad: bool = False
    use_lr_scheduler: bool = False
    stop_at_min_lr: bool = False
    lr_scheduler_patience: int = 50
    lr_scheduler_cooldown: int = 10
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    print_frequency: int = 10
    log_values: bool = True


if TYPE_CHECKING:
    from src.lisco.nn.approx_mpc import ApproxMPC


def train_approx_mpc(
    model: ApproxMPC,
    train_dataset: tuple[torch.Tensor, torch.Tensor],
    config: ApproxMPCTrainConfig,
    val_dataset: tuple[torch.Tensor, torch.Tensor] | None = None,
    print_frequency: int | None = None,
    log_values: bool | None = None,
) -> TrainLogger:
    """Train an ApproxMPC model.

    Args:
        model: ApproxMPC instance.
        train_dataset: (X_train, Y_train) tensors.
        config: Typed training configuration.
        val_dataset: Optional (X_val, Y_val) tensors.
        print_frequency: How often to print progress.
        log_values: Whether to log training metrics.

    Returns:
        TrainLogger with training history.
    """
    # Unpack config
    N_epochs = config.N_epochs
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    lr = config.lr
    use_amsgrad = config.use_amsgrad
    use_adaptive_clip_grad = config.use_adaptive_clip_grad
    use_lr_scheduler = config.use_lr_scheduler
    stop_at_min_lr = config.stop_at_min_lr
    if print_frequency is None:
        print_frequency = config.print_frequency
    if log_values is None:
        log_values = config.log_values

    # Unpack datasets
    X_train, Y_train = train_dataset
    n_data = X_train.shape[0]
    n_steps = n_data // batch_size

    # Setup logger
    train_logger = TrainLogger(device=X_train.device)

    # Setup optimizer
    optim = create_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, use_amsgrad=use_amsgrad)

    # LR Scheduler
    if use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=config.lr_reduce_factor,
            patience=config.lr_scheduler_patience,
            threshold=1e-5, threshold_mode='rel',
            cooldown=config.lr_scheduler_cooldown,
            min_lr=config.min_lr, eps=0.0
        )

    # Training loop
    train_tic = time.perf_counter()

    for epoch in range(N_epochs):

        # Shuffle data
        idx_shuffle = torch.randperm(n_data)

        train_loss = 0.0
        # Training steps
        for idx_train_batch in range(n_steps):
            idx_batch = idx_shuffle[idx_train_batch*batch_size:(idx_train_batch+1)*batch_size]
            x_batch = X_train[idx_batch]
            y_batch = Y_train[idx_batch]

            loss = model.training_step(x_batch, y_batch, optim, use_adaptive_clip_grad=use_adaptive_clip_grad)
            train_loss += loss.item()

        train_loss = train_loss / n_steps

        # Validation
        if val_dataset is not None:
            with torch.no_grad():
                X_val, Y_val = val_dataset
                y_val_pred = model.forward(X_val)
                val_loss = torch.nn.functional.mse_loss(y_val_pred, Y_val).item()

        # LR Scheduler step
        if use_lr_scheduler:
            lr_scheduler.step(train_loss)

        # Log values
        if log_values:
            train_logger.log_value_buffered(epoch + 1, "epoch")
            train_logger.log_value_buffered(train_loss, "loss")
            if val_dataset is not None:
                train_logger.log_value_buffered(val_loss, "val_loss")
            train_logger.log_value_buffered(optim.param_groups[0]["lr"], "lr")

        # Print progress
        if (epoch + 1) % print_frequency == 0:
            print("Epoch {0}: Loss = {1:.3e}".format(epoch + 1, train_loss))
            if val_dataset is not None:
                print("         Val Loss = {0:.3e}".format(val_loss))
            print("-------------------------------")

        # Early stopping
        if stop_at_min_lr and use_lr_scheduler:
            if optim.param_groups[0]["lr"] <= config.min_lr:
                break

    return finalize_training(train_logger, train_tic, epoch, log_values)
