"""Standalone training function for iterative solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from src.lisco.training.training_utils import TrainLogger, create_optimizer, finalize_training, adaptive_clip_grad
from src.lisco.nn.nn_models import setup_fast_train


@dataclass
class SolverTrainConfig:
    """Training config for iterative solvers (Solver and subclasses)."""

    N_epochs: int = 1000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 0.0
    use_amsgrad: bool = True
    resampling_strategy: Literal["all", "closed_loop"] = "all"
    Tk_lim: float = 1e-6
    n_steps_max_total: int | None = None
    n_steps_max_no_improvement: int | None = None
    # Not used; kept so that saved configurations load.
    buffer_max_size: int = 1000
    p_buffer: float = 0.5
    p_fresh_min: float = 0.3
    use_adaptive_clip_grad: bool = False
    print_frequency: int = 100
    log_values: bool = True
    tf32: bool = False
    bf16: bool = False


if TYPE_CHECKING:
    from src.lisco.nn.solver import Solver


def train_solver(
    solver: Solver,
    config: SolverTrainConfig,
    print_frequency: int | None = None,
    log_values: bool | None = None,
) -> TrainLogger:
    """Train an iterative solver.

    Args:
        solver: Solver instance (or any subclass).
        config: Typed training configuration.
        print_frequency: Optional override for how often to print progress.
        log_values: Optional override for whether to log training metrics.

    Returns:
        TrainLogger with training history.
    """
    # Configure NLP for this batch size
    solver.nlp.setup_for_batch(config.batch_size, force_rebuild=True)

    # GPU training levers (opt-in; both default off)
    setup_fast_train(solver, tf32=config.tf32, bf16=config.bf16)

    # Unpack config
    N_epochs = config.N_epochs
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    lr = config.lr
    use_amsgrad = config.use_amsgrad
    resampling_strategy = config.resampling_strategy
    Tk_lim = config.Tk_lim
    n_steps_max_total = config.n_steps_max_total
    n_steps_max_no_improvement = config.n_steps_max_no_improvement

    use_adaptive_clip_grad = config.use_adaptive_clip_grad
    if print_frequency is None:
        print_frequency = config.print_frequency
    if log_values is None:
        log_values = config.log_values

    # Train logger
    train_logger = TrainLogger(device=solver.nlp.device)

    # Optimizer initialization
    optim = create_optimizer(solver.parameters(), lr=lr, weight_decay=weight_decay, use_amsgrad=use_amsgrad)

    # Start train timer
    train_tic = time.perf_counter()

    # Initial batch generation
    zk_batch, p_batch = solver.sample_z_p(batch_size)

    # Initial evaluations
    Tk_best = solver.nlp.Tk_batch_func(zk_batch, p_batch)
    Tk_best = Tk_best.detach().clone()
    n_steps_batch = torch.zeros(batch_size)
    n_steps_no_improv = torch.zeros(batch_size)

    for epoch in range(N_epochs):
        # Training step
        loss, zk_new, Tk_batch = solver.training_step(
            zk_batch,
            p_batch,
            optim,
            use_adaptive_clip_grad=use_adaptive_clip_grad,
        )

        # update step
        zk_batch = zk_new.detach().clone()

        with torch.no_grad():
            # update n_steps_batch and n_steps_no_improv
            n_steps_batch += 1
            # update Tk_best
            mask_best = Tk_batch < Tk_best
            Tk_best[mask_best] = Tk_batch[mask_best]
            n_steps_no_improv[~mask_best] += 1
            n_steps_no_improv[mask_best] = 0

            # Create boolean mask for resampling conditions
            mask_tol = Tk_batch < Tk_lim

            # Build max iteration mask based on strategy
            if n_steps_max_total is not None:
                mask_max_total = n_steps_batch >= n_steps_max_total
            else:
                mask_max_total = torch.zeros_like(mask_tol, dtype=torch.bool)

            if n_steps_max_no_improvement is not None:
                mask_max_no_improv = n_steps_no_improv >= n_steps_max_no_improvement
            else:
                mask_max_no_improv = torch.zeros_like(mask_tol, dtype=torch.bool)

            # Combine all masks with logical OR
            mask_resample = mask_tol | mask_max_total | mask_max_no_improv

            # Resample where necessary
            n_resample = torch.sum(mask_resample)
            n_resample_tol = torch.sum(mask_tol)
            n_resample_iter = n_resample - n_resample_tol
            # Number of samples advanced along a closed-loop trajectory this epoch
            n_advance = torch.zeros((), device=Tk_batch.device)

            if n_resample > 0:
                if resampling_strategy == "all":
                    z_new, p_new = solver.sample_z_p(n_resample)
                    zk_batch[mask_resample] = z_new
                    p_batch[mask_resample] = p_new
                    Tk_best[mask_resample] = float('inf')
                    n_steps_batch[mask_resample] = 0
                    n_steps_no_improv[mask_resample] = 0

                elif resampling_strategy == "closed_loop":
                    q_reset = 0.02  # geometric reset probability; mean trajectory depth 1/q_reset

                    # Successor state: w = [u_0, x_1, u_1, ...] and z = [w, nu, lam], so
                    # x_1 = z[:, nu:nu+nx] with per-stage nu, nx. zk_batch already holds
                    # zk_new and Tk_batch / mask_tol were evaluated at that same iterate, so
                    # at tolerance the dynamics constraint gives x_1 = f(x_0, u_0) directly.
                    # Clone: the slice is a view and zk_batch is overwritten below.
                    x1 = zk_batch[:, solver.nlp.nu:solver.nlp.nu + solver.nlp.nx].clone()

                    # Geometric reset: a tolerance-converged sample advances w.p. (1-q_reset).
                    # Iteration-capped and stalled samples are never in mask_tol, so they
                    # always reset uniformly and never propagate a meaningless x_1.
                    keep = torch.rand(batch_size, device=x1.device) > q_reset
                    advance = mask_tol & keep

                    # NOTE: an advanced sample inherits only p; its initial iterate is
                    # the cold start from sample_z_p (nlp.batch_gen_z).
                    z_new, p_new = solver.sample_z_p(n_resample)
                    zk_batch[mask_resample] = z_new
                    p_batch[mask_resample] = p_new
                    # advance is a subset of mask_tol, which is a subset of mask_resample,
                    # so this overwrites exactly the advanced slots' freshly drawn p with
                    # the inherited successor state.
                    p_batch[advance] = x1[advance]
                    Tk_best[mask_resample] = float('inf')
                    n_steps_batch[mask_resample] = 0
                    n_steps_no_improv[mask_resample] = 0
                    n_advance = advance.sum()

                else:
                    raise NotImplementedError("Resampling strategy not implemented yet.")

            # Print
            if (epoch+1) % print_frequency == 0:
                print("Epoch {0}: Loss = {1:.3e}, Tk_min = {2:.3e}, N_tol={3:}".format(epoch+1, loss, Tk_batch.min().item(), n_resample_tol.item()))
                print("-------------------------------")

            # Log values
            if log_values:
                train_logger.log_value_buffered(epoch+1, "epoch")
                train_logger.log_value_buffered(loss, "loss")
                train_logger.log_value_buffered(n_resample, "n_resample")
                train_logger.log_value_buffered(n_resample_tol, "n_resample_tol")
                train_logger.log_value_buffered(n_resample_iter, "n_resample_iter")
                train_logger.log_value_buffered(n_advance, "n_advance")

    return finalize_training(train_logger, train_tic, epoch, log_values, p_final_batch=p_batch)
