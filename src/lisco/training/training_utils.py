"""Training utilities: logger, optimizer, gradient clipping, and finalization."""

import json
import time
from pathlib import Path

import torch


class TrainLogger:
    def __init__(self, buffer_size=100, device=torch.device('cuda')):
        """
        Initialize TrainLogger with GPU buffering capability.

        Args:
            buffer_size (int): Number of entries to accumulate in GPU buffer before flushing to CPU
            device (str): Device to store buffers on ('cuda' or 'cpu')
        """
        self.history = {}
        self.buffer_size = buffer_size
        self.device = device if torch.cuda.is_available() else torch.device('cpu')

        # Buffers: dict of {key: tensor}
        self.buffers = {}
        self.buffer_counts = {}  # Track how many entries in each buffer

    def _init_buffer(self, key):
        if key not in self.buffers:
            dev = self.device
            self.buffers[key] = torch.empty(self.buffer_size, device=dev, dtype=torch.float32)
            self.buffer_counts[key] = 0

    def _flush_buffer(self, key):
        """Flush buffer contents to CPU history."""
        if key not in self.buffers or self.buffer_counts[key] == 0:
            return

        # Get valid entries from buffer
        count = self.buffer_counts[key]
        buffer_data = self.buffers[key][:count].detach().cpu().tolist()

        # Add to history
        if key not in self.history:
            self.history[key] = []
        self.history[key].extend(buffer_data)

        # Reset buffer count
        self.buffer_counts[key] = 0

    def flush_all_buffers(self):
        """Flush all buffers to CPU history."""
        for key in list(self.buffers.keys()):
            self._flush_buffer(key)

    def log_value_buffered(self, val, key):
        # normalize to float32 tensor on target device
        if torch.is_tensor(val):
            t = val.detach()
            if t.device != self.device:
                t = t.to(self.device, non_blocking=True)
            # always cast to float32
            if t.dtype != torch.float32:
                t = t.float()
        else:
            # python numbers become float32
            t = torch.tensor(val, device=self.device, dtype=torch.float32)

        # initialize buffer if needed
        if key not in self.buffers:
            self._init_buffer(key)

        # ensure scalar
        if t.numel() != 1:
            raise ValueError(f"log_value_buffered('{key}'): expected scalar, got shape {tuple(t.shape)}")

        idx = self.buffer_counts[key]
        self.buffers[key][idx] = t
        self.buffer_counts[key] += 1

        if self.buffer_counts[key] >= self.buffer_size:
            self._flush_buffer(key)

    def log_data(self, val, key):
        """Log data that overwrites previous value."""
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        self.history[key] = val

    def log_array(self, val, key):
        """Log array/tensor as list."""
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        val = val.tolist()
        self.history[key] = val

    def save_history(self, exp_pth, file_name="history", as_json=True):
        """Save history to disk. Flushes all buffers first."""
        # Ensure all buffers are flushed before saving
        self.flush_all_buffers()

        exp_pth = Path(exp_pth)
        exp_pth.mkdir(parents=True, exist_ok=True)

        # Save as .pt
        torch.save(self.history, exp_pth.joinpath(file_name + ".pt"))

        # Save as JSON
        if as_json:
            with open(exp_pth.joinpath(file_name + ".json"), 'w') as f:
                json.dump(self.history, f, indent=4)

        print("History saved to: ", exp_pth.joinpath(file_name))


def create_optimizer(params, lr, weight_decay, use_amsgrad=True):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=use_amsgrad)


def adaptive_clip_grad(params, clip_factor=0.01, eps=1e-3):
    """Per-parameter adaptive gradient clipping (AGC, Brock et al. 2021)."""
    for p in params:
        if p.grad is None:
            continue
        param_norm = torch.linalg.vector_norm(p.detach(), ord=2)
        grad_norm = torch.linalg.vector_norm(p.grad.detach(), ord=2)
        max_norm = (param_norm + eps) * clip_factor
        # branchless: a python `if` on a GPU tensor forces a host sync per parameter
        scale = torch.where(grad_norm > max_norm,
                            max_norm / (grad_norm + 1e-16),
                            torch.ones_like(grad_norm))
        p.grad.mul_(scale)


def finalize_training(train_logger, train_tic, epoch, log_values=True, **extra_log_data):
    """Compute train time, log final data, print summary."""
    train_toc = time.perf_counter()
    train_time = train_toc - train_tic

    if log_values:
        train_logger.log_data(train_time, "train_time")
        for key, value in extra_log_data.items():
            if isinstance(value, torch.Tensor):
                train_logger.log_array(value.detach().cpu(), key)
            else:
                train_logger.log_data(value, key)
        train_logger.flush_all_buffers()

    print("#----------------------------------#")
    print("Training complete.")
    print(f"Total number of epochs: {epoch + 1}")
    print(f"Total training time: {train_time:.2f} sec")
    print("#----------------------------------#")

    return train_logger
