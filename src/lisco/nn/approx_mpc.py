import json
from pathlib import Path

import torch

from src.lisco.nn.nn_models import FeedforwardNN
from src.lisco.training.training_utils import adaptive_clip_grad


class ApproxMPC(torch.nn.Module):
    """Approximate MPC via a feedforward NN mapping state to control input.

    Args:
        model: Neural network mapping scaled x -> scaled u.
        scale_data: Whether to apply min-max scaling to inputs/outputs.
        bounds: Dict with 'x_lb', 'x_ub', 'u_lb', 'u_ub' tensors (required if scale_data=True).
    """

    def __init__(self, model, scale_data: bool, bounds=None):
        super().__init__()
        self.model = model
        self.scale_data = scale_data

        if scale_data:
            assert bounds is not None, "Bounds must be provided if scale_data is True."
            bounds = bounds.copy()
            for key in bounds:
                if not torch.is_tensor(bounds[key]):
                    bounds[key] = torch.tensor(bounds[key], dtype=torch.float32)
            self.register_buffer("x_lb", bounds["x_lb"])
            self.register_buffer("x_ub", bounds["x_ub"])
            self.register_buffer("u_lb", bounds["u_lb"])
            self.register_buffer("u_ub", bounds["u_ub"])

        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, x_batch, y_batch, optim, use_adaptive_clip_grad: bool = False):
        """Single supervised training step."""
        optim.zero_grad(set_to_none=True)
        y_pred = self.forward(x_batch).to(y_batch.dtype)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        loss.backward()
        if use_adaptive_clip_grad:
            adaptive_clip_grad(self.parameters())
        optim.step()
        return loss

    def scale_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_lb) / (self.x_ub - self.x_lb)

    def scale_outputs(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.u_lb) / (self.u_ub - self.u_lb)

    def rescale_outputs(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * (self.u_ub - self.u_lb) + self.u_lb

    @torch.no_grad()
    def make_step(self, x, scale_inputs=True, rescale_outputs=True, clip_outputs=True):
        """Apply the approximate MPC to compute control input."""
        x_in = self.scale_inputs(x) if scale_inputs else x
        y_scaled = self.forward(x_in)
        y = self.rescale_outputs(y_scaled) if rescale_outputs else y_scaled
        if clip_outputs:
            y = torch.clamp(y, self.u_lb, self.u_ub)
        return y

    def save(self, save_dir):
        """Save ApproxMPC config (JSON) and model weights (.pt) to save_dir.

        Writes:
            approx_mpc_config.json  — all hyperparameters, human-readable
            weights.pt              — FeedforwardNN state dict

        The full ApproxMPC can be reconstructed with ApproxMPC.load(save_dir).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        bounds = None
        if self.scale_data:
            bounds = {
                "x_lb": self.x_lb.tolist(),
                "x_ub": self.x_ub.tolist(),
                "u_lb": self.u_lb.tolist(),
                "u_ub": self.u_ub.tolist(),
            }

        config = {
            "nn_params":  self.model.to_dict(),
            "scale_data": self.scale_data,
            "bounds":     bounds,
        }

        with open(save_dir / "approx_mpc_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.model.save_weights(save_dir)

    @classmethod
    def load(cls, load_dir) -> "ApproxMPC":
        """Reconstruct an ApproxMPC from a directory written by save()."""
        load_dir = Path(load_dir)

        with open(load_dir / "approx_mpc_config.json") as f:
            config = json.load(f)

        nn_cfg = dict(config["nn_params"])
        nn_cfg["nn_dtype"] = getattr(torch, nn_cfg["nn_dtype"])
        model = FeedforwardNN(**nn_cfg)
        model.load_weights(load_dir)

        bounds = None
        if config["bounds"] is not None:
            bounds = {k: torch.tensor(v) for k, v in config["bounds"].items()}

        return cls(model, scale_data=config["scale_data"], bounds=bounds)
