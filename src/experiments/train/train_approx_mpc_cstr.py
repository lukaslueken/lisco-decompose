"""ApproxMPC training script for CSTR Cascade NMPC.

Loads a pre-sampled dataset (produced by sample_mpc_data.py), scales inputs and
outputs to [0, 1] using the bounds stored in meta.json, trains a FeedforwardNN-based
ApproxMPC, and saves the result to the fixed folder APPROX_MPC_DIR, which the
closed-loop study reads; re-running overwrites it.

Usage:
    python3 -m src.experiments.train.train_approx_mpc_cstr
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.experiments.common import dict_config_to_json, project_root
from src.experiments.train.sample_mpc_data import TRAIN_DATA_DIR, VAL_DATA_DIR
from src.lisco import FeedforwardNN, FeedforwardNNConfig, TrainLogger
from src.lisco.nn.approx_mpc import ApproxMPC
from src.lisco.training.train_approx_mpc import ApproxMPCTrainConfig, train_approx_mpc

# Fixed output folder of the trained ApproxMPC (relative to the project root).
APPROX_MPC_DIR = "results/approx_mpc_training/approx_mpc_nr5_N50"


def _load_dataset(
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load X.pt, Y.pt and meta.json from a dataset directory."""
    X = torch.load(data_dir / "X.pt", weights_only=True)
    Y = torch.load(data_dir / "Y.pt", weights_only=True)
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    return X, Y, meta


def _scale(tensor: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return (tensor - lb) / (ub - lb)


# =============================================================================
# Experiment Runner
# =============================================================================


def run_approx_mpc_training(config: dict) -> tuple[ApproxMPC, TrainLogger, Path]:
    """Train an ApproxMPC from a pre-sampled dataset.

    Args:
        config: Dict with keys:
            "data_dir"  — path to dataset directory (str or Path)
            "nn"        — FeedforwardNNConfig
            "train"     — ApproxMPCTrainConfig
            "save_dir"  — output directory of the trained model

    Returns:
        Tuple of (trained ApproxMPC, training logger, save directory path).
    """
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (project_root() / data_dir).resolve()

    val_data_dir = config.get("val_data_dir")
    if val_data_dir is not None:
        val_data_dir = Path(val_data_dir)
        if not val_data_dir.is_absolute():
            val_data_dir = (project_root() / val_data_dir).resolve()

    nn_config: FeedforwardNNConfig = config["nn"]
    train_config: ApproxMPCTrainConfig = config["train"]
    save_dir_str: str = config["save_dir"]

    # -------------------------------------------------------------------------
    # 1. Load dataset(s)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("CSTR ApproxMPC Training")
    print("=" * 60)
    print(f"\n[1/4] Loading dataset from {data_dir}...")

    X_raw, Y_raw, meta = _load_dataset(data_dir)

    bounds_meta = meta["bounds"]
    x_lb = torch.tensor(bounds_meta["x_lb"], dtype=torch.float32)
    x_ub = torch.tensor(bounds_meta["x_ub"], dtype=torch.float32)
    u_lb = torch.tensor(bounds_meta["u_lb"], dtype=torch.float32)
    u_ub = torch.tensor(bounds_meta["u_ub"], dtype=torch.float32)
    bounds = {"x_lb": x_lb, "x_ub": x_ub, "u_lb": u_lb, "u_ub": u_ub}

    nx = X_raw.shape[1]
    nu = Y_raw.shape[1]
    n_data = X_raw.shape[0]
    mode = meta["sampling_config"]["mode"]
    nr = meta["model_params"]["nr"]
    N = meta["nmpc_params"]["N"]

    print(f"  Mode: {mode}")
    print(f"  nr={nr}, N={N}")
    print(f"  n_data={n_data}, nx={nx}, nu={nu}")
    print(f"  Feasibility rate: {meta['sampling_stats']['feasibility_rate']:.2%}")

    # Scale inputs and outputs to [0, 1]
    X = _scale(X_raw, x_lb, x_ub)
    Y = _scale(Y_raw, u_lb, u_ub)
    train_dataset = (X, Y)

    val_dataset = None
    if val_data_dir is not None:
        print(f"  Loading validation dataset from {val_data_dir}...")
        X_val_raw, Y_val_raw, _ = _load_dataset(val_data_dir)
        X_val = _scale(X_val_raw, x_lb, x_ub)
        Y_val = _scale(Y_val_raw, u_lb, u_ub)
        val_dataset = (X_val, Y_val)
        print(f"  n_val={X_val_raw.shape[0]}")

    # -------------------------------------------------------------------------
    # 2. Build neural network and ApproxMPC
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Building FeedforwardNN and ApproxMPC...")

    nn_model = FeedforwardNN(
        n_in=nx,
        n_out=nu,
        **nn_config.__dict__,
    )

    n_params = sum(p.numel() for p in nn_model.parameters())
    print(
        f"  Architecture: {nx} -> {nn_config.n_neurons}x{nn_config.n_hidden_layers} -> {nu}"
    )
    print(f"  Parameters: {n_params:,}")
    print(f"  Activation: {nn_config.act_fn}")

    approx_mpc = ApproxMPC(model=nn_model, scale_data=True, bounds=bounds)

    # -------------------------------------------------------------------------
    # 3. Train
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Training ApproxMPC...")
    train_logger = train_approx_mpc(approx_mpc, train_dataset, train_config, val_dataset=val_dataset)

    # -------------------------------------------------------------------------
    # 4. Save
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Saving results...")

    save_dir = Path(save_dir_str)
    if not save_dir.is_absolute():
        save_dir = project_root() / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    approx_mpc.save(save_dir)
    train_logger.save_history(save_dir, file_name="training_log")

    # Persist problem metadata (model + nmpc params) from the training data's meta.json.
    # Downstream evaluation scripts read this to reconstruct CSTRNMPC.
    problem_meta = {
        "model_params": meta["model_params"],
        "nmpc_params": meta["nmpc_params"],
    }
    with open(save_dir / "problem_meta.json", "w") as f:
        json.dump(problem_meta, f, indent=2)

    # Store full config (nn and train as dicts, data_dir as string)
    config_to_save = {
        "data_dir": str(data_dir),
        "val_data_dir": str(val_data_dir) if val_data_dir is not None else None,
        "nn": nn_config.__dict__,
        "train": train_config.__dict__,
        "save_dir": save_dir_str,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(dict_config_to_json(config_to_save), f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    return approx_mpc, train_logger, save_dir


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    config = {
        "data_dir": TRAIN_DATA_DIR,
        "val_data_dir": VAL_DATA_DIR,
        "nn": FeedforwardNNConfig(
            n_neurons=1024,
            n_hidden_layers=2,
            act_fn="gelu",
            nn_dtype=torch.float32,
        ),
        "train": ApproxMPCTrainConfig(
            N_epochs=2500,
            batch_size=1024,
            lr=5*1e-4,
            use_lr_scheduler=False,
        ),
        "save_dir": APPROX_MPC_DIR,
    }

    approx_mpc, train_logger, save_dir = run_approx_mpc_training(config)
