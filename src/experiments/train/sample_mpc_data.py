"""Sample (x0, u0*) pairs from CSTR NMPC for ApproxMPC training.

Two modes:
- open_loop:   random x0 samples, each solved independently with IPOPT.
               Failed solves are dropped; sampling continues until n_samples is reached.
- closed_loop: closed-loop NMPC trajectories of fixed length.
               Entire trajectory is discarded if any step fails.
               Sampling continues until n_trajectories successful trajectories are collected.

Outputs are in physical units. Scaling happens in the training script.

The datasets are written to fixed folders (TRAIN_DATA_DIR, VAL_DATA_DIR), which
`train_approx_mpc_cstr.py` reads; re-running overwrites them.

Usage:
    python3 -m src.experiments.train.sample_mpc_data
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.experiments.common import HSLLIB_PATH, project_root
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams

# Fixed dataset folders (relative to the project root).
TRAIN_DATA_DIR = "results/approx_mpc_training/data/train_closed_loop_nr5_N50"
VAL_DATA_DIR = "results/approx_mpc_training/data/val_open_loop_nr5_N50"


@dataclass
class SamplingConfig:
    """Config for MPC data sampling."""

    mode: Literal["open_loop", "closed_loop"]
    # Open-loop
    n_samples: int = 1000       # target number of (x0, u0) pairs
    batch_size: int = 64        # x0 samples drawn per iteration
    # Closed-loop
    n_trajectories: int = 100   # target number of successful trajectories
    n_steps: int = 50           # steps per trajectory


def _data_stats(t: torch.Tensor) -> dict:
    """Per-dimension min/max/mean/std as lists."""
    return {
        "min":  t.min(dim=0).values.tolist(),
        "max":  t.max(dim=0).values.tolist(),
        "mean": t.mean(dim=0).tolist(),
        "std":  t.std(dim=0).tolist(),
    }


def sample_open_loop(
    nmpc: CSTRNMPC,
    config: SamplingConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample random (x0, u0*) pairs by solving IPOPT independently."""
    n_w = nmpc.nlp["n_w"]
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    n_collected = 0
    n_attempted = 0

    while n_collected < config.n_samples:
        # Uniform x0 in scaled [0, 1]^nx
        x0_scaled_batch = np.random.rand(config.batch_size, nmpc.nx)
        x0_phys_batch = x0_scaled_batch * nmpc.x_scale + nmpc.x_min
        n_attempted += config.batch_size

        for i in range(config.batch_size):
            z_i, info_i = nmpc.solve(x0_scaled_batch[i])
            if not info_i.get("success", False):
                continue
            # w_opt is scaled; u0 = first nu elements
            u0_scaled = np.asarray(z_i[:n_w])[:nmpc.nu]
            u0_phys = u0_scaled * nmpc.u_scale + nmpc.u_min
            X_list.append(x0_phys_batch[i])
            Y_list.append(u0_phys)
            n_collected += 1

        print(
            f"  Collected {min(n_collected, config.n_samples)}/{config.n_samples} "
            f"(feasibility: {n_collected/n_attempted:.2%})",
            end="\r",
        )

    print()
    X = torch.tensor(np.array(X_list[: config.n_samples]), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list[: config.n_samples]), dtype=torch.float32)
    stats = {
        "n_attempted": n_attempted,
        "n_collected": int(len(X)),
        "feasibility_rate": n_collected / n_attempted,
    }
    return X, Y, stats


def sample_closed_loop(
    nmpc: CSTRNMPC,
    config: SamplingConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample (x_t, u_t*) pairs along closed-loop NMPC trajectories.

    Entire trajectory is discarded if any step fails.
    """
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    n_traj_success = 0
    n_traj_attempted = 0

    while n_traj_success < config.n_trajectories:
        # Draw a batch of random initial states (scaled), convert to physical
        x0_scaled_batch = np.random.rand(config.batch_size, nmpc.nx)
        x0_phys_batch = x0_scaled_batch * nmpc.x_scale + nmpc.x_min

        for i in range(config.batch_size):
            if n_traj_success >= config.n_trajectories:
                break

            n_traj_attempted += 1
            x_curr = x0_phys_batch[i].copy()
            traj_X: list[np.ndarray] = []
            traj_Y: list[np.ndarray] = []
            feasible = True

            for _ in range(config.n_steps):
                u_opt, info = nmpc.mpc_step(x_curr)
                if not info.get("success", False):
                    feasible = False
                    break
                traj_X.append(x_curr.copy())
                traj_Y.append(np.asarray(u_opt).flatten())
                x_curr = nmpc.model.simulate_step(x_curr, u_opt)

            if feasible:
                n_traj_success += 1
                X_list.append(np.array(traj_X))
                Y_list.append(np.array(traj_Y))

            print(
                f"  Trajectories: {n_traj_success}/{config.n_trajectories} succeeded "
                f"(feasibility: {n_traj_success/n_traj_attempted:.2%})",
                end="\r",
            )

    print()
    X = torch.tensor(np.concatenate(X_list), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(Y_list), dtype=torch.float32)
    n_samples = int(len(X))
    stats = {
        "n_traj_attempted": n_traj_attempted,
        "n_traj_success": n_traj_success,
        "n_samples": n_samples,
        "feasibility_rate": n_traj_success / n_traj_attempted,
    }
    return X, Y, stats


def save_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    nmpc: CSTRNMPC,
    model_params: CSTRParams,
    nmpc_params: NMPCParams,
    config: SamplingConfig,
    stats: dict,
    save_dir: Path,
    ipopt_overrides: dict | None = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(X, save_dir / "X.pt")
    torch.save(Y, save_dir / "Y.pt")

    sampling_cfg_dict: dict = {
        "mode": config.mode,
        "batch_size": config.batch_size,
        "ipopt_overrides": ipopt_overrides or {},
    }
    if config.mode == "open_loop":
        sampling_cfg_dict["n_samples"] = config.n_samples
    else:
        sampling_cfg_dict["n_trajectories"] = config.n_trajectories
        sampling_cfg_dict["n_steps"] = config.n_steps

    meta = {
        "model_params": model_params.to_dict(),
        "nmpc_params": nmpc_params.to_dict(),
        "sampling_config": sampling_cfg_dict,
        "bounds": {
            "x_lb": nmpc.x_min.tolist(),
            "x_ub": nmpc.x_max.tolist(),
            "u_lb": nmpc.u_lb.tolist(),
            "u_ub": nmpc.u_ub.tolist(),
        },
        "sampling_stats": stats,
        "data_stats": {
            "X": _data_stats(X),
            "Y": _data_stats(Y),
        },
    }

    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_samples = stats.get("n_collected") or stats.get("n_samples")
    print(f"\nDataset saved to: {save_dir}")
    print(f"  X: {tuple(X.shape)},  Y: {tuple(Y.shape)}")
    print(f"  Feasibility rate: {stats['feasibility_rate']:.2%}")
    print(f"  Samples: {n_samples}")
    if "sampling_time_s" in stats:
        print(f"  Sampling time: {stats['sampling_time_s']:.2f}s")


def run_sampling(
    model_params: CSTRParams,
    nmpc_params: NMPCParams,
    config: SamplingConfig,
    save_dir: str,
    ipopt_overrides: dict | None = None,
) -> Path:
    """Run MPC data sampling and save results.

    Args:
        model_params: CSTR model parameters.
        nmpc_params:  NMPC parameters.
        config:       Sampling configuration.
        save_dir:     Output directory (relative to project root).
        ipopt_overrides: Optional IPOPT options, without the "ipopt." prefix.

    Returns:
        Path to the saved dataset directory.
    """
    print("=" * 60)
    print("CSTR MPC Data Sampling")
    print(f"  mode={config.mode}, nr={model_params.nr}, N={nmpc_params.N}")
    if config.mode == "open_loop":
        print(f"  n_samples={config.n_samples}, batch_size={config.batch_size}")
    else:
        print(
            f"  n_trajectories={config.n_trajectories}, "
            f"n_steps={config.n_steps}, batch_size={config.batch_size}"
        )
    print("=" * 60)

    print("\n[1/3] Building CSTR NMPC (IPOPT)...")
    nmpc = CSTRNMPC(model_params, nmpc_params)
    if ipopt_overrides:
        nmpc.override_ipopt_options(ipopt_overrides)

    print("\n[2/3] Sampling data...")
    sampling_t0 = time.perf_counter()
    if config.mode == "open_loop":
        X, Y, stats = sample_open_loop(nmpc, config)
    elif config.mode == "closed_loop":
        X, Y, stats = sample_closed_loop(nmpc, config)
    else:
        raise ValueError(f"Unknown sampling mode: '{config.mode}'")
    stats["sampling_time_s"] = time.perf_counter() - sampling_t0

    print("\n[3/3] Saving dataset...")
    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        save_dir = project_root() / save_dir

    save_dataset(
        X,
        Y,
        nmpc,
        model_params,
        nmpc_params,
        config,
        stats,
        save_dir,
        ipopt_overrides=ipopt_overrides,
    )
    return save_dir


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    model_params = CSTRParams(nr=5, dt=1.0)
    nmpc_params = NMPCParams(N=50, w_R=10.0, w_S=1.0, w_uA=0.1, w_uB=0.0)

    # Training data: 3000 closed-loop NMPC trajectories of 50 steps (150000 pairs),
    # IPOPT at tol 1e-5 with mu_target 5e-3.
    run_sampling(
        model_params=model_params,
        nmpc_params=nmpc_params,
        config=SamplingConfig(mode="closed_loop", n_trajectories=3000, n_steps=50),
        save_dir=TRAIN_DATA_DIR,
        ipopt_overrides={
            "tol": 1e-5,
            "mu_target": 5e-3,
            "linear_solver": "ma27",
            "hsllib": HSLLIB_PATH,
        },
    )

    # Validation data: 5000 independent open-loop solves, IPOPT at tol 1e-6.
    # Only used for the logged validation loss during ApproxMPC training.
    run_sampling(
        model_params=model_params,
        nmpc_params=nmpc_params,
        config=SamplingConfig(mode="open_loop", n_samples=5000, batch_size=64),
        save_dir=VAL_DATA_DIR,
        ipopt_overrides={
            "tol": 1e-6,
            "linear_solver": "ma27",
            "hsllib": HSLLIB_PATH,
        },
    )
