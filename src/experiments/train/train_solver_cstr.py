"""Solver training script for CSTR Cascade Economic NMPC.

Trains either a monolithic Solver or a windowed SolverDecomposed from a
dict-based config. The paper runs are executed through
`run_sweep_solver_cstr.py`; running this file trains the single config below
(the closed-loop-resampled LISCO run of the closed-loop study).

Usage:
    python3 -m src.experiments.train.train_solver_cstr
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import torch

from src.experiments.common import (
    setup_nlp_from_params,
    dict_config_to_json,
    project_root,
)

from src.lisco import (
    FeedforwardNN,
    FeedforwardNNConfig,
    Solver,
    SolverDecomposed,
    TrainLogger,
    NLPConfig,
)
from src.lisco.training.train_solver import SolverTrainConfig, train_solver
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import NMPCParams

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def _build_solver(
    architecture: str,
    nn_model: torch.nn.Module,
    nlp,
    solver_cfg: dict,
    decomp_cfg: dict | None,
):
    gamma = solver_cfg.get("gamma", 0.1)
    convexification = solver_cfg.get("convexification", False)
    jvp_linesearch = solver_cfg.get("jvp_linesearch", False)

    if architecture == "monolithic":
        solver = Solver(
            model=nn_model,
            nlp=nlp,
            convexification=convexification,
            gamma=gamma,
            jvp_linesearch=jvp_linesearch,
        )
    elif architecture == "decomposed":
        if decomp_cfg is None:
            raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
        solver = SolverDecomposed(
            model=nn_model,
            nlp=nlp,
            n_nodes=decomp_cfg["n_nodes"],
            n_slide_steps=decomp_cfg["n_slide_steps"],
            overlap_merge_mode=decomp_cfg.get("overlap_merge_mode", "average"),
            convexification=convexification,
            gamma=gamma,
            jvp_linesearch=jvp_linesearch,
        )
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected 'monolithic' or 'decomposed'.")

    if "norm_Fk_max" in solver_cfg:
        solver.norm_Fk_max = solver_cfg["norm_Fk_max"]

    return solver


def _solver_nn_dims(architecture: str, nlp, decomp_cfg: dict | None) -> tuple[int, int]:
    if architecture == "monolithic":
        return nlp.n_p + nlp.n_z + 1, nlp.n_z

    if architecture == "decomposed":
        if decomp_cfg is None:
            raise ValueError("Missing 'decomposition' config for architecture='decomposed'.")
        n_nodes = int(decomp_cfg["n_nodes"])
        n_in = nlp.n_p + n_nodes * nlp.n_z_stage + n_nodes + 2 * n_nodes
        n_out = n_nodes * nlp.n_z_stage
        return n_in, n_out

    raise ValueError(f"Unknown architecture '{architecture}'.")


def run_solver_training(config: dict) -> tuple[torch.nn.Module, TrainLogger, object, Path]:
    """Train a CSTR solver (monolithic or decomposed) from a dict config.

    The config key ``predictor_path`` is not used (it is ``None`` in every
    paper run).
    """
    model_params = config["model"]
    nmpc_params = config["nmpc"]
    nlp_config = config["nlp"]
    nn_config = config["nn"]
    train_config = config["train"]
    architecture = config.get("architecture", "monolithic")
    solver_cfg = dict(config.get("solver", {}))
    decomp_cfg = config.get("decomposition")
    save_dir_str = config.get("save_dir", "results/solver_training")

    print("=" * 60)
    print("CSTR Cascade Solver Training")
    print(f"  architecture={architecture}")
    if architecture == "decomposed" and decomp_cfg is not None:
        print(
            f"  decomposition=(n_nodes={decomp_cfg['n_nodes']}, "
            f"n_slide_steps={decomp_cfg['n_slide_steps']}, "
            f"merge={decomp_cfg.get('overlap_merge_mode', 'average')})"
        )
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup CSTR model and NMPC
    # -------------------------------------------------------------------------
    print("\n[1/4] Setting up CSTR model and NMPC...")
    nmpc, nlp = setup_nlp_from_params(model_params, nmpc_params, nlp_config)

    print(f"  - Reactors: {model_params.nr}")
    print(f"  - Horizon: {nmpc_params.N}")
    print(f"  - Weights: w_R={nmpc_params.w_R}, w_S={nmpc_params.w_S}")
    print(f"  - Objective: {nmpc_params.objective}")
    print(f"  - n_p (parameters): {nlp.n_p}")
    print(f"  - n_z (primal-dual): {nlp.n_z}")
    if architecture == "decomposed":
        print(f"  - n_z_stage: {nlp.n_z_stage}")
    print(f"  - Device: {nlp.device}")

    # -------------------------------------------------------------------------
    # 2. Create neural network and solver
    # -------------------------------------------------------------------------
    print("\n[2/4] Creating neural network and solver...")

    n_in, n_out = _solver_nn_dims(architecture, nlp, decomp_cfg)

    nn_model = FeedforwardNN(
        n_in=n_in,
        n_out=n_out,
        **dataclasses.asdict(nn_config),
    ).to(nlp_config.device)

    n_params = sum(p.numel() for p in nn_model.parameters())
    print(
        f"  - Architecture: {n_in} -> {nn_config.n_neurons}x{nn_config.n_hidden_layers} -> {n_out}"
    )
    print(f"  - Parameters: {n_params:,}")
    print(f"  - Activation: {nn_config.act_fn}")

    solver = _build_solver(
        architecture=architecture,
        nn_model=nn_model,
        nlp=nlp,
        solver_cfg=solver_cfg,
        decomp_cfg=decomp_cfg,
    )

    # -------------------------------------------------------------------------
    # 3. Train
    # -------------------------------------------------------------------------
    print("\n[3/4] Training solver...")
    train_logger = train_solver(solver, config=train_config)

    # -------------------------------------------------------------------------
    # 4. Save results
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if architecture == "decomposed" and decomp_cfg is not None:
        suffix = (
            f"_decomp_n{decomp_cfg['n_nodes']}"
            f"_s{decomp_cfg['n_slide_steps']}"
            f"_{decomp_cfg.get('overlap_merge_mode', 'average')}"
        )
    else:
        suffix = "_mono"

    save_dir = Path(save_dir_str)
    if not save_dir.is_absolute():
        save_dir = project_root() / save_dir
    save_dir = save_dir / f"{timestamp}_nr{model_params.nr}_N{nmpc_params.N}_T{int(nmpc_params.N*model_params.dt)}{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    solver.save(save_dir)
    train_logger.save_history(save_dir, file_name="training_log")
    with open(save_dir / "config.json", "w") as f:
        json.dump(dict_config_to_json(config), f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    return solver, train_logger, nlp, save_dir


if __name__ == "__main__":
    total_time = 50 # seconds
    dt = 1.0 # seconds
    config = {
        "architecture": "decomposed", # "monolithic", "decomposed"
        "model": CSTRParams(nr=5, dt=dt),
        "nmpc": NMPCParams(N=int(total_time/dt), w_R=10.0, w_S=1.0, w_uA=0.1, w_uB=0.0),
        "nlp": NLPConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float64,
            sigma=1e-2,        # dual regularization
            rho=1e-2,          # primal regularization; required PD for economic objective
            regularize=True,
            eps=1e-1,
            distributed_KKT_functions=True,
        ),
        "nn": FeedforwardNNConfig(
            n_neurons=1024,
            n_hidden_layers=2,
            act_fn="gelu",
            nn_dtype=torch.float32,
        ),
        "solver": {
            "gamma": 0.1,
            "convexification": False,
            "norm_Fk_max": 100.0,
            "jvp_linesearch": False,
        },
        # Not used; kept so that the saved config.json matches the paper runs.
        "predictor_path": None,
        # Only used when architecture == "decomposed"
        "decomposition": {
            "n_nodes": 5,
            "n_slide_steps": 1,
            "overlap_merge_mode": "average",
        },
        "train": SolverTrainConfig(
        N_epochs=int(total_time*6000/dt),
        batch_size= 256,
        lr=5*1e-4,
        weight_decay=0.0,
        use_amsgrad=True,
        resampling_strategy="closed_loop", # "all", "closed_loop"
        Tk_lim=1e-16,
        n_steps_max_total=1000,
        n_steps_max_no_improvement=2,
        use_adaptive_clip_grad=True,
        print_frequency=100,
        log_values=True,
        tf32=True,
        bf16=True,
        ),
        "save_dir": "results/solver_training",
    }

    solver, train_logger, nlp, save_dir = run_solver_training(config)
