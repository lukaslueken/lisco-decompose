"""Solver loading and IPOPT-feasible sample generation for batch evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.lisco import Solver, SolverDecomposed


def _load_solver_from_run_dir(run_dir: Path):
    cfg_path = run_dir / "solver_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing solver config: {cfg_path}")

    with open(cfg_path) as f:
        cfg = json.load(f)
    solver_class = cfg.get("solver_class", "Solver")

    if solver_class == "SolverDecomposed":
        solver = SolverDecomposed.load(run_dir)
    elif solver_class == "Solver":
        solver = Solver.load(run_dir)
    else:
        raise ValueError(f"Unsupported solver class '{solver_class}' in {cfg_path}")

    return solver


def _sample_ipopt_feasible_batch(
    nlp,
    n_samples: int,
    eval_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sample p and IPOPT solutions until n_samples feasible or n_samples*10 attempts."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    torch.manual_seed(int(eval_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(eval_seed))

    max_attempts = int(n_samples) * 10
    attempted = 0
    feasible = 0
    p_list: list[torch.Tensor] = []
    z_list: list[torch.Tensor] = []

    while feasible < n_samples and attempted < max_attempts:
        p_i = nlp.batch_gen_p(1)[0]
        z_opt_i, info = nlp.nmpc.solve(p_i.detach().cpu().numpy())
        attempted += 1

        success = bool(info.get("success", False) or info.get("return_status") == "Solve_Succeeded")
        if success:
            p_list.append(p_i.detach().clone())
            z_list.append(torch.as_tensor(z_opt_i, device=nlp.device, dtype=nlp.dtype))
            feasible += 1

    if feasible < 1:
        raise RuntimeError(
            f"IPOPT sampling produced no feasible solutions after {attempted}/{max_attempts} attempts."
        )

    p_eval = torch.stack(p_list, dim=0)
    z_opt = torch.stack(z_list, dim=0)
    stats = {
        "n_ipopt_attempted": int(attempted),
        "n_ipopt_feasible": int(feasible),
        "ipopt_feasible_fraction": float(feasible / attempted),
        "n_ipopt_target": int(n_samples),
        "n_ipopt_max_attempts": int(max_attempts),
    }
    return p_eval, z_opt, stats
