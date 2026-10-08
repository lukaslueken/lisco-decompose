"""Closed-loop evaluation of an ApproxMPC on the CSTR cascade plant.

Loads a trained ApproxMPC, simulates closed-loop trajectories, and writes
per-step metrics + aggregated statistics into the save folder.

Called from stage 1 of
`src/experiments/visualization/visualize_closed_loop_comparison_cstr.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.experiments.common import project_root
from src.experiments.eval.eval_closed_loop_base import (
    ClosedLoopEvalConfig,
    run_closed_loop_eval,
)
from src.lisco.nn.approx_mpc import ApproxMPC
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams


def _make_approx_mpc_step(approx_mpc: ApproxMPC, nu: int):
    """Build a closure that evaluates the ApproxMPC for a single state."""
    def step(x_phys: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(x_phys, dtype=torch.float32).unsqueeze(0)
        u_t = approx_mpc.make_step(
            x_t,
            scale_inputs=True,
            rescale_outputs=True,
            clip_outputs=False,
        )
        return u_t.squeeze(0).detach().cpu().numpy().reshape(nu)
    return step


def run_approx_mpc_closed_loop_eval(
    approx_mpc_dir: str | Path,
    config: ClosedLoopEvalConfig,
    save_dir: str | Path | None = None,
) -> Path:
    """Run closed-loop evaluation for an ApproxMPC and save results.

    Results land in `approx_mpc_dir` unless `save_dir` is given, which lets a
    study keep its own self-contained cache instead of writing into the run
    folder.
    """
    approx_mpc_dir = Path(approx_mpc_dir)
    if not approx_mpc_dir.is_absolute():
        approx_mpc_dir = (project_root() / approx_mpc_dir).resolve()
    save_dir = approx_mpc_dir if save_dir is None else Path(save_dir)

    print("=" * 60)
    print("ApproxMPC Closed-Loop Evaluation")
    print("=" * 60)
    print(f"\n[1/2] Loading ApproxMPC from {approx_mpc_dir}...")

    approx_mpc = ApproxMPC.load(approx_mpc_dir)
    approx_mpc.eval()

    problem_meta_path = approx_mpc_dir / "problem_meta.json"
    if not problem_meta_path.exists():
        raise FileNotFoundError(
            f"problem_meta.json not found in {approx_mpc_dir}. "
            "Re-train the ApproxMPC to produce it, or place a copy there manually."
        )
    with open(problem_meta_path) as f:
        problem_meta = json.load(f)

    model_params = CSTRParams.from_dict(problem_meta["model_params"])
    nmpc_params = NMPCParams.from_dict(problem_meta["nmpc_params"])
    nmpc = CSTRNMPC(model_params, nmpc_params)

    print(f"  nr={model_params.nr}, N={nmpc_params.N}, objective={nmpc_params.objective}")

    print("\n[2/2] Running closed-loop eval...")

    def make_controller_step():
        return _make_approx_mpc_step(approx_mpc, nmpc.nu)

    return run_closed_loop_eval(
        make_controller_step=make_controller_step,
        nmpc=nmpc,
        problem_meta=problem_meta,
        save_dir=save_dir,
        config=config,
        file_tag="closed_loop",
        extra_config={"approx_mpc_dir": str(approx_mpc_dir)},
    )
