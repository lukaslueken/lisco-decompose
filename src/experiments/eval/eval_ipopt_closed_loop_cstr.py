"""Closed-loop evaluation of the exact IPOPT NMPC on the CSTR cascade plant.

Per plant step: call `nmpc.mpc_step(x_phys)` (scale → IPOPT solve → extract
and unscale u0). Writes results into `results/mpc/<timestamp>_nr{nr}_N{N}[_tag]/`
using the same schema + file tag as the ApproxMPC/Solver closed-loop evals, so
downstream aggregation can glob across controllers.

Called from stage 1 of
`src/experiments/visualization/visualize_closed_loop_comparison_cstr.py`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.experiments.common import project_root
from src.experiments.eval.eval_closed_loop_base import (
    ClosedLoopEvalConfig,
    run_closed_loop_eval,
)
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams


# =============================================================================
# Controller step
# =============================================================================


def _make_ipopt_step(nmpc: CSTRNMPC):
    """Build a closure that runs one IPOPT NMPC step and returns u0 (phys)."""
    nu = nmpc.nu

    def step(x_phys: np.ndarray) -> np.ndarray:
        u0, _info = nmpc.mpc_step(np.asarray(x_phys, dtype=np.float64))
        return np.asarray(u0, dtype=np.float64).reshape(nu)

    return step


# =============================================================================
# Runner
# =============================================================================


def run_ipopt_closed_loop_eval(
    model_params: CSTRParams,
    nmpc_params: NMPCParams,
    config: ClosedLoopEvalConfig,
    ipopt_overrides: dict | None = None,
    tag: str = "",
    save_root: str | Path = "results/mpc",
    run_dir: str | Path | None = None,
) -> Path:
    """Run closed-loop evaluation with the exact IPOPT NMPC and save results.

    Args:
        model_params: CSTR cascade parameters.
        nmpc_params:  NMPC parameters.
        config:       Closed-loop eval config (shared with other evals).
        ipopt_overrides: Optional dict of IPOPT options (keys without the
            "ipopt." prefix), applied via `nmpc.override_ipopt_options`.
        tag: Optional short label appended to the run folder name. Ignored
            when `run_dir` is provided.
        save_root: Root under which a new run folder is created. Ignored
            when `run_dir` is provided.
        run_dir: If given, write results into this exact folder (created if
            missing). Enables repeat runs into the same folder (e.g. from the
            comparison vis script). If `None`, a timestamped folder
            `<save_root>/<timestamp>_nr{nr}_N{N}[_<tag>]/` is created.
    """
    print("=" * 60)
    print("IPOPT NMPC Closed-Loop Evaluation")
    print("=" * 60)

    nmpc = CSTRNMPC(model_params, nmpc_params)
    if ipopt_overrides:
        nmpc.override_ipopt_options(ipopt_overrides)

    problem_meta = {
        "model_params": nmpc.model_params.to_dict(),
        "nmpc_params":  nmpc.nmpc_params.to_dict(),
    }

    print(
        f"  nr={model_params.nr}, N={nmpc_params.N}, "
        f"objective={nmpc_params.objective}"
    )
    print(
        f"  ipopt: max_iter={nmpc_params.max_iter}, tol={nmpc_params.ipopt_tol:.1e}, "
        f"overrides={ipopt_overrides or {}}"
    )

    if run_dir is not None:
        save_dir = Path(run_dir)
        if not save_dir.is_absolute():
            save_dir = (project_root() / save_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_nr{model_params.nr}_N{nmpc_params.N}"
        if tag:
            folder_name += f"_{tag}"
        save_dir = (project_root() / save_root / folder_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "problem_meta.json", "w") as f:
        json.dump(problem_meta, f, indent=2)

    print(f"  save_dir: {save_dir}")
    print("\n[1/1] Running closed-loop eval...")

    def make_controller_step():
        return _make_ipopt_step(nmpc)

    return run_closed_loop_eval(
        make_controller_step=make_controller_step,
        nmpc=nmpc,
        problem_meta=problem_meta,
        save_dir=save_dir,
        config=config,
        file_tag="closed_loop",
        extra_config={
            "controller":       "ipopt",
            "tag":              tag,
            "ipopt_overrides":  ipopt_overrides or {},
            "ipopt_tol":        nmpc_params.ipopt_tol,
            "ipopt_max_iter":   nmpc_params.max_iter,
        },
    )
