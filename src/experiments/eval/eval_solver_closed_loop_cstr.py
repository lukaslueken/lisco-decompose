"""Closed-loop evaluation of a learned Solver on the CSTR cascade plant.

Uses the fast single-sample path (`Solver.solve_fast_single`, CPU/casadi_single
backend) to measure the fastest possible per-step controller times.

Per plant step: fresh (seeded random) cold-start z0 → solve with max_iter /
inf-norm Fk tolerance → extract u0 from z_best. No warm start between steps.

Writes results into the solver run folder using the same schema + file tag as
the ApproxMPC closed-loop eval, so downstream aggregation can glob across
controllers.

Called from stage 1 of
`src/experiments/visualization/visualize_closed_loop_comparison_cstr.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import json
import numpy as np
import torch


SolverEvalProfile = Literal["cpu_all_ca"]


_PROFILE_SETUP: dict[str, dict] = {
    # CPU for NN + CasADi single-point KKT. Default, no CUDA needed.
    "cpu_all_ca": {
        "nlp_device":      "cpu",
        "nn_device":       "cpu",
        "kkt_backend":     "casadi_single",
        "kkt_device":      "cpu",
        "nn_compile_mode": "none",
        "nn_dtype":        torch.bfloat16,
    },
}

from src.experiments.common import project_root
from src.experiments.eval.eval_closed_loop_base import (
    ClosedLoopEvalConfig,
    run_closed_loop_eval,
)
from src.lisco import FeedforwardNN, Solver, SolverDecomposed
from src.lisco.nlp_handler import NLPHandler
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams


# =============================================================================
# Solver step config
# =============================================================================


@dataclass
class SolverStepConfig:
    """Solver-specific knobs for the per-plant-step NLP solve."""

    max_iter: int = 100
    fk_inf_tol: float = 1e-6    # tolerance on inf-norm of Fk
    alpha: float = 1.0


# =============================================================================
# Solver loading
# =============================================================================


def _load_solver_for_profile(run_dir: Path, profile: SolverEvalProfile) -> Solver:
    """Load a Solver configured for the requested single-eval profile.

    Supported profiles:
    - ``cpu_all_ca``:    NN + KKT on CPU (casadi_single). No CUDA needed.
    """
    if profile not in _PROFILE_SETUP:
        raise ValueError(
            f"Unknown profile '{profile}'. Supported: {sorted(_PROFILE_SETUP)}"
        )
    setup = _PROFILE_SETUP[profile]

    cfg_path = run_dir / "solver_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing solver config: {cfg_path}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    solver_class = cfg.get("solver_class", "Solver")

    model_params = CSTRParams.from_dict(cfg["model_params"])
    nmpc_params = NMPCParams.from_dict(cfg["nmpc_params"])
    nmpc = CSTRNMPC(model_params, nmpc_params)

    nlp_cfg = dict(cfg["nlp_params"])
    nlp_cfg["dtype"] = getattr(torch, str(nlp_cfg["dtype"]))
    nlp_cfg["device"] = setup["nlp_device"]
    nlp = NLPHandler(nmpc=nmpc, **nlp_cfg)

    nn_cfg = dict(cfg["nn_params"])
    nn_cfg["nn_dtype"] = getattr(torch, str(nn_cfg["nn_dtype"]))
    model = FeedforwardNN(**nn_cfg)
    model.load_weights(run_dir)
    model.to(nlp.device)
    model.eval()

    solver_params = dict(cfg.get("solver_params", {}))
    if solver_class == "SolverDecomposed":
        solver = SolverDecomposed(model=model, nlp=nlp, **solver_params)
    elif solver_class == "Solver":
        solver = Solver(model=model, nlp=nlp, **solver_params)
    else:
        raise ValueError(f"Unsupported solver class '{solver_class}' in {cfg_path}")

    solver.eval()
    # setup_for_single_eval deep-copies the model and casts it to nn_device /
    # nn_dtype itself -- do not pre-cast here.
    solver.setup_for_single_eval(
        nn_device=setup["nn_device"],
        nn_dtype=setup["nn_dtype"],
    )
    return solver


# =============================================================================
# Controller step
# =============================================================================


def _make_solver_step(
    solver: Solver,
    nmpc: CSTRNMPC,
    nlp: NLPHandler,
    step_cfg: SolverStepConfig,
):
    """Build a closure that runs a single cold-start solve and returns u0 (phys).

    The per-step solve outcome is logged on the closure itself: the runner builds
    one closure per trajectory, so the lists are per-trajectory and line up with
    the trajectory's step_time.  All three values are already returned by the
    solve, so logging them costs one list append each.
    """
    nu = nmpc.nu
    converged: list[bool] = []
    n_iter_list: list[int] = []
    fk_inf_list: list[float] = []

    def step(x_phys: np.ndarray) -> np.ndarray:
        p = np.asarray(nmpc._scale_x(x_phys), dtype=np.float64).reshape(-1)
        z0 = nlp.batch_gen_z(1).squeeze(0)

        z_best, n_iter, success, fk_inf = solver.solve_fast_single(
            p=p,
            z0=z0,
            max_iter=step_cfg.max_iter,
            tol=step_cfg.fk_inf_tol,
            alpha=step_cfg.alpha,
            update_mode="line_search",
        )
        converged.append(bool(success))
        n_iter_list.append(int(n_iter))
        fk_inf_list.append(float(fk_inf))
        z_np = z_best.detach().cpu().numpy().reshape(-1)
        u0_scaled = np.asarray(z_np[:nu], dtype=np.float64)
        return np.asarray(nmpc._unscale_u(u0_scaled), dtype=np.float64).reshape(nu)

    step.converged = converged
    step.n_iter = n_iter_list
    step.fk_inf = fk_inf_list
    return step


# =============================================================================
# Runner
# =============================================================================


def run_solver_closed_loop_eval(
    solver_dir: str | Path,
    config: ClosedLoopEvalConfig,
    step_cfg: SolverStepConfig = SolverStepConfig(),
    profile: SolverEvalProfile = "cpu_all_ca",
    save_dir: str | Path | None = None,
) -> Path:
    """Run closed-loop evaluation for a trained Solver and save results.

    Results land in `solver_dir` unless `save_dir` is given, which lets a study
    keep its own self-contained cache instead of writing into the run folder.
    """
    solver_dir = Path(solver_dir)
    if not solver_dir.is_absolute():
        solver_dir = (project_root() / solver_dir).resolve()
    save_dir = solver_dir if save_dir is None else Path(save_dir)

    setup = _PROFILE_SETUP[profile]

    print("=" * 60)
    print("Solver Closed-Loop Evaluation")
    print("=" * 60)
    print(f"\n[1/2] Loading Solver (profile={profile}) from {solver_dir}...")

    solver = _load_solver_for_profile(solver_dir, profile)
    nmpc = solver.nlp.nmpc
    nlp = solver.nlp

    problem_meta = {
        "model_params": nmpc.model_params.to_dict(),
        "nmpc_params":  nmpc.nmpc_params.to_dict(),
    }

    print(
        f"  nr={nmpc.model_params.nr}, N={nmpc.nmpc_params.N}, "
        f"objective={nmpc.nmpc_params.objective}"
    )
    print(
        f"  solver step: max_iter={step_cfg.max_iter}, "
        f"fk_inf_tol={step_cfg.fk_inf_tol:.1e}, alpha={step_cfg.alpha}"
    )
    print(
        f"  profile={profile}: nn_device={setup['nn_device']}, "
        f"kkt_backend={setup['kkt_backend']}, kkt_device={setup['kkt_device']}, "
        f"nn_compile_mode={setup['nn_compile_mode']}"
    )

    # Seed torch for reproducible batch_gen_z calls (cold z0 per plant step).
    if config.seed is not None:
        torch.manual_seed(config.seed)

    print("\n[2/2] Running closed-loop eval...")

    def make_controller_step():
        return _make_solver_step(solver, nmpc, nlp, step_cfg)

    return run_closed_loop_eval(
        make_controller_step=make_controller_step,
        nmpc=nmpc,
        problem_meta=problem_meta,
        save_dir=save_dir,
        config=config,
        file_tag="closed_loop",
        extra_config={
            "solver_dir":      str(solver_dir),
            "max_iter":        step_cfg.max_iter,
            "fk_inf_tol":      step_cfg.fk_inf_tol,
            "alpha":           step_cfg.alpha,
            "update_mode":     "line_search",
            "warm_start":      "cold (seeded random via nlp.batch_gen_z)",
            "profile":         profile,
            "nlp_device":      setup["nlp_device"],
            "nn_device":       setup["nn_device"],
            "kkt_backend":     setup["kkt_backend"],
            "kkt_device":      setup["kkt_device"],
            "nn_compile_mode": setup["nn_compile_mode"],
            "nn_dtype":        str(setup["nn_dtype"]).removeprefix("torch."),
        },
    )
