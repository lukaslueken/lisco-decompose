"""Generic closed-loop evaluation utilities for the CSTR cascade.

Controller-agnostic helpers (cost, violations, statistics, sampling) and a
generic runner `run_closed_loop_eval` that takes a factory producing
per-trajectory `controller_step(x_phys) -> u_phys` callables.

Used by `eval_approx_mpc_closed_loop_cstr.py` and the solver closed-loop eval.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.models.cstr_nmpc import CSTRNMPC


# =============================================================================
# Config
# =============================================================================


@dataclass
class ClosedLoopEvalConfig:
    """Config for closed-loop CSTR evaluation of any controller."""

    x0_batch: torch.Tensor | None = None    # (B, nx) physical; if None, sample
    n_trajectories: int = 100               # used only when x0_batch is None
    n_sim: int = 100                        # plant steps per trajectory
    viol_tol: float = 1e-6                  # per-dim violations below this count as 0
    seed: int | None = None                 # for x0 sampling


# =============================================================================
# Cost and constraint helpers
# =============================================================================


def stage_cost_economic(x_phys: np.ndarray, u_phys: np.ndarray, nmpc: CSTRNMPC) -> float:
    """NMPC stage cost at realized (x, u). Mirrors CSTRNMPC._build_distributed_nlp.

    Operates on scaled variables to match the NMPC formulation.
    """
    params = nmpc.nmpc_params
    nr = nmpc.model_params.nr

    x_s = (x_phys - nmpc.x_min) / nmpc.x_scale
    u_s = (u_phys - nmpc.u_min) / nmpc.u_scale

    if params.objective == "economic":
        cR = x_s[3 * nr - 1]
        cS = x_s[4 * nr - 1]
        stage = -params.w_R * cR + params.w_S * cS
    else:
        raise ValueError(f"Unknown objective type: {params.objective}")

    uA = u_s[:nr]
    uB = u_s[nr:]
    control_cost = params.w_uA * float(np.sum(uA**2)) + params.w_uB * float(np.sum(uB**2))

    return float(stage + control_cost)


def input_violation_sum(u_phys: np.ndarray, nmpc: CSTRNMPC, tol: float = 0.0) -> float:
    """L1 sum of positive box-constraint violations for inputs (scaled)."""
    u_s  = (u_phys     - nmpc.u_min) / nmpc.u_scale
    lb_s = (nmpc.u_lb  - nmpc.u_min) / nmpc.u_scale
    ub_s = (nmpc.u_ub  - nmpc.u_min) / nmpc.u_scale
    low  = np.where(np.isfinite(lb_s), np.maximum(0.0, lb_s - u_s), 0.0)
    high = np.where(np.isfinite(ub_s), np.maximum(0.0, u_s - ub_s), 0.0)
    low  = np.where(low  > tol, low,  0.0)
    high = np.where(high > tol, high, 0.0)
    return float(np.sum(low) + np.sum(high))


def state_violation_sum(x_phys: np.ndarray, nmpc: CSTRNMPC, tol: float = 0.0) -> float:
    """L1 sum of positive box-constraint violations for states (scaled)."""
    x_s  = (x_phys     - nmpc.x_min) / nmpc.x_scale
    lb_s = (nmpc.x_lb  - nmpc.x_min) / nmpc.x_scale
    ub_s = (nmpc.x_ub  - nmpc.x_min) / nmpc.x_scale
    low  = np.where(np.isfinite(lb_s), np.maximum(0.0, lb_s - x_s), 0.0)
    high = np.where(np.isfinite(ub_s), np.maximum(0.0, x_s - ub_s), 0.0)
    low  = np.where(low  > tol, low,  0.0)
    high = np.where(high > tol, high, 0.0)
    return float(np.sum(low) + np.sum(high))


# =============================================================================
# Stats helpers
# =============================================================================


def summary_stats(values: list[float]) -> dict:
    """min/max/mean/p10/median/p90/p99 for a list of floats."""
    if len(values) == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p99": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "mean":   float(arr.mean()),
        "p10":    float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p99":    float(np.percentile(arr, 99)),
    }


# =============================================================================
# Simulation + sampling
# =============================================================================


def sample_x0_batch_uniform(nmpc: CSTRNMPC, n: int, seed: int | None) -> np.ndarray:
    """Uniform samples in physical state space [x_min, x_max]."""
    rng = np.random.default_rng(seed)
    x0_scaled = rng.random((n, nmpc.nx))
    return x0_scaled * nmpc.x_scale + nmpc.x_min


def _sync_cuda_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _capture_torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def _restore_torch_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def simulate_closed_loop_trajectory(
    x0_phys: np.ndarray,
    controller_step: Callable[[np.ndarray], np.ndarray],
    nmpc: CSTRNMPC,
    n_sim: int,
    viol_tol: float = 0.0,
) -> dict:
    """Simulate one closed-loop trajectory.

    `controller_step(x_phys) -> u_phys` is called at each step. Step time is
    measured around this call. Trajectory is marked failed if the plant state
    becomes non-finite.
    """
    nx, nu = nmpc.nx, nmpc.nu

    x_list = [x0_phys.astype(np.float64).copy()]
    u_list: list[np.ndarray] = []
    stage_cost_list: list[float] = []
    step_time_list: list[float] = []
    u_viol_list: list[float] = []
    x_viol_list: list[float] = []

    failed = False
    fail_step: int | None = None
    x_curr = x0_phys.astype(np.float64).copy()

    for k in range(n_sim):
        if not np.all(np.isfinite(x_curr)):
            failed = True
            fail_step = k
            break

        _sync_cuda_if_needed()
        t0 = time.perf_counter()
        u_phys = np.asarray(controller_step(x_curr), dtype=np.float64).reshape(nu)
        _sync_cuda_if_needed()
        t1 = time.perf_counter()

        stage_cost_list.append(stage_cost_economic(x_curr, u_phys, nmpc))
        step_time_list.append(t1 - t0)
        u_viol_list.append(input_violation_sum(u_phys, nmpc, tol=viol_tol))
        x_viol_list.append(state_violation_sum(x_curr, nmpc, tol=viol_tol))
        u_list.append(u_phys)

        x_next = np.asarray(
            nmpc.model.simulate_step(x_curr, u_phys), dtype=np.float64
        ).reshape(nx)
        x_list.append(x_next)
        x_curr = x_next

    return {
        "x": [xi.tolist() for xi in x_list],
        "u": [ui.tolist() for ui in u_list],
        "stage_cost": stage_cost_list,
        "step_time": step_time_list,
        "input_viol": u_viol_list,
        "state_viol": x_viol_list,
        "failed": failed,
        "fail_step": fail_step,
    }


# =============================================================================
# Runner
# =============================================================================


def run_closed_loop_eval(
    make_controller_step: Callable[[], Callable[[np.ndarray], np.ndarray]],
    nmpc: CSTRNMPC,
    problem_meta: dict,
    save_dir: Path,
    config: ClosedLoopEvalConfig,
    file_tag: str = "closed_loop",
    extra_config: dict | None = None,
) -> Path:
    """Generic closed-loop eval runner.

    For each x0, a fresh controller step callable is built via
    `make_controller_step()` — this gives controllers that keep warm-start
    state a way to reset between trajectories.

    Writes `<file_tag>_results.json` into `save_dir`.
    Returns the path to the results JSON.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve x0 batch
    if config.x0_batch is not None:
        x0_batch = np.asarray(config.x0_batch, dtype=np.float64)
        if x0_batch.ndim != 2 or x0_batch.shape[1] != nmpc.nx:
            raise ValueError(
                f"x0_batch must have shape (B, {nmpc.nx}); got {x0_batch.shape}"
            )
        print(f"  Using provided x0_batch: {x0_batch.shape[0]} trajectories")
    else:
        x0_batch = sample_x0_batch_uniform(nmpc, config.n_trajectories, config.seed)
        print(f"  Sampled {config.n_trajectories} random x0 from [x_min, x_max]")

    n_total = int(x0_batch.shape[0])

    if n_total > 0:
        print("  Warming up controller once before timed simulation...")
        warmup_rng_state = _capture_torch_rng_state()
        try:
            warmup_step = make_controller_step()
            _sync_cuda_if_needed()
            _ = np.asarray(warmup_step(x0_batch[0]), dtype=np.float64)
            _sync_cuda_if_needed()
        finally:
            _restore_torch_rng_state(warmup_rng_state)

    # Simulate
    print(f"  Simulating {n_total} trajectories × {config.n_sim} steps...")
    trajectories: list[dict] = []
    for i in range(n_total):
        step_fn = make_controller_step()
        traj = simulate_closed_loop_trajectory(
            x0_batch[i], step_fn, nmpc, config.n_sim, viol_tol=config.viol_tol
        )
        # Per-step solve outcome, if the controller step logged one (solver arms).
        for key in ("converged", "n_iter", "fk_inf"):
            if hasattr(step_fn, key):
                traj[key] = getattr(step_fn, key)
        trajectories.append(traj)
        if (i + 1) % max(1, n_total // 20) == 0 or (i + 1) == n_total:
            n_failed = sum(1 for t in trajectories if t["failed"])
            print(f"    [{i+1}/{n_total}] failed so far: {n_failed}", end="\r")
    print()

    # Aggregate
    per_traj_total_cost: list[float] = []
    per_traj_max_u_viol: list[float] = []
    per_traj_max_x_viol: list[float] = []

    pooled_stage_cost: list[float] = []
    pooled_step_time: list[float] = []
    pooled_u_viol: list[float] = []
    pooled_x_viol: list[float] = []

    for traj in trajectories:
        sc = traj["stage_cost"]
        uv = traj["input_viol"]
        xv = traj["state_viol"]
        per_traj_total_cost.append(float(sum(sc)) if sc else 0.0)
        per_traj_max_u_viol.append(float(max(uv)) if uv else 0.0)
        per_traj_max_x_viol.append(float(max(xv)) if xv else 0.0)

        pooled_stage_cost.extend(sc)
        pooled_step_time.extend(traj["step_time"])
        pooled_u_viol.extend(uv)
        pooled_x_viol.extend(xv)

    aggregates = {
        "total_cost": summary_stats(per_traj_total_cost),
        "stage_cost": summary_stats(pooled_stage_cost),
        "step_time":  summary_stats(pooled_step_time),
        "input_viol": summary_stats(pooled_u_viol),
        "state_viol": summary_stats(pooled_x_viol),
    }

    n_failed = sum(1 for t in trajectories if t["failed"])

    raw_config = {
        "n_trajectories": n_total,
        "n_sim": config.n_sim,
        "viol_tol": config.viol_tol,
        "seed": config.seed,
        "x0_batch_provided": config.x0_batch is not None,
    }
    if extra_config:
        raw_config.update(extra_config)

    raw = {
        "config": raw_config,
        "problem_meta": problem_meta,
        "bounds": {
            "x_lb": nmpc.x_lb.tolist(),
            "x_ub": nmpc.x_ub.tolist(),
            "u_lb": nmpc.u_lb.tolist(),
            "u_ub": nmpc.u_ub.tolist(),
        },
        "x0_batch": x0_batch.tolist(),
        "trajectories": trajectories,
        "per_trajectory": {
            "total_cost":     per_traj_total_cost,
            "max_input_viol": per_traj_max_u_viol,
            "max_state_viol": per_traj_max_x_viol,
        },
        "aggregates": aggregates,
        "summary": {
            "n_trajectories": n_total,
            "n_failed": n_failed,
            "failure_rate": (n_failed / n_total) if n_total > 0 else 0.0,
        },
    }

    results_path = save_dir / f"{file_tag}_results.json"
    with open(results_path, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\n  Failed: {n_failed}/{n_total} ({100*n_failed/max(1,n_total):.1f}%)")
    print(f"  total_cost  median={aggregates['total_cost']['median']}")
    print(f"  step_time   median={aggregates['step_time']['median']:.3e}s")
    print(f"  input_viol  median={aggregates['input_viol']['median']:.3e}")
    print(f"  state_viol  median={aggregates['state_viol']['median']:.3e}")
    print(f"\nResults saved to: {results_path}")

    return results_path
