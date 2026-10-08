"""Shared utilities for the paper experiment scripts.

Provides NLP setup, config serialization, the batch solver evaluation used by
the smoothing study, and the project root.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import torch

from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams
from src.lisco.nlp_handler import NLPHandler, NLPConfig

# Path to the HSL library that provides MA27 for IPOPT. Set it to your local
# installation; every IPOPT configuration in the experiments uses it.
HSLLIB_PATH = "/usr/local/lib/libcoinhsl.so.2.2.6"

# =============================================================================
# NLP Setup
# =============================================================================


def setup_nlp_from_params(
    model_params: CSTRParams,
    nmpc_params: NMPCParams,
    nlp_config: NLPConfig,
) -> tuple[CSTRNMPC, NLPHandler]:
    """Create CSTR model, NMPC, and NLPHandler from typed config objects.

    Typed alternative to setup_nlp() for dict-based experiment configs.

    Args:
        model_params: CSTRParams instance.
        nmpc_params: NMPCParams instance.
        nlp_config: NLPConfig instance with device, dtype, sigma, eps, n_threads.

    Returns:
        Tuple of (CSTRNMPC instance, NLPHandler instance).
    """
    nmpc = CSTRNMPC(model_params, nmpc_params)
    nlp = NLPHandler(nmpc=nmpc, **dataclasses.asdict(nlp_config))
    return nmpc, nlp


# =============================================================================
# Config Serialization
# =============================================================================


def dict_config_to_json(config: dict) -> dict:
    """Serialize a dict-based experiment config to a JSON-safe dict.

    Supports mixed values commonly used in experiment scripts:
    - dataclasses
    - nested dict/list/tuple
    - `torch.dtype`
    - `pathlib.Path`
    - objects exposing `to_dict()`

    This is intended for dict-based experiment configs (e.g. solver training)
    and is general enough to support future dict-based solver configs as well.
    """

    def _convert(v):
        if hasattr(v, "to_dict") and callable(v.to_dict):
            return _convert(v.to_dict())
        if dataclasses.is_dataclass(v):
            return {k: _convert(val) for k, val in dataclasses.asdict(v).items()}
        if isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        return v

    return _convert(config)


# =============================================================================
# Sweep Plan Lookup
# =============================================================================

# Local copy of the shipped plan, written by run_sweep_solver_cstr.py.
DEFAULT_PLAN_PATH = Path("results/solver_training/_plans/sweep_solver_cstr_paper.json")


def load_plan_runs(study: str, plan_path: str | Path = DEFAULT_PLAN_PATH) -> list[dict]:
    """Return the plan entries tagged with ``study`` as dicts with keys
    ``plan_run_id``, ``run_dir`` and ``config``.

    Fails if a tagged run has not been trained yet (status != "done").
    """
    plan_path = Path(plan_path)
    if not plan_path.is_absolute():
        plan_path = project_root() / plan_path
    with open(plan_path) as f:
        plan = json.load(f)

    runs, missing = [], []
    for run in plan["runs"]:
        if study not in run.get("studies", []):
            continue
        if run.get("status") != "done":
            missing.append(run["plan_run_id"])
            continue
        runs.append({
            "plan_run_id": run["plan_run_id"],
            "run_dir": run["save_dir"],
            "config": run["config"],
        })
    if missing:
        raise ValueError(f"Runs for study '{study}' not trained yet: {missing}")
    return runs


# =============================================================================
# Solver Evaluation
# =============================================================================


def _nearest_rank_k(n: int, q_percent: int) -> int:
    return max(1, min(n, int(math.ceil(float(q_percent) * float(n) / 100.0))))


def _summarize_trajectory_values(vals_tb: torch.Tensor) -> dict[str, list[float]]:
    """Summarize a (T, B) trajectory tensor into per-iteration statistics."""
    if vals_tb.dim() != 2:
        raise ValueError(f"Expected shape (T, B), got {tuple(vals_tb.shape)}")
    n = int(vals_tb.shape[1])
    if n < 1:
        raise ValueError("Cannot summarize empty batch dimension.")

    k10 = _nearest_rank_k(n, 10)
    k25 = _nearest_rank_k(n, 25)
    k75 = _nearest_rank_k(n, 75)
    k90 = _nearest_rank_k(n, 90)
    k95 = _nearest_rank_k(n, 95)
    k99 = _nearest_rank_k(n, 99)

    x = vals_tb
    return {
        "mean": x.mean(dim=1).detach().cpu().tolist(),
        "std": x.std(dim=1, unbiased=False).detach().cpu().tolist(),
        "min": x.min(dim=1).values.detach().cpu().tolist(),
        "p10": x.kthvalue(k10, dim=1).values.detach().cpu().tolist(),
        "p25": x.kthvalue(k25, dim=1).values.detach().cpu().tolist(),
        "median": x.median(dim=1).values.detach().cpu().tolist(),
        "p75": x.kthvalue(k75, dim=1).values.detach().cpu().tolist(),
        "p90": x.kthvalue(k90, dim=1).values.detach().cpu().tolist(),
        "p95": x.kthvalue(k95, dim=1).values.detach().cpu().tolist(),
        "p99": x.kthvalue(k99, dim=1).values.detach().cpu().tolist(),
        "max": x.max(dim=1).values.detach().cpu().tolist(),
    }


def _eval_objective_batch(
    nlp: NLPHandler,
    w_batch: torch.Tensor,
    p_batch: torch.Tensor,
) -> torch.Tensor:
    """Evaluate objective f(w,p) on a batch; fallback to per-sample calls if needed."""
    f_batch_func = getattr(nlp, "f_batch_func", None)
    if callable(f_batch_func):
        return f_batch_func(w_batch, p_batch).reshape(-1)

    f_func = getattr(nlp, "f_func", None)
    if not callable(f_func):
        raise AttributeError("NLP handler has neither f_batch_func nor f_func.")

    vals = []
    for i in range(w_batch.shape[0]):
        vals.append(f_func(w_batch[i], p_batch[i]).reshape(()))
    return torch.stack(vals, dim=0)


def basic_eval_solver(
    solver,
    nlp: NLPHandler,
    max_iter: int,
    p_eval: torch.Tensor,
    z0_eval: torch.Tensor,
    z_opt: torch.Tensor,
    n_samples: int = 512,
    eval_seed: int = 42,
    tol_fk: float = 1e-6,
    convergence_tols: list[float] | None = None,
    update_mode: str = "line_search",
) -> dict:
    """Basic solver trajectory evaluation on best-so-far iterates.

    Reduced to the metrics used by the smoothing study: the per-iterate
    fk_inf convergence fraction and the relative objective gap to the IPOPT
    reference ``z_opt``. ``p_eval`` and ``z0_eval`` are always supplied;
    ``n_samples`` and ``eval_seed`` are only recorded in ``meta``.
    """
    active_batch_size = int(p_eval.shape[0])
    nlp.setup_for_batch(active_batch_size, force_rebuild=False)
    p_eval = p_eval.to(device=nlp.device, dtype=nlp.dtype)

    z_opt = z_opt.to(device=nlp.device, dtype=nlp.dtype)
    if z_opt.shape[0] != active_batch_size or z_opt.shape[1] != nlp.n_z:
        raise ValueError(
            f"z_opt must have shape ({active_batch_size}, {nlp.n_z}), got {tuple(z_opt.shape)}"
        )
    z0_eval = z0_eval.to(device=nlp.device, dtype=nlp.dtype)
    if z0_eval.shape[0] != active_batch_size or z0_eval.shape[1] != nlp.n_z:
        raise ValueError(
            f"z0_eval must have shape ({active_batch_size}, {nlp.n_z}), got {tuple(z0_eval.shape)}"
        )
    if convergence_tols is None:
        convergence_tols = [1e-2, 1e-4, 1e-6]
    convergence_tols = [float(t) for t in convergence_tols]
    tol_fk_value = float(tol_fk)

    with torch.no_grad():
        _z_final, z_traj, _step_times = solver.solve_batch(
            p_batch=p_eval,
            max_iter=int(max_iter),
            alpha=1.0,
            update_mode=str(update_mode),
            return_trajectory=True,
            z0_batch=z0_eval,
        )

        fk_inf_list = []
        for z_k in z_traj:
            fk_k = nlp.Fk_batch_func(z_k, p_eval)
            fk_inf_list.append(fk_k.norm(p=float("inf"), dim=1))
        fk_inf_tb = torch.stack(fk_inf_list, dim=0)  # (T, B)

        convergence_fraction = {
            "tolerances": convergence_tols,
            "fk_inf": {},
        }
        for tol in convergence_tols:
            tol_key = f"{tol:.0e}"
            convergence_fraction["fk_inf"][tol_key] = (
                (fk_inf_tb < tol).float().mean(dim=1).detach().cpu().tolist()
            )

        conv_fk = (fk_inf_tb[-1] < tol_fk_value)
        final_summary = {
            "n_samples": active_batch_size,
            "n_converged_fk_inf": int(conv_fk.sum().item()),
            "convergence_rate_fk_inf": float(conv_fk.float().mean().item()),
        }

        meta = {
            "eval_mode": "single_fixed_batch",
            "eval_seed": int(eval_seed),
            "n_samples": active_batch_size,
            "max_iter": int(max_iter),
            "tol_fk": tol_fk_value,
            "tol_metric": "fk_inf",
            "tol_value": tol_fk_value,
            "convergence_tols": convergence_tols,
            "update_mode": str(update_mode),
            "p_eval_source": "external",
            "z0_eval_source": "external",
            "has_z_opt": True,
        }

        result = {
            "meta": meta,
            "convergence_fraction": convergence_fraction,
            "final_summary": final_summary,
        }

        w_ref, _nu_ref, _lam_ref = nlp.extract_primal_dual_batch(z_opt)
        ref_cost = _eval_objective_batch(nlp, w_ref, p_eval)
        ref_cost_denom = torch.clamp(ref_cost.abs(), min=1.0)

        delta_cost_rel = []
        for z_k in z_traj:
            w_k, _nu_k, _lam_k = nlp.extract_primal_dual_batch(z_k)
            pred_cost = _eval_objective_batch(nlp, w_k, p_eval)
            d_cost_signed = pred_cost - ref_cost
            delta_cost_rel.append(d_cost_signed.abs() / ref_cost_denom)

        result["ipopt_delta_summary"] = {
            "delta_cost_rel": _summarize_trajectory_values(torch.stack(delta_cost_rel, dim=0)),
        }

    return result


def iterations_to_tolerance(
    kkt_inf_traj: torch.Tensor,
    tolerances: list[float],
) -> dict[float, torch.Tensor]:
    """Return first iteration index where KKT_inf is below each tolerance.

    Args:
        kkt_inf_traj: Tensor of shape (T, B), where T is number of iterations
            and B is batch size.
        tolerances: List of tolerances.

    Returns:
        Dict mapping tolerance -> tensor of shape (B,) with first hit index.
        Samples that never hit get value T.
    """
    if not torch.is_tensor(kkt_inf_traj):
        raise TypeError("kkt_inf_traj must be a torch.Tensor")
    if kkt_inf_traj.ndim != 2:
        raise ValueError(f"Expected shape (T, B), got {tuple(kkt_inf_traj.shape)}")

    T = int(kkt_inf_traj.shape[0])
    out: dict[float, torch.Tensor] = {}
    for tol in tolerances:
        hit = kkt_inf_traj < float(tol)
        any_hit = hit.any(dim=0)
        first = hit.int().argmax(dim=0)
        first = torch.where(
            any_hit,
            first,
            torch.full_like(first, fill_value=T),
        )
        out[float(tol)] = first
    return out


# =============================================================================
# Saving
# =============================================================================


def project_root() -> Path:
    """Return the project root (common.py -> experiments/ -> src/ -> root)."""
    return Path(__file__).resolve().parent.parent.parent
