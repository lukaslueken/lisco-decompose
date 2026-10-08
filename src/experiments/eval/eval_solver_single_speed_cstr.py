"""Single data-point (B=1) speed evaluation for one saved solver run.

Primary goal: an accurate, fair latency comparison of the learned solver vs IPOPT.
Per sample the timed region covers the cold-start solve to tolerance, so
`solve_time` is comparable to IPOPT's wall-clock solve on the same p.

Profiles:
- cpu_all_ca: solver + CasADi single-point residual on CPU (bfloat16 network).
- ipopt:      IPOPT solve on the same sampled p.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.experiments.common import project_root

from src.lisco import FeedforwardNN, Solver, SolverDecomposed
from src.lisco.nlp_handler import NLPHandler
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _summary(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _sync_device(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_solver_from_run_dir(run_dir: Path, nlp_device: str):
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
    nlp_cfg["device"] = nlp_device
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
    return solver


def _sync_solver_devices(solver) -> None:
    _sync_device(getattr(solver, "_single_nn_device", solver.nlp.device))
    _sync_device(getattr(solver, "_single_kkt_device", solver.nlp.device))


@torch.no_grad()
def _measure_solver_profile(
    name: str,
    solver,
    p_eval_cpu: torch.Tensor,
    z0_eval_cpu: torch.Tensor,
    max_iter: int,
    warmup: int,
    tol_metric: str,
    tol_value: float,
) -> dict:
    if tol_metric != "fk_inf":
        raise ValueError(f"solve_fast_single supports tolerance on Fk_inf only. Got '{tol_metric}'.")

    n_samples = int(p_eval_cpu.shape[0])
    if int(z0_eval_cpu.shape[0]) != n_samples:
        raise ValueError(
            f"z0_eval_cpu batch size must match p_eval_cpu. "
            f"Got {int(z0_eval_cpu.shape[0])} vs {n_samples}."
        )

    def _make_z0(i: int):
        """Return the cold-start z0 of sample i."""
        return z0_eval_cpu[i]

    for i in range(int(warmup)):
        j = i % n_samples
        z0_i = _make_z0(j)
        solver.solve_fast_single(
            p=p_eval_cpu[j], z0=z0_i, max_iter=max_iter,
            tol=float(tol_value), alpha=1.0, update_mode="line_search",
        )
    _sync_solver_devices(solver)

    solve_times_s: list[float] = []
    iters_run: list[float] = []
    iter_to_tol_vals: list[float] = []
    time_to_tol_s_vals: list[float] = []
    converged_mask: list[float] = []

    for i in range(n_samples):
        p_i = p_eval_cpu[i]

        _sync_solver_devices(solver)
        t0 = time.perf_counter()
        z0_i = _make_z0(i)
        z_best, n_iter, success, fk_inf_best = solver.solve_fast_single(
            p=p_i, z0=z0_i, max_iter=max_iter,
            tol=float(tol_value), alpha=1.0, update_mode="line_search",
        )
        _sync_solver_devices(solver)
        solve_t = float(time.perf_counter() - t0)

        solve_times_s.append(solve_t)
        iters_run.append(float(n_iter))
        converged_mask.append(1.0 if success else 0.0)
        if success:
            iter_to_tol_vals.append(float(n_iter))
            time_to_tol_s_vals.append(solve_t)

    return {
        "name": name,
        "device": str(solver._single_kkt_device),
        "single_eval": {
            "nn_device": str(solver._single_nn_device),
            "kkt_device": str(solver._single_kkt_device),
            "kkt_backend": str(solver._single_kkt_backend),
            "nn_compile_mode": str(solver._single_nn_compile_mode),
        },
        "n_samples": n_samples,
        "tolerance": {
            "metric": tol_metric,
            "value": float(tol_value),
            "n_converged": int(sum(converged_mask)),
            "convergence_rate": float(np.mean(converged_mask)) if converged_mask else float("nan"),
        },
        "timing": {
            "solve_time_ms": _summary([1e3 * x for x in solve_times_s]),
            "iterations_run": _summary(iters_run),
            "iter_to_tol": _summary(iter_to_tol_vals),
            "time_to_tol_ms": _summary([1e3 * x for x in time_to_tol_s_vals]),
        },
        "raw": {
            "solve_time_ms": [1e3 * x for x in solve_times_s],
            "iterations_run": iters_run,
            "iter_to_tol": [it if m else None for it, m in zip(iters_run, converged_mask)],
        },
    }


@torch.no_grad()
def _measure_ipopt_profile(solver_cpu, p_eval_cpu: torch.Tensor, warmup: int) -> dict:
    nlp = solver_cpu.nlp
    n_samples = int(p_eval_cpu.shape[0])

    for i in range(int(warmup)):
        nlp.nmpc.solve(p_eval_cpu[i % n_samples].detach().cpu().numpy())

    solve_times_s: list[float] = []
    iterations: list[float] = []
    success: list[float] = []

    for i in range(n_samples):
        p_np = p_eval_cpu[i].detach().cpu().numpy()
        t0 = time.perf_counter()
        z_opt_np, info = nlp.nmpc.solve(p_np)
        solve_times_s.append(float(time.perf_counter() - t0))

        iterations.append(float(info.get("iterations", -1)))
        ok = bool(info.get("success", False) or info.get("return_status") == "Solve_Succeeded")
        success.append(1.0 if ok else 0.0)

    return {
        "name": "ipopt",
        "device": "cpu",
        "n_samples": n_samples,
        "success_rate": float(np.mean(success)) if success else float("nan"),
        "timing": {
            # External wall-clock around nlp.nmpc.solve() for fair comparison.
            "solve_time_ms": _summary([1e3 * x for x in solve_times_s]),
            "iterations": _summary(iterations),
        },
        "raw": {
            "solve_time_ms": [1e3 * x for x in solve_times_s],
            "iterations": iterations,
        },
    }


def run_eval_solver_single_speed_cstr(
    run_dir: Path,
    include_ipopt: bool = True,
    n_samples: int = 128,
    warmup: int = 16,
    max_iter: int = 100,
    tol_metric: str = "fk_inf",
    tol_value: float = 1e-6,
    eval_seed: int = 42,
    overwrite: bool = True,
    ipopt_overrides: dict | None = None,
) -> dict | None:
    """Measure the CPU single-sample solver profile (and IPOPT) for one run.

    Writes ``eval_single_speed_metrics.json`` into ``run_dir``.
    """
    profiles = ["cpu_all_ca"]

    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = (project_root() / run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    metrics_path = run_dir / "eval_single_speed_metrics.json"
    if metrics_path.exists() and not overwrite:
        print(f"Skipping existing speed evaluation (overwrite=False): {metrics_path}")
        return None

    print("Loading CPU solver...")
    solver_cpu = _load_solver_from_run_dir(run_dir, "cpu")
    if ipopt_overrides:
        solver_cpu.nlp.nmpc.override_ipopt_options(ipopt_overrides)

    torch.manual_seed(int(eval_seed))
    p_eval_cpu = solver_cpu.nlp.batch_gen_p(int(n_samples), offset=0.0).detach().cpu()
    z0_eval_cpu = solver_cpu.nlp.batch_gen_z(int(n_samples)).detach().cpu()

    metrics = {
        "meta": {
            "run_dir": str(run_dir),
            "profiles_requested": list(profiles),
            "include_ipopt": bool(include_ipopt),
            "n_samples": int(n_samples),
            "warmup": int(warmup),
            "max_iter": int(max_iter),
            "tol_metric": str(tol_metric),
            "tol_value": float(tol_value),
            "eval_seed": int(eval_seed),
            "ipopt_overrides": ipopt_overrides,
            "update_mode": "line_search",
            "warm_start_policy": "none_predictor_disabled",
        },
        "profiles": {},
        "speedup_vs_ipopt_median": {},
    }

    print("Measuring profile: cpu_all_ca")
    solver_cpu.setup_for_single_eval(nn_device="cpu", nn_dtype=torch.bfloat16)
    metrics["profiles"]["cpu_all_ca"] = _measure_solver_profile(
        name="cpu_all_ca", solver=solver_cpu, p_eval_cpu=p_eval_cpu, z0_eval_cpu=z0_eval_cpu,
        max_iter=max_iter, warmup=warmup, tol_metric=tol_metric, tol_value=tol_value,
    )

    if include_ipopt:
        print("Measuring profile: ipopt")
        metrics["profiles"]["ipopt"] = _measure_ipopt_profile(
            solver_cpu=solver_cpu, p_eval_cpu=p_eval_cpu, warmup=warmup,
        )

    if "ipopt" in metrics["profiles"]:
        ipopt_p50 = float(metrics["profiles"]["ipopt"]["timing"]["solve_time_ms"]["p50"])
        for name, m in metrics["profiles"].items():
            if name == "ipopt":
                continue
            p50 = float(m["timing"]["solve_time_ms"]["p50"])
            metrics["speedup_vs_ipopt_median"][name] = float(ipopt_p50 / p50) if p50 > 0.0 else float("nan")

    _write_json(metrics_path, metrics)

    print("=" * 70)
    print("Single-point speed evaluation complete")
    print(f"run_dir: {run_dir}")
    print(f"metrics: {metrics_path}")
    print("=" * 70)
    return metrics
