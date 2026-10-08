"""Minimal sequential sweep runner for CSTR solver experiments.

Reads a `sweep_*.json` plan and executes all runs with `status == "planned"`
by calling `run_solver_training(...)`.

On first use the shipped plan `plans/sweep_<label>.json` is copied to
`results/solver_training/_plans/sweep_<label>.json`; only that local copy is
updated (status, `save_dir`, timestamps) after each state transition. The
study scripts read the trained run folders from the local copy.

Usage:
    python3 -m src.experiments.train.run_sweep_solver_cstr
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch

from src.experiments.common import project_root
from src.experiments.train.train_solver_cstr import run_solver_training

from src.lisco import FeedforwardNNConfig, NLPConfig
from src.lisco.training import SolverTrainConfig
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import NMPCParams


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_plan_path(sweep_label: str) -> Path:
    return project_root() / "results" / "solver_training" / "_plans" / f"sweep_{sweep_label}.json"


def _load_plan(plan_path: Path) -> dict:
    if not plan_path.exists():
        raise FileNotFoundError(f"Sweep plan not found: {plan_path}")
    with open(plan_path) as f:
        plan = json.load(f)
    if not isinstance(plan, dict):
        raise ValueError(f"Plan file must contain a JSON object: {plan_path}")
    plan.setdefault("runs", [])
    if not isinstance(plan["runs"], list):
        raise ValueError(f"Plan field 'runs' must be a list: {plan_path}")
    plan.setdefault("plan_id", plan_path.stem)
    plan.setdefault("created_at", _now_str())
    return plan


def _write_plan(plan_path: Path, plan: dict) -> None:
    plan["updated_at"] = _now_str()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)


def _dtype_from_json(value):
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected dtype string, got {type(value).__name__}")

    if value.startswith("torch."):
        value = value.split(".", 1)[1]

    if not hasattr(torch, value):
        raise ValueError(f"Unsupported torch dtype string: {value!r}")

    dtype = getattr(torch, value)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Resolved torch.{value} is not a torch.dtype")
    return dtype


def _status_counts(runs: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        status = "invalid"
        if isinstance(run, dict):
            status = str(run.get("status", "planned"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _rebuild_runtime_config(cfg_json: dict) -> dict:
    """Convert serialized JSON config back to the dataclass-based runtime config."""
    if not isinstance(cfg_json, dict):
        raise TypeError(f"Run config must be a dict, got {type(cfg_json).__name__}")

    cfg = copy.deepcopy(cfg_json)

    cfg["model"] = CSTRParams.from_dict(cfg["model"])
    cfg["nmpc"] = NMPCParams.from_dict(cfg["nmpc"])

    nlp_cfg = dict(cfg["nlp"])
    if "dtype" in nlp_cfg:
        nlp_cfg["dtype"] = _dtype_from_json(nlp_cfg["dtype"])
    cfg["nlp"] = NLPConfig(**nlp_cfg)

    nn_cfg = dict(cfg["nn"])
    if "nn_dtype" in nn_cfg:
        nn_cfg["nn_dtype"] = _dtype_from_json(nn_cfg["nn_dtype"])
    cfg["nn"] = FeedforwardNNConfig(**nn_cfg)

    cfg["train"] = SolverTrainConfig(**cfg["train"])

    return cfg


if __name__ == "__main__":
    # Edit this label to choose which sweep file to run.
    sweep_label = "solver_cstr_paper"
    dry_run = False

    plan_path = _default_plan_path(sweep_label)
    if not plan_path.exists():
        # Work on a local copy; the shipped plan is never modified.
        shipped_path = project_root() / "plans" / f"sweep_{sweep_label}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shipped_path, plan_path)
        print(f"Copied shipped plan {shipped_path} -> {plan_path}")
    plan = _load_plan(plan_path)
    runs = plan["runs"]

    planned_indices = [
        i
        for i, run in enumerate(runs)
        if isinstance(run, dict) and str(run.get("status", "planned")) == "planned"
    ]

    counts0 = _status_counts(runs)
    print("=" * 70)
    print("Solver Sweep Runner")
    print(f"plan_path: {plan_path}")
    print(f"plan_id: {plan.get('plan_id')}")
    print(f"runs_total: {len(runs)}")
    print(f"planned: {len(planned_indices)}")
    print(f"status_counts: {counts0}")
    if dry_run:
        print("Mode: dry_run (no training will be executed)")
    print("=" * 70)

    if len(planned_indices) == 0:
        print("No planned runs found. Nothing to do.")
        raise SystemExit(0)

    if dry_run:
        for idx in planned_indices:
            run = runs[idx]
            print(f"- {run.get('plan_run_id', f'run_{idx:03d}')} (status={run.get('status')})")
        raise SystemExit(0)

    n_success = 0

    for order, idx in enumerate(planned_indices, start=1):
        run = runs[idx]
        run_id = str(run.get("plan_run_id", f"run_{idx:03d}"))
        print(f"\n[{order}/{len(planned_indices)}] {run_id}")

        if "config" not in run:
            raise KeyError(f"Run {run_id} is missing required field 'config'")

        # Persist start metadata before training starts. Status stays planned
        # so the plan only uses the two states: planned/done.
        run["status"] = "planned"
        run["started_at"] = _now_str()
        run["finished_at"] = None
        run["duration_sec"] = None
        _write_plan(plan_path, plan)

        t0 = time.perf_counter()
        runtime_cfg = _rebuild_runtime_config(run["config"])
        solver, train_logger, nlp, save_dir = run_solver_training(runtime_cfg)
        # Explicitly drop large refs before next run (best-effort).
        del solver, train_logger, nlp

        run["status"] = "done"
        run["save_dir"] = str(save_dir)
        run["finished_at"] = _now_str()
        run["duration_sec"] = round(time.perf_counter() - t0, 3)
        _write_plan(plan_path, plan)

        n_success += 1
        print(f"  done: {save_dir}")

    counts1 = _status_counts(runs)
    print("\n" + "=" * 70)
    print("Sweep Run Summary")
    print(f"success: {n_success}")
    print(f"status_counts: {counts1}")
    print(f"plan_path: {plan_path}")
    print("=" * 70)
