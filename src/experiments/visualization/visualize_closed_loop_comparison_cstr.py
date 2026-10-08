"""Closed-loop CSTR comparison: IPOPT vs LISCO vs approx. MPC (interactive).

Run cell-by-cell (``# %%``).

Controllers: LISCO = the plan entry tagged ``closed_loop`` that was trained
with closed-loop resampling; the second ``closed_loop`` entry (uniform
resampling) is the extra LISCO arm of Table 2; approx. MPC = the model trained
by ``train_approx_mpc_cstr.py`` (``APPROX_MPC_DIR``).

Stage 1 (``RUN_EVAL``): build one shared ``x0_batch``, run each controller's
    closed-loop eval against it -> ``OUTPUT_DIR/metrics/<controller>/``.
Stage 2 (cheap, re-run freely): load those JSONs and build the two tables:

- Table 1  comparison_table.tex (tbl:closed_loop_comparison)
- Table 2  step_bin_table.tex (tbl:step_bin_convergence)

LISCO's ``solver_config.json`` is the authority for the NLP; the approx. MPC's
``problem_meta.json`` is checked against it before stage 1, and IPOPT is built
from the same params (no run folder needed).  All arms share ``x0_batch``, so
the overlay and the cost gap are paired per trajectory.  Cached results produced
under other settings are rejected, not mixed in.

Both solver arms are cold-started: LISCO draws a fresh random z0 per step, and
``CSTRNMPC.solve`` re-uses its fixed ``w0``.
"""

# %% Imports
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments.common import (
    DEFAULT_PLAN_PATH,
    HSLLIB_PATH,
    load_plan_runs,
    project_root,
)
from src.experiments.eval.eval_approx_mpc_closed_loop_cstr import (
    run_approx_mpc_closed_loop_eval,
)
from src.experiments.eval.eval_closed_loop_base import (
    ClosedLoopEvalConfig,
    sample_x0_batch_uniform,
    summary_stats,
)
from src.experiments.eval.eval_ipopt_closed_loop_cstr import run_ipopt_closed_loop_eval
from src.experiments.eval.eval_solver_closed_loop_cstr import (
    SolverStepConfig,
    run_solver_closed_loop_eval,
)
from src.experiments.train.train_approx_mpc_cstr import APPROX_MPC_DIR
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams


# %% Configuration
# IPOPT is absent: it is built from LISCO's params, so there is nothing to point at.
PLAN_PATH = DEFAULT_PLAN_PATH
_LISCO_RUNS = {
    run["config"]["train"]["resampling_strategy"]: run["run_dir"]
    for run in load_plan_runs("closed_loop", PLAN_PATH)
}
CONTROLLERS = {
    "lisco":      _LISCO_RUNS["closed_loop"],
    "approx_mpc": APPROX_MPC_DIR,
}

CONTROLLER_ORDER = ["ipopt", "lisco", "approx_mpc"]
CONTROLLER_LABELS = {"ipopt": "IPOPT", "lisco": "LISCO", "approx_mpc": "approx. MPC"}

CLOSED_LOOP_KWARGS = {
    "n_trajectories": 50,
    "n_sim": 50,
    "viol_tol": 1e-5,
    "seed": 42,
}

SOLVER_STEP_KWARGS = {"max_iter": 300, "fk_inf_tol": 1e-5, "alpha": 1.0}
SOLVER_PROFILE = "cpu_all_ca"

# Table 2 only -- the uniform (open-loop) trained counterpart of CONTROLLERS["lisco"],
# run through the same closed-loop protocol on the same x0_batch.  Deliberately kept
# out of CONTROLLERS / CONTROLLER_ORDER so Table 1 is unaffected.
LISCO_UNIFORM = _LISCO_RUNS["all"]
STEP_BINS = [(1,1), (2, 5), (6, 10), (11, 25), (26, 50)]  # 1-indexed plant steps, inclusive

IPOPT_OVERRIDES = {
    "mu_target": 5e-3,
    "tol": 1e-5,
    "linear_solver": "ma27",
    "hsllib": HSLLIB_PATH,
}

OUTPUT_DIR = Path("results/_aggregated/closed_loop_comparison_nr5_N50")

RUN_EVAL = False        # stage 1: expensive
EVAL_OVERWRITE = False  # stage 1: re-run controllers that already have results
SAVE_FIGURES = True    # export cell


# %% Helpers -- files and validation
def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _normalize_problem_meta(meta: dict) -> dict:
    """Round-trip through the dataclasses so field order cancels."""
    return {
        "model_params": CSTRParams.from_dict(meta["model_params"]).to_dict(),
        "nmpc_params": NMPCParams.from_dict(meta["nmpc_params"]).to_dict(),
    }


def _metrics_dir(name: str, output_dir: Path) -> Path:
    return output_dir / "metrics" / name


def _results_path(name: str, output_dir: Path) -> Path:
    return _metrics_dir(name, output_dir) / "closed_loop_results.json"


def prepare_controllers(controllers: dict) -> dict:
    """Resolve run dirs and pin the shared NLP down before stage 1 spends time."""
    solver_dir = _resolve(controllers["lisco"])
    approx_mpc_dir = _resolve(controllers["approx_mpc"])

    cfg_path = solver_dir / "solver_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing solver config: {cfg_path}")
    reference_meta = _normalize_problem_meta(_read_json(cfg_path))

    approx_meta_path = approx_mpc_dir / "problem_meta.json"
    if not approx_meta_path.exists():
        raise FileNotFoundError(f"Missing approx. MPC problem meta: {approx_meta_path}")
    if _normalize_problem_meta(_read_json(approx_meta_path)) != reference_meta:
        raise ValueError(
            "problem_meta mismatch: the approx. MPC was not trained on the same NLP "
            f"as LISCO.\n  lisco:      {cfg_path}\n  approx_mpc: {approx_meta_path}"
        )

    model_params = CSTRParams.from_dict(reference_meta["model_params"])
    nmpc_params = NMPCParams.from_dict(reference_meta["nmpc_params"])
    print(f"nr={model_params.nr}, N={nmpc_params.N}, objective={nmpc_params.objective}")
    print(f"  lisco:      {solver_dir}")
    print(f"  approx_mpc: {approx_mpc_dir}")
    return {
        "solver_dir": solver_dir,
        "approx_mpc_dir": approx_mpc_dir,
        "model_params": model_params,
        "nmpc_params": nmpc_params,
        "problem_meta": reference_meta,
    }


def _check_result_meta(results: dict, name: str) -> None:
    """Reject cached results produced under different settings than requested."""
    hint = "Re-run stage 1 with EVAL_OVERWRITE=True."
    cfg = results["config"]

    expected = dict(CLOSED_LOOP_KWARGS)
    if name == "lisco":
        expected.update(SOLVER_STEP_KWARGS, profile=SOLVER_PROFILE)
    elif name == "ipopt":
        expected["ipopt_overrides"] = IPOPT_OVERRIDES

    for key, want in expected.items():
        if key in cfg and cfg[key] != want:
            raise ValueError(
                f"Cached results for {name} were produced with {key}={cfg[key]!r}, "
                f"but this study requests {key}={want!r}. {hint}"
            )
    if not cfg.get("x0_batch_provided"):
        raise ValueError(f"Cached {name} results did not use the shared x0_batch. {hint}")


# %% Helpers -- evaluation
def build_or_load_x0_batch(prepared: dict, output_dir: Path,
                           overwrite: bool = False) -> torch.Tensor:
    """The one x0 batch every controller is evaluated on."""
    ref_dir = output_dir / "shared_reference"
    batch_path, meta_path = ref_dir / "x0_batch.pt", ref_dir / "reference.json"
    expected = {
        "seed": CLOSED_LOOP_KWARGS["seed"],
        "n_trajectories": CLOSED_LOOP_KWARGS["n_trajectories"],
        "problem_meta": prepared["problem_meta"],
    }

    if not overwrite and batch_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        for key, want in expected.items():
            if meta.get(key) != want:
                raise ValueError(
                    f"Cached shared reference has {key}={meta.get(key)!r}, but this "
                    f"study requests {key}={want!r}. Re-run stage 1 with "
                    "EVAL_OVERWRITE=True."
                )
        x0_batch = torch.load(batch_path, map_location="cpu")
        print(f"Loaded shared x0_batch {tuple(x0_batch.shape)} from {ref_dir}")
        return x0_batch

    nmpc = CSTRNMPC(prepared["model_params"], prepared["nmpc_params"])
    x0_batch = torch.as_tensor(
        sample_x0_batch_uniform(nmpc, int(expected["n_trajectories"]),
                                int(expected["seed"])),
        dtype=torch.float64,
    )
    ref_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x0_batch, batch_path)
    _write_json(meta_path, {"nx": int(nmpc.nx), **expected})
    print(f"Wrote shared x0_batch {tuple(x0_batch.shape)} to {ref_dir}")
    return x0_batch


def run_evaluations(prepared: dict, output_dir: Path,
                    overwrite: bool = False) -> list[Path]:
    """Run every controller's closed-loop eval on the shared x0 batch."""
    x0_batch = build_or_load_x0_batch(prepared, output_dir, overwrite=overwrite)
    config = ClosedLoopEvalConfig(x0_batch=x0_batch, **CLOSED_LOOP_KWARGS)
    written: list[Path] = []

    for i, name in enumerate(CONTROLLER_ORDER, start=1):
        path = _results_path(name, output_dir)
        if path.exists() and not overwrite:
            print(f"[{i}/{len(CONTROLLER_ORDER)}] skip existing: {name} -> {path}")
            continue

        print(f"\n[{i}/{len(CONTROLLER_ORDER)}] closed-loop eval: {name} ...")
        save_dir = _metrics_dir(name, output_dir)
        if name == "ipopt":
            run_ipopt_closed_loop_eval(
                model_params=prepared["model_params"],
                nmpc_params=prepared["nmpc_params"],
                config=config,
                ipopt_overrides=IPOPT_OVERRIDES,
                run_dir=save_dir,
            )
        elif name == "lisco":
            run_solver_closed_loop_eval(
                prepared["solver_dir"], config, SolverStepConfig(**SOLVER_STEP_KWARGS),
                profile=SOLVER_PROFILE, save_dir=save_dir,
            )
        elif name == "approx_mpc":
            run_approx_mpc_closed_loop_eval(
                prepared["approx_mpc_dir"], config, save_dir=save_dir,
            )
        written.append(path)

    # Table 2's second arm.  Same protocol, same config (hence the same x0_batch);
    # kept outside the loop above so CONTROLLER_ORDER stays a three-arm list.
    path = _results_path("lisco_uniform", output_dir)
    if path.exists() and not overwrite:
        print(f"skip existing: lisco_uniform -> {path}")
    else:
        print("\nclosed-loop eval: lisco_uniform ...")
        run_solver_closed_loop_eval(
            _resolve(LISCO_UNIFORM), config, SolverStepConfig(**SOLVER_STEP_KWARGS),
            profile=SOLVER_PROFILE, save_dir=_metrics_dir("lisco_uniform", output_dir),
        )
        written.append(path)

    return written


# %% Helpers -- loading
def load_results(prepared: dict, output_dir: Path) -> dict[str, dict]:
    """Load the cached per-controller results, validating as we go."""
    results_by_ctrl: dict[str, dict] = {}
    for name in CONTROLLER_ORDER:
        path = _results_path(name, output_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing results for {name}: {path}. Run stage 1 with RUN_EVAL=True."
            )
        results_by_ctrl[name] = _read_json(path)
        _check_result_meta(results_by_ctrl[name], name)

    ref = CONTROLLER_ORDER[0]
    ref_x0 = np.asarray(results_by_ctrl[ref]["x0_batch"], dtype=np.float64)
    for name in CONTROLLER_ORDER:
        x0 = np.asarray(results_by_ctrl[name]["x0_batch"], dtype=np.float64)
        if x0.shape != ref_x0.shape or not np.array_equal(x0, ref_x0):
            raise ValueError(f"x0_batch mismatch between {ref} and {name}.")
        if _normalize_problem_meta(results_by_ctrl[name]["problem_meta"]) != \
                prepared["problem_meta"]:
            raise ValueError(
                f"problem_meta of cached {name} results differs from the prepared NLP. "
                "Re-run stage 1 with EVAL_OVERWRITE=True."
            )

    print(f"Loaded {len(results_by_ctrl)} controllers: {ref_x0.shape[0]} trajectories "
          f"x {CLOSED_LOOP_KWARGS['n_sim']} steps.")
    for name in CONTROLLER_ORDER:
        print(f"  {CONTROLLER_LABELS[name]:12s} "
              f"failed={results_by_ctrl[name]['summary']['n_failed']}")
    return results_by_ctrl


# %% Table
# Row label -> path into the results JSON.  Failures come first: total_cost sums
# however many steps ran before the state went non-finite, so a nonzero failure
# count makes that arm's cost rows incomparable.
_TABLE_ROWS = {
    "failed trajectories":    ("summary", "n_failed"),
    "failure rate":           ("summary", "failure_rate"),
    "cost (min)":             ("aggregates", "total_cost", "min"),
    "cost (mean)":            ("aggregates", "total_cost", "mean"),
    "cost (median)":          ("aggregates", "total_cost", "median"),
    "stage cost (mean)":      ("aggregates", "stage_cost", "mean"),
    "solve time median [s]":  ("aggregates", "step_time", "median"),
    "solve time p90 [s]":     ("aggregates", "step_time", "p90"),
    "solve time p99 [s]":     ("aggregates", "step_time", "p99"),
    "solve time max [s]":     ("aggregates", "step_time", "max"),
}


def _train_time_h(run_dir: Path) -> float:
    """Training wall-clock in hours, from the run's training_log.json."""
    path = run_dir / "training_log.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing training log for training time: {path}")
    value = _read_json(path)["train_time"]
    if isinstance(value, list):  # older logs stored a per-epoch list
        value = value[-1]
    return float(value) / 3600.0


def _sampling_time_h(approx_mpc_dir: Path) -> float:
    """Dataset sampling wall-clock in hours -- two hops from the run dir.

    The sampling time is recorded by the sampler, not the trainer, so it lives in
    the dataset's meta.json; config.json's data_dir is the only link to it.
    """
    cfg_path = approx_mpc_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing approx. MPC config: {cfg_path}")
    meta_path = _resolve(_read_json(cfg_path)["data_dir"]) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing dataset meta for sampling time: {meta_path} "
            f"(referenced by {cfg_path} data_dir)"
        )
    return float(_read_json(meta_path)["sampling_stats"]["sampling_time_s"]) / 3600.0


def _dig(results: dict, path: tuple[str, ...]):
    for key in path:
        results = results[key]
    return results


def build_comparison_table(results_by_ctrl: dict[str, dict],
                           prepared: dict) -> pd.DataFrame:
    """Table 1: the main result.

    ``prepared`` carries the run dirs: the offline-cost rows are read from the
    training artifacts, not from the closed-loop results.
    """
    rows = [(label, [_dig(results_by_ctrl[n], path) for n in CONTROLLER_ORDER])
            for label, path in _TABLE_ROWS.items()]

    # Max violation per trajectory (one value per trajectory), not pooled per-step values.
    for prefix, key in (("input viol.", "max_input_viol"), ("state viol.", "max_state_viol")):
        stats_by_ctrl = [summary_stats(results_by_ctrl[n]["per_trajectory"][key])
                         for n in CONTROLLER_ORDER]
        for stat in ("mean", "max"):
            rows.append((f"{prefix} ({stat})", [s[stat] for s in stats_by_ctrl]))

    # Offline cost.  NaN marks "not applicable" -- IPOPT trains nothing, and
    # LISCO needs no sampled dataset -- and paper_table_to_latex renders it as a dash.
    offline = {
        "sampling time [h]": {"approx_mpc": _sampling_time_h(prepared["approx_mpc_dir"])},
        "training time [h]": {"lisco": _train_time_h(prepared["solver_dir"]),
                              "approx_mpc": _train_time_h(prepared["approx_mpc_dir"])},
    }
    for label, by_ctrl in offline.items():
        rows.append((label, [by_ctrl.get(n, float("nan")) for n in CONTROLLER_ORDER]))

    table = pd.DataFrame(
        [values for _, values in rows],
        index=pd.Index([label for label, _ in rows], name="metric"),
        columns=[CONTROLLER_LABELS[name] for name in CONTROLLER_ORDER],
    )
    table.columns.name = "controller"
    return table


LISCO_ARMS = {"closed-loop trained": "lisco", "uniform trained": "lisco_uniform"}


def build_step_bin_table(output_dir: Path) -> pd.DataFrame:
    """Table 2: how deep into the horizon each LISCO arm holds the residual tolerance.

    A step counts as converged when the solve reached ``fk_inf_tol`` -- the flag the
    solver itself returns, stored per step by the closed-loop eval.  The denominator
    is fixed at ``n_trajectories * bin width``: a trajectory that diverged is
    truncated, and its missing steps count as not converged rather than shrinking
    the bin, so the two columns stay comparable.
    """
    n_traj = int(CLOSED_LOOP_KWARGS["n_trajectories"])
    n_sim = int(CLOSED_LOOP_KWARGS["n_sim"])

    columns = {}
    for label, name in LISCO_ARMS.items():
        converged = np.zeros((n_traj, n_sim), dtype=bool)
        for i, traj in enumerate(_read_json(_results_path(name, output_dir))["trajectories"]):
            flags = traj["converged"]
            converged[i, :len(flags)] = flags
        columns[label] = [float(converged[:, lo - 1:hi].mean()) for lo, hi in STEP_BINS]
        columns[label].append(float(converged.mean()))

    index = [f"steps {lo}-{hi}" for lo, hi in STEP_BINS] + [f"all 1-{n_sim}"]
    table = pd.DataFrame(columns, index=pd.Index(index, name="step bin"))
    table.columns.name = "LISCO"
    return table


def _paper_blocks(table: pd.DataFrame) -> list[list[tuple[str, str, float, int, str]]]:
    """Paper rows as (source label, display label, scale, decimals, suffix) blocks.

    A subset of the full table: cost, solve times, violations, offline cost.
    """
    violations = [(label, label, 1.0, 4, "")
                  for label in table.index if "viol." in str(label)]
    return [
        [("cost (mean)", "closed-loop cost (mean)", 1.0, 4, "")],
        [("solve time median [s]", "solve time median [ms]", 1e3, 2, ""),
         ("solve time p90 [s]", "solve time p90 [ms]", 1e3, 2, ""),
         ("solve time max [s]", "solve time max [ms]", 1e3, 2, "")],
        violations,
        [("sampling time [h]", "sampling time [h]", 1.0, 2, ""),
         ("training time [h]", "training time [h]", 1.0, 2, "")],
    ]


def paper_table_to_latex(table: pd.DataFrame) -> str:
    """Bare booktabs tabular for the paper -- no float, caption or label.

    The paper's main .tex owns the table environment.  Blocks are separated by a
    \\midrule; each row carries its own unit scale and precision, so the seconds
    stored in the frame print as fixed-point milliseconds here.
    """
    lines = [
        f"\\begin{{tabular}}{{{'l' + 'r' * len(table.columns)}}}",
        "\\toprule",
        " & ".join(["metric"] + [str(c) for c in table.columns]) + " \\\\",
        "\\midrule",
    ]
    for block in _paper_blocks(table):
        for source, label, scale, decimals, suffix in block:
            cells = ["--" if np.isnan(v) else f"{v * scale:.{decimals}f}{suffix}"
                     for v in table.loc[source].to_numpy(dtype=float)]
            lines.append(" & ".join([label] + cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"  # the last block's separator closes the table
    return "\n".join(lines + ["\\end{tabular}"])


def step_bin_table_to_latex(table: pd.DataFrame) -> str:
    """Table 2 as a bare booktabs tabular -- no float, caption or label.

    Fractions print as percentages, matching how the main paper table renders its
    violation fractions, and the bin labels get en-dashes.
    """
    lines = [
        f"\\begin{{tabular}}{{{'l' + 'r' * len(table.columns)}}}",
        "\\toprule",
        " & ".join([str(table.index.name)] + [str(c) for c in table.columns]) + " \\\\",
        "\\midrule",
    ]
    for idx in table.index:
        cells = [f"{v * 100.0:.1f}\\%" for v in table.loc[idx].to_numpy(dtype=float)]
        lines.append(" & ".join([str(idx).replace("-", "--")] + cells) + " \\\\")
    lines.insert(-1, "\\midrule")  # separate the all-steps summary row
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


# %% Export
def export_all(results_by_ctrl: dict[str, dict], prepared: dict,
               output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    table = build_comparison_table(results_by_ctrl, prepared)
    step_bin_table = build_step_bin_table(output_dir)
    for name, text in (("comparison_table.tex", paper_table_to_latex(table)),
                       ("step_bin_table.tex", step_bin_table_to_latex(step_bin_table))):
        path = output_dir / name
        path.write_text(text + "\n")
        written.append(path)

    path = output_dir / "study_config.json"
    _write_json(path, {
        "controllers": CONTROLLERS,
        "lisco_uniform": LISCO_UNIFORM,
        "closed_loop_kwargs": CLOSED_LOOP_KWARGS,
        "solver_step_kwargs": SOLVER_STEP_KWARGS,
        "solver_profile": SOLVER_PROFILE,
        "ipopt_overrides": IPOPT_OVERRIDES,
    })
    written.append(path)
    return written


# %% Stage 1 -- run evaluations (expensive; set RUN_EVAL = True)
prepared = prepare_controllers(CONTROLLERS)
if RUN_EVAL:
    paths = run_evaluations(prepared, _resolve(OUTPUT_DIR), overwrite=EVAL_OVERWRITE)
    print(f"\nWrote {len(paths)} results files.")


# %% Stage 2 -- load cached results
results_by_ctrl = load_results(prepared, _resolve(OUTPUT_DIR))


# %% Table 1 -- main comparison
comparison_table = build_comparison_table(results_by_ctrl, prepared)
print(paper_table_to_latex(comparison_table))
comparison_table


# %% Table 2 -- convergence by step bin (LISCO arms)
step_bin_table = build_step_bin_table(_resolve(OUTPUT_DIR))
print(step_bin_table_to_latex(step_bin_table))
step_bin_table


# %% Export -- tables (set SAVE_FIGURES = True)
if SAVE_FIGURES:
    out = _resolve(OUTPUT_DIR)
    paths = export_all(results_by_ctrl, prepared, out)
    print(f"Wrote {len(paths)} files to {out}")
    for p in paths:
        print(f"  {p.name}")

