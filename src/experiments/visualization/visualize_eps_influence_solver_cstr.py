"""Epsilon-influence study for CSTR solver runs (interactive, cell-based).

Run cell-by-cell (``# %%``).

Runs: the plan entries tagged ``smoothing`` in the local sweep plan
(``PLAN_PATH``), i.e. the six nr=2, N=50 runs with eps = 1e-1 ... 1e-6.

Stage 1 (``RUN_EVAL``): build one shared p_eval / z0_eval / IPOPT reference,
    then evaluate every run in ``RUNS`` against it.  One JSON per run in
    ``OUTPUT_DIR/metrics``.
Stage 2 (cheap, re-run freely): load those JSONs and build the figures.

- Figure 1  convergence fraction of ``fk_inf`` over iterations
            (paper: fig1_convergence_fraction.pdf, fig:eps_convergence_fraction).
- Figure 2  final relative objective gap vs the IPOPT reference
            (paper: fig2_relative_gap.pdf, fig:eps_relative_gap).

The IPOPT reference runs at tight ``tol`` with no ``mu_target`` override, so it
stays close to the original NLP while the solvers vary ``eps``.
"""

# %% Imports
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch

plt.style.use(["science", "ieee", "no-latex"])

from src.experiments.common import (
    DEFAULT_PLAN_PATH,
    HSLLIB_PATH,
    basic_eval_solver,
    load_plan_runs,
    project_root,
)
from src.experiments.eval.eval_solver_cstr import (
    _load_solver_from_run_dir,
    _sample_ipopt_feasible_batch,
)


# %% Configuration
PLAN_PATH = DEFAULT_PLAN_PATH
RUNS = [
    {
        "run_dir": run["run_dir"],
        "label": f"eps=1e{int(round(np.log10(run['config']['nlp']['eps'])))}",
    }
    for run in load_plan_runs("smoothing", PLAN_PATH)
]

EVAL_KWARGS = {
    "n_samples": 256,
    "max_iter": 300,
    "eval_seed": 42,
    "tol_fk": 1e-5,
    "convergence_tols": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "update_mode": "line_search",
}

IPOPT_OVERRIDES = {
    "tol": 1e-8,
    "linear_solver": "ma27",
    "hsllib": HSLLIB_PATH,
}

FIG_TOL = 1e-5
OUTPUT_DIR = Path("results/solver_training/_aggregated/eps_influence_solver_cstr")

RUN_EVAL = False
EVAL_OVERWRITE = False
SAVE_FIGURES = True


# %% Helpers -- files and run metadata
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


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor in {path}, got {type(value)}")
    return value


def _tol_key(tol: float) -> str:
    """Cache key, not a label: indexes the metrics JSON.  Format is load-bearing."""
    return f"{float(tol):.0e}"


def _pow10(value: float) -> str:
    """Math-mode body for a power of ten: 1e-5 -> ``10^{-5}``.

    No ``$`` so it composes into larger math strings.  Mantissa is assumed 1,
    which holds for every eps and tolerance in this study.
    """
    return rf"10^{{{int(round(np.log10(float(value))))}}}"


_METRIC_MATH = {
    "fk_inf": r"\|\mathbf{r}_\phi\|_\infty",
    "kkt_inf": r"\|\mathbf{r}\|_\infty",
}


def _safe_name(label: str) -> str:
    return label.replace("=", "_").replace("/", "_").replace(" ", "_")


def _solver_config_without_eps(cfg: dict, run_dir: Path) -> dict:
    nlp_params = cfg.get("nlp_params")
    if not isinstance(nlp_params, dict):
        raise ValueError(f"Missing or invalid nlp_params in {run_dir / 'solver_config.json'}")
    if "eps" not in nlp_params:
        raise ValueError(f"Missing nlp_params.eps in {run_dir / 'solver_config.json'}")

    out = dict(cfg)
    out_nlp = dict(nlp_params)
    out_nlp.pop("eps")
    out["nlp_params"] = out_nlp
    return out


def prepare_runs(run_specs: list[dict]) -> list[dict]:
    """Resolve run dirs, infer eps, and reject config differences beyond eps."""
    runs: list[dict] = []
    reference_cfg = None
    reference_run = None

    for spec in run_specs:
        run_dir = _resolve(spec["run_dir"])
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        cfg = _read_json(run_dir / "solver_config.json")
        cfg_wo_eps = _solver_config_without_eps(cfg, run_dir)
        if reference_cfg is None:
            reference_cfg = cfg_wo_eps
            reference_run = run_dir
        elif cfg_wo_eps != reference_cfg:
            raise ValueError(
                "Solver configurations are not identical except for nlp_params.eps.\n"
                f"reference_run: {reference_run}\n"
                f"mismatch_run:  {run_dir}"
            )

        eps = float(spec.get("eps", cfg["nlp_params"]["eps"]))
        runs.append({
            "label": str(spec.get("label", f"eps={_tol_key(eps)}")),
            "run_dir": run_dir,
            "eps": eps,
        })

    if not runs:
        raise ValueError("RUNS must contain at least one run.")
    return sorted(runs, key=lambda run: float(run["eps"]))


def _metrics_path(run: dict, output_dir: Path) -> Path:
    return output_dir / "metrics" / f"{_safe_name(run['label'])}.json"


def _reference_paths(output_dir: Path) -> dict[str, Path]:
    ref_dir = output_dir / "shared_reference"
    return {
        "dir": ref_dir,
        "meta": ref_dir / "reference.json",
        "p_eval": ref_dir / "p_eval.pt",
        "z0_eval": ref_dir / "z0_eval.pt",
        "z_opt": ref_dir / "z_opt.pt",
    }


def _check_eval_meta(meta: dict, run: dict) -> None:
    # n_samples is skipped: the batch comes from the shared reference, which may
    # hold fewer samples than requested.  _check_reference_meta covers it.
    for key, want in EVAL_KWARGS.items():
        if key == "n_samples":
            continue
        if key in meta and meta[key] != want:
            raise ValueError(
                f"Cached metrics for {run['label']} were produced with "
                f"{key}={meta[key]!r}, but this study requests {key}={want!r}. "
                "Re-run stage 1 with EVAL_OVERWRITE=True."
            )
    if meta.get("p_eval_source") != "external" or meta.get("z0_eval_source") != "external":
        raise ValueError(
            f"Cached metrics for {run['label']} were not produced with shared p_eval/z0_eval. "
            "Re-run stage 1 with EVAL_OVERWRITE=True."
        )
    if not bool(meta.get("has_z_opt")):
        raise ValueError(
            f"Cached metrics for {run['label']} have no IPOPT reference. "
            "Re-run stage 1 with EVAL_OVERWRITE=True."
        )


def _check_reference_meta(meta: dict) -> None:
    expected = {
        "eval_seed": int(EVAL_KWARGS["eval_seed"]),
        "n_samples_requested": int(EVAL_KWARGS["n_samples"]),
        "ipopt_overrides": IPOPT_OVERRIDES,
    }
    for key, want in expected.items():
        if meta.get(key) != want:
            raise ValueError(
                f"Cached shared reference has {key}={meta.get(key)!r}, "
                f"but this study requests {key}={want!r}. "
                "Re-run stage 1 with EVAL_OVERWRITE=True."
            )


# %% Helpers -- evaluation
def build_or_load_reference(runs: list[dict], output_dir: Path,
                            overwrite: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Shared p, z0 and IPOPT reference for all eps runs."""
    paths = _reference_paths(output_dir)
    if (
        not overwrite
        and paths["p_eval"].exists()
        and paths["z0_eval"].exists()
        and paths["z_opt"].exists()
        and paths["meta"].exists()
    ):
        meta = _read_json(paths["meta"])
        _check_reference_meta(meta)
        print(f"Loaded shared reference from {paths['dir']}")
        return (
            _load_tensor(paths["p_eval"]),
            _load_tensor(paths["z0_eval"]),
            _load_tensor(paths["z_opt"]),
            meta,
        )

    paths["dir"].mkdir(parents=True, exist_ok=True)
    ref_solver = _load_solver_from_run_dir(runs[0]["run_dir"])
    ref_solver.nlp.nmpc.override_ipopt_options(IPOPT_OVERRIDES)

    p_eval, z_opt, ipopt_stats = _sample_ipopt_feasible_batch(
        nlp=ref_solver.nlp,
        n_samples=int(EVAL_KWARGS["n_samples"]),
        eval_seed=int(EVAL_KWARGS["eval_seed"]),
    )

    torch.manual_seed(int(EVAL_KWARGS["eval_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(EVAL_KWARGS["eval_seed"]))
    ref_solver.nlp.setup_for_batch(int(p_eval.shape[0]), force_rebuild=False)
    z0_eval = ref_solver.nlp.batch_gen_z(int(p_eval.shape[0]))

    torch.save(p_eval.detach().cpu(), paths["p_eval"])
    torch.save(z0_eval.detach().cpu(), paths["z0_eval"])
    torch.save(z_opt.detach().cpu(), paths["z_opt"])

    meta = {
        "mode": "shared_ipopt_reference",
        "reference_run_dir": str(runs[0]["run_dir"]),
        "p_eval_path": str(paths["p_eval"]),
        "z0_eval_path": str(paths["z0_eval"]),
        "z_opt_path": str(paths["z_opt"]),
        "eval_seed": int(EVAL_KWARGS["eval_seed"]),
        "n_samples_requested": int(EVAL_KWARGS["n_samples"]),
        "n_samples": int(p_eval.shape[0]),
        "ipopt_overrides": IPOPT_OVERRIDES,
        **ipopt_stats,
    }
    _write_json(paths["meta"], meta)
    print(f"Wrote shared reference to {paths['dir']}")
    return p_eval.detach().cpu(), z0_eval.detach().cpu(), z_opt.detach().cpu(), meta


def run_evaluations(runs: list[dict], output_dir: Path, overwrite: bool = False) -> list[Path]:
    """Evaluate every eps run against the shared reference and cache metrics."""
    p_eval, z0_eval, z_opt, reference_meta = build_or_load_reference(
        runs, output_dir, overwrite=overwrite
    )
    written: list[Path] = []

    for i, run in enumerate(runs, start=1):
        path = _metrics_path(run, output_dir)
        if path.exists() and not overwrite:
            print(f"[{i}/{len(runs)}] skip existing: {run['label']} -> {path.name}")
            continue

        print(f"[{i}/{len(runs)}] evaluating {run['label']} ...")
        solver = _load_solver_from_run_dir(run["run_dir"])
        metrics = basic_eval_solver(
            solver=solver,
            nlp=solver.nlp,
            p_eval=p_eval,
            z0_eval=z0_eval,
            z_opt=z_opt,
            **EVAL_KWARGS,
        )
        metrics["study"] = {
            "label": run["label"],
            "run_dir": str(run["run_dir"]),
            "eps": float(run["eps"]),
            "reference": reference_meta,
            "ipopt_overrides": IPOPT_OVERRIDES,
        }
        _write_json(path, metrics)
        written.append(path)

    return written


# %% Helpers -- loading and summaries
def load_runs(runs: list[dict], output_dir: Path) -> list[dict]:
    """Load cached per-run metrics for plotting and tables."""
    loaded: list[dict] = []
    skipped: list[str] = []

    for run in runs:
        path = _metrics_path(run, output_dir)
        if not path.exists():
            skipped.append(f"{run['label']}: no metrics at {path}")
            continue

        metrics = _read_json(path)
        _check_eval_meta(metrics["meta"], run)
        loaded.append({
            "label": run["label"],
            "run_dir": str(run["run_dir"]),
            "eps": float(run["eps"]),
            "metrics": metrics,
        })

    print(f"Loaded {len(loaded)} eps runs.")
    for run in loaded:
        print(f"  {run['label']:8s} eps={run['eps']:.1e}")
    for msg in skipped:
        print(f"  skipped: {msg}")
    return loaded


# %% Figures
def plot_convergence_fraction(runs: list[dict], tol: float | None = None,
                              metric: str = "fk_inf"):
    """Figure 1: convergence fraction over iterations.

    ``tol`` resolves ``FIG_TOL`` at call time, so editing the config cell and
    re-running only this cell takes effect.
    """
    if not runs:
        raise ValueError("No runs loaded.")

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))
    # The ieee style cycles only 4 colours, so 6 eps values would wrap and the
    # legend could not identify their curves.  A viridis ramp also puts eps in
    # monotone light->dark order.
    ax.set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 0.9, len(runs))))
    tol_value = FIG_TOL if tol is None else tol
    tol_key = _tol_key(tol_value)

    for run in sorted(runs, key=lambda r: float(r["eps"]), reverse=True):
        curves = run["metrics"]["convergence_fraction"][metric]
        y = np.asarray(curves[tol_key], dtype=np.float64)
        x = np.arange(1, y.size + 1, dtype=np.int64)
        # Parenthesised final convergence fraction, so the legend carries both.
        ax.plot(x, y, linewidth=1.2,
                label=f"$\\epsilon = {_pow10(run['eps'])}$ ({float(y[-1]):.3f})")

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(f"Convergence fraction (${_METRIC_MATH[metric]} \\leq {_pow10(tol_value)}$)")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction converged")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    return fig, ax


def plot_relative_gap(runs: list[dict]):
    """Figure 2: final relative objective gap vs the shared IPOPT reference.

    ``delta_cost_rel`` divides by ``max(|f_ipopt|, 1)``.  The clamp is inactive
    for these economic runs (cost ~ -190 at nr=2, N=50).
    """
    if not runs:
        raise ValueError("No runs loaded.")

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))
    floor = 1e-16
    bxp_stats = []
    labels = []

    for run in sorted(runs, key=lambda r: float(r["eps"]), reverse=True):
        summary = run["metrics"]["ipopt_delta_summary"]["delta_cost_rel"]
        p10 = max(float(np.asarray(summary["p10"], dtype=np.float64)[-1]), floor)
        p90 = max(float(np.asarray(summary["p90"], dtype=np.float64)[-1]), floor)
        bxp_stats.append({
            "med": max(float(np.asarray(summary["median"], dtype=np.float64)[-1]), floor),
            "q1": p10,
            "q3": p90,
            "whislo": p10,
            "whishi": p90,
            "fliers": [],
        })
        labels.append(f"${_pow10(run['eps'])}$")

    positions = list(range(1, len(bxp_stats) + 1))
    ax.bxp(bxp_stats, positions=positions, showfliers=False, widths=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$|q - q^{*}| \, / \, |q^{*}|$")
    ax.set_title("Final objective gap vs IPOPT")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, ax


# %% Export
def export_all(runs: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, _ = plot_convergence_fraction(runs)
    path_png = output_dir / "fig1_convergence_fraction.png"
    path_pdf = output_dir / "fig1_convergence_fraction.pdf"
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
    written.append(path_png)
    written.append(path_pdf)

    fig, _ = plot_relative_gap(runs)
    path_png = output_dir / "fig2_relative_gap.png"
    path_pdf = output_dir / "fig2_relative_gap.pdf"
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
    written.append(path_png)
    written.append(path_pdf)

    snapshot = {
        "runs": [
            {"label": run["label"], "run_dir": str(run["run_dir"]), "eps": float(run["eps"])}
            for run in runs
        ],
        "eval_kwargs": EVAL_KWARGS,
        "ipopt_overrides": IPOPT_OVERRIDES,
        "fig_tol": float(FIG_TOL),
    }
    path = output_dir / "study_config.json"
    _write_json(path, snapshot)
    written.append(path)
    return written


# %% Stage 1 -- run evaluations (expensive)
prepared_runs = prepare_runs(RUNS)
if RUN_EVAL:
    paths = run_evaluations(prepared_runs, _resolve(OUTPUT_DIR), overwrite=EVAL_OVERWRITE)
    print(f"Wrote {len(paths)} metric files.")


# %% Stage 2 -- load cached metrics
runs = load_runs(prepared_runs, _resolve(OUTPUT_DIR))
runs


# %% Figure 1 -- convergence fraction over iterations
fig1, ax1 = plot_convergence_fraction(runs)


# %% Figure 2 -- final objective gap against IPOPT
fig2, ax2 = plot_relative_gap(runs)


# %% Export -- figures (set SAVE_FIGURES = True)
if SAVE_FIGURES:
    out = _resolve(OUTPUT_DIR)
    paths = export_all(runs, out)
    print(f"Wrote {len(paths)} files to {out}")
    for p in paths:
        print(f"  {p.name}")
