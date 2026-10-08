"""Decomposition (n_nodes x n_slide_steps) iteration heatmaps for CSTR (interactive).

Run cell-by-cell (``# %%``).

Runs: the plan entries tagged ``overlap`` in the local sweep plan
(``PLAN_PATH``): nr=5, the selected (n_nodes, n_slide_steps) pairs, N in
{49, 50, 51}.

Stage 1 (``RUN_EVAL``): sample one shared ``p_eval`` from the first run's NLP,
    run ``solver.solve_batch`` on every run in ``RUNS``, and cache the per-run
    first-hit statistics in ``OUTPUT_DIR/aggregated_eval.json``.  No ``eval_*``
    script is invoked and no per-run artifact is touched.
Stage 2 (cheap, re-run freely): load that aggregate and render the simplified
    p90 heatmap indexed by (n_slide_steps, n_nodes)
    (paper: decomposition_heatmap_p90_simple.pdf, fig:decomposition_heatmap).

Censoring.  Samples that never reach the tolerance are censored to ``max_iter``,
so a statistic is a real measurement only where enough samples converged: the
q-th quantile needs ``conv_rate > q``.  In the p90 heatmap, cells where fewer
than 90% of the samples converged are shown in grey.

Caveats.  ``N`` varies across the sweep (the tiling constrains it), so cells
differ slightly in horizon -- ``load_aggregate`` prints the spread.  Memory
scales as (max_iter+1, n_samples, n_z), about 4 GB at the settings below,
because ``solve_batch`` returns the whole trajectory.
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
plt.ion()  # plt.show() returns immediately; a full run does not wait on windows

from src.experiments.common import (
    DEFAULT_PLAN_PATH,
    iterations_to_tolerance,
    load_plan_runs,
    project_root,
)
from src.experiments.eval.eval_solver_cstr import _load_solver_from_run_dir


# %% Configuration
PLAN_PATH = DEFAULT_PLAN_PATH
# Plan order; the first run (nr=5, N=50, n_nodes=5, n_slide_steps=1) provides the
# NLP that samples the shared p_eval.
RUNS = [run["run_dir"] for run in load_plan_runs("overlap", PLAN_PATH)]

EVAL_KWARGS = {
    "n_samples": 512,
    "eval_seed": 42,
    "max_iter": 200,
    "target_tol": 1e-5,
}

# Statistics stored per run: (row key, label, quantile, statistic).  The q-th
# quantile is a measurement only where conv_rate > q, else it is the cap.
PANELS = [
    ("median_iter", "median", 0.50, "median"),
    ("p90_iter", "p90", 0.90, "90th percentile"),
    ("p99_iter", "p99", 0.99, "99th percentile"),
    ("max_iter", "max", 1.00, "maximum"),
]

OUTPUT_DIR = Path("results/_aggregated/decomposition_cstr_nr5")

RUN_EVAL = False        # stage 1: expensive
EVAL_OVERWRITE = False  # stage 1: rebuild an aggregate that already exists
SAVE_FIGURES = True    # export cell


# %% Helpers -- files and validation
def _resolve_path(value: str | Path) -> Path:
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


def prepare_runs(runs: list[str | Path]) -> list[Path]:
    """Resolve run dirs and reject anything that breaks the shared p_eval.

    Done before stage 1 so a bad sweep costs nothing.  `nr` must be uniform:
    p_eval has width n_p = nx = 5*nr, and a mismatch would otherwise surface as
    a shape error deep inside solve_batch.
    """
    if not runs:
        raise ValueError("RUNS is empty.")
    resolved = [_resolve_path(p) for p in runs]

    nrs, Ns = set(), []
    for run_dir in resolved:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        cfg = _read_json(run_dir / "solver_config.json")
        if cfg.get("solver_class") != "SolverDecomposed":
            raise ValueError(
                f"Run {run_dir} has solver_class={cfg.get('solver_class')}; "
                "expected SolverDecomposed."
            )
        nrs.add(int(cfg["model_params"]["nr"]))
        Ns.append(int(cfg["nmpc_params"]["N"]))

    if len(nrs) > 1:
        raise ValueError(f"RUNS mixes nr={sorted(nrs)}; p_eval would not transfer.")

    spread = {N: Ns.count(N) for N in sorted(set(Ns))}
    print(f"Prepared {len(resolved)} runs: nr={nrs.pop()}, N spread {spread}")
    return resolved


def _check_config(config: dict, resolved_runs: list[Path]) -> None:
    """Reject a cached aggregate built under different settings than requested."""
    hint = "Rebuild with RUN_EVAL=True and EVAL_OVERWRITE=True."
    if config.get("run_paths") != [str(rd) for rd in resolved_runs]:
        raise ValueError(
            f"Cached aggregate covers different run_paths (order or contents). {hint}"
        )
    for key, want in EVAL_KWARGS.items():
        if config.get(key) != want:
            raise ValueError(
                f"Cached aggregate was built with {key}={config.get(key)!r}, but "
                f"this study requests {key}={want!r}. {hint}"
            )


# %% Helpers -- evaluation
def _eval_run_first_hit(run_dir: Path,
                        p_eval: torch.Tensor | None) -> tuple[dict, torch.Tensor]:
    """Solve one run's batch and return (row, p_eval_used).

    With `p_eval=None` a batch is generated from this run's NLP and returned, so
    the caller can hand the same one to every subsequent run.
    """
    solver = _load_solver_from_run_dir(run_dir)
    nlp = solver.nlp
    max_iter = int(EVAL_KWARGS["max_iter"])
    n_samples = int(EVAL_KWARGS["n_samples"])

    if p_eval is None:
        seed = int(EVAL_KWARGS["eval_seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        nlp.setup_for_batch(n_samples, force_rebuild=False)
        p_eval = nlp.batch_gen_p(n_samples).detach()

    p_eval_dev = p_eval.to(device=nlp.device, dtype=nlp.dtype)
    nlp.setup_for_batch(int(p_eval_dev.shape[0]), force_rebuild=False)

    with torch.no_grad():
        _z_final, z_traj, _step_times = solver.solve_batch(
            p_batch=p_eval_dev,
            max_iter=max_iter,
            alpha=1.0,
            update_mode="line_search",
            return_trajectory=True,
        )
        fk_inf_tb = torch.stack(
            [nlp.Fk_batch_func(z_k, p_eval_dev).norm(p=float("inf"), dim=1)
             for z_k in z_traj],
            dim=0,
        )  # (T+1, B)

    tol = float(EVAL_KWARGS["target_tol"])
    first_hit = iterations_to_tolerance(fk_inf_tb.detach().cpu(), [tol])[tol]
    T = int(fk_inf_tb.shape[0])  # iterations_to_tolerance marks non-converged with T
    converged = first_hit < T
    censored = torch.where(
        converged, first_hit, torch.full_like(first_hit, max_iter)
    ).to(torch.float64).numpy()

    cfg = _read_json(run_dir / "solver_config.json")
    sp = cfg["solver_params"]
    row = {
        "run_dir": str(run_dir),
        "nr": int(cfg["model_params"]["nr"]),
        "N": int(cfg["nmpc_params"]["N"]),
        "n_nodes": int(sp["n_nodes"]),
        "n_slide_steps": int(sp["n_slide_steps"]),
        "median_iter": float(np.percentile(censored, 50)),
        "p90_iter": float(np.percentile(censored, 90)),
        "p99_iter": float(np.percentile(censored, 99)),
        "max_iter": float(censored.max()),
        "convergence_rate": float(converged.double().mean().item()),
    }
    return row, p_eval


def build_aggregate(resolved_runs: list[Path], output_dir: Path) -> dict:
    """Evaluate every run against one shared p_eval."""
    rows: list[dict] = []
    p_eval: torch.Tensor | None = None
    for i, run_dir in enumerate(resolved_runs, start=1):
        print(f"[{i}/{len(resolved_runs)}] evaluating {run_dir.name} ...")
        row, p_eval = _eval_run_first_hit(run_dir, p_eval)
        print(
            f"  n_nodes={row['n_nodes']}, n_slide_steps={row['n_slide_steps']}: "
            f"median={row['median_iter']:.0f}, p90={row['p90_iter']:.0f}, "
            f"p99={row['p99_iter']:.0f}, max={row['max_iter']:.0f}, "
            f"conv_rate={row['convergence_rate']:.3f}"
        )
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(p_eval.detach().cpu(), output_dir / "p_eval.pt")  # provenance only
    return {
        "config": {
            "run_paths": [str(rd) for rd in resolved_runs],
            **EVAL_KWARGS,
            "metric": "fk_inf",
            "update_mode": "line_search",
        },
        "rows": rows,
    }


# %% Helpers -- loading
def load_aggregate(resolved_runs: list[Path], output_dir: Path) -> dict:
    """Load the cached aggregate, validating it against the requested settings."""
    cache_path = output_dir / "aggregated_eval.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Aggregate not found at {cache_path}. Build it with RUN_EVAL=True."
        )
    aggregated = _read_json(cache_path)
    _check_config(aggregated["config"], resolved_runs)

    rows = aggregated["rows"]
    required = {key for key, _, _, _ in PANELS} | {"convergence_rate"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Cached aggregate is missing row keys {sorted(missing)}. "
            "Rebuild with RUN_EVAL=True and EVAL_OVERWRITE=True."
        )

    # N postdates the original cache format, hence the .get().
    Ns = sorted({row["N"] for row in rows if "N" in row})
    print(f"Loaded {len(rows)} runs from {cache_path}")
    print(f"  N spread:       {Ns or 'not recorded'}")
    return aggregated


def _grids(rows: list[dict]) -> tuple[list[int], list[int], dict[str, np.ndarray]]:
    """Per-metric (n_slide_steps x n_nodes) grids, NaN where a cell is absent."""
    nodes_vals = sorted({r["n_nodes"] for r in rows})
    slide_vals = sorted({r["n_slide_steps"] for r in rows})
    keys = [key for key, _, _, _ in PANELS] + ["convergence_rate"]
    grids = {k: np.full((len(slide_vals), len(nodes_vals)), np.nan) for k in keys}
    for r in rows:
        i = slide_vals.index(r["n_slide_steps"])
        j = nodes_vals.index(r["n_nodes"])
        if not np.isnan(grids["convergence_rate"][i, j]):
            raise ValueError(
                f"Duplicate (n_nodes={r['n_nodes']}, "
                f"n_slide_steps={r['n_slide_steps']}) entry."
            )
        for key in keys:
            grids[key][i, j] = r[key]
    return nodes_vals, slide_vals, grids


# %% Figures
def _ink(im, value: float) -> str:
    """Text colour with contrast against the cell, whatever the colormap does."""
    r, g, b, _ = im.cmap(im.norm(value))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"


# %% Stage 1 -- build the aggregate (expensive; set RUN_EVAL = True)
resolved_runs = prepare_runs(RUNS)
if RUN_EVAL:
    cache_path = _resolve_path(OUTPUT_DIR) / "aggregated_eval.json"
    if cache_path.exists() and not EVAL_OVERWRITE:
        print(f"aggregate exists: {cache_path} (set EVAL_OVERWRITE=True to rebuild)")
    else:
        _write_json(cache_path, build_aggregate(resolved_runs, _resolve_path(OUTPUT_DIR)))
        print(f"Saved: {cache_path}")


# %% Stage 2 -- load the cached aggregate
aggregated = load_aggregate(resolved_runs, _resolve_path(OUTPUT_DIR))


# %% Simplified p90 heatmap
from matplotlib.colors import LinearSegmentedColormap


def _p90_cells(aggregated: dict):
    """(m vals, s vals, p90 grid, cells where p90 is a measurement)."""
    nodes_vals, slide_vals, grids = _grids(aggregated["rows"])
    grid, conv = grids["p90_iter"], grids["convergence_rate"]
    return nodes_vals, slide_vals, grid, np.isfinite(grid) & (conv >= 0.90)


def plot_p90_simple(aggregated: dict):
    """Simplified p90 heatmap: m across, s down, colour only where p90 is real."""
    nodes_vals, slide_vals, grid, valid = _p90_cells(aggregated)
    base = plt.get_cmap("Blues")
    cmap = LinearSegmentedColormap.from_list(
        "blues_trunc", base(np.linspace(0.15, 0.85, 256)))
    cmap.set_bad("white")  # absent and censored cells paint themselves
    measured = np.ma.masked_where(~valid, grid)

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6}):
        fig, ax = plt.subplots(figsize=(3.15, 2.35), constrained_layout=True)
        im = ax.imshow(measured, origin="upper", aspect="auto", cmap=cmap,
                       vmin=measured.min(), vmax=measured.max())

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if valid[i, j]:
                    ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center",
                            color=_ink(im, grid[i, j]), fontsize=6.5)
                elif not np.isfinite(grid[i, j]):
                    ax.text(j, i, "–", ha="center", va="center",
                            color="0.55", fontsize=7)
                else:  # present, but fewer than 90% of samples converged
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor="0.85"))

        ax.add_patch(plt.Rectangle(  # default configuration m=5, s=1
            (nodes_vals.index(5) - 0.5, slide_vals.index(1) - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.2, zorder=5))
        for x in np.arange(0.5, len(nodes_vals) - 1):  # keep blank cells readable
            ax.axvline(x, color="0.9", linewidth=0.4)
        for y in np.arange(0.5, len(slide_vals) - 1):
            ax.axhline(y, color="0.9", linewidth=0.4)

        ax.set_xticks(range(len(nodes_vals)), nodes_vals)
        ax.set_yticks(range(len(slide_vals)), slide_vals)
        ax.minorticks_off()
        ax.set_xlabel(r"Window length $m$")
        ax.set_ylabel(r"Stride $s$")

        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label("Iterations", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        cb.ax.minorticks_off()

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.85"),
                   plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                 edgecolor="0.7", linewidth=0.5)]
        fig.legend(handles, ["$<$90 % converged", "not evaluated"],
                   loc="outside lower center", ncol=2, frameon=False,
                   handlelength=1.1, handleheight=1.0, columnspacing=1.4,
                   borderpad=0.0)
    return fig, ax


# %% Figure -- simplified p90 heatmap
fig6, ax6 = plot_p90_simple(aggregated)
plt.show()


# %% Export -- simplified p90 heatmap (set SAVE_FIGURES = True)
if SAVE_FIGURES:
    out = _resolve_path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig = plot_p90_simple(aggregated)[0]
    fig.savefig(out / "decomposition_heatmap_p90_simple.pdf", dpi=300)
    fig.savefig(out / "decomposition_heatmap_p90_simple.png", dpi=300)
    plt.close(fig)
    print(f"Wrote decomposition_heatmap_p90_simple.pdf/.png to {out}")
