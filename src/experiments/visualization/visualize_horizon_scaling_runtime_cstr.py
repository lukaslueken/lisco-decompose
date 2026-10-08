"""Horizon-scaling runtime study across reactor counts (interactive, cell-based).

Run this file cell-by-cell (``# %%``) in an interactive Python session.

Two stages:

Runs: the plan entries tagged ``horizon`` in the local sweep plan
(``PLAN_PATH``): monolithic and decomposed solvers for nr in {2, 3, 5}.

Stage 1 (``RUN_EVAL``): for every run listed in ``RUNS``, measure single-point
    CPU speed (profile ``cpu_all_ca``) and, once per (nr, N), IPOPT.
    Expensive.  Results land in each run dir as
    ``eval_single_speed_metrics.json``.

Stage 2 (cheap, re-run freely): load those per-run JSONs into a tidy DataFrame
    and render.  There is no aggregated cache file -- the per-run JSONs *are*
    the cache, and they are re-validated on every load.

Deliverables:

- Figure 1  ``plot_solve_time(df, nr)`` -- solve time vs N for one reactor
            count, decomposed solver vs IPOPT.  Mono is excluded: it does not
            converge at nr=5, so it has no place in a speed comparison.
            Exported once per nr; the paper uses nr=5
            (figure_1_cpu_nr5_solve_time_vs_N.pdf).
- Paper 1   ``build_training_table(df)`` -- parameters, training time and
            convergence by (nr, N), mono against decomp at matched budget
            (table_paper_1_training.tex, tab:horizon_training).
- Paper 2   ``build_runtime_table(df)`` -- problem size, solve time and
            speedup over IPOPT by (nr, N), decomposed solver only
            (table_paper_2_runtime.tex, tab:horizon_runtime).

DataFrame schema, one row per (nr, arm, N, profile):

    nr | arm | N | profile | run_dir | n_z | n_params | train_time_s
       | solve_min | solve_median | solve_p95 | solve_max
       | iter_min | iter_median | iter_p95 | iter_max
       | speedup_min | speedup_median | speedup_p95 | speedup_max
       | conv_rate

``arm`` is one of mono / decomp / ipopt.  IPOPT rows carry ``profile=None``,
``conv_rate`` = IPOPT's success rate, and their ``iter_*`` columns hold IPOPT's
own Newton iteration counts -- not comparable to learned-solver iterations.

Speedups are per-sample: the solver and IPOPT loops iterate the same
``p_eval`` in the same order, so sample i is the same problem on both sides.
IPOPT runs at the same ``tol`` as the solver and its ``mu_target`` is the
counterpart of the learned solver's ``eps`` smoothing, so both solve the
equivalent relaxation.

Points whose convergence rate is below 100% are drawn with hollow markers and
labelled with the rate: a fast non-converging point is not a speed win.

Validation is fail-fast: a run whose cached metrics were produced under
different settings than ``SINGLE_SPEED_KWARGS`` is rejected rather than
silently mixed in.  Runs that are simply absent are skipped and reported.
"""

# %% Imports
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "no-latex"])

from src.experiments.common import (
    DEFAULT_PLAN_PATH,
    HSLLIB_PATH,
    load_plan_runs,
    project_root,
)
from src.experiments.eval.eval_solver_single_speed_cstr import (
    run_eval_solver_single_speed_cstr,
)
from src.models.cstr_cascade import CSTRParams
from src.models.cstr_nmpc import CSTRNMPC, NMPCParams


# %% Configuration
# Runs: {nr: {arm: {N: run_dir}}}, built from the plan entries tagged
# "horizon".  All runs use dt=1.0, so the horizon length N equals the
# predicted time interval.
PLAN_PATH = DEFAULT_PLAN_PATH
RUNS: dict = {}
for _run in load_plan_runs("horizon", PLAN_PATH):
    _cfg = _run["config"]
    _arm = "mono" if _cfg["architecture"] == "monolithic" else "decomp"
    RUNS.setdefault(int(_cfg["model"]["nr"]), {}).setdefault(_arm, {})[
        int(_cfg["nmpc"]["N"])
    ] = _run["run_dir"]

ARM_MARKERS = {"mono": "o", "decomp": "s", "ipopt": "x"}
ARM_ORDER = ("mono", "decomp", "ipopt")

# Figure 1 (one nr per figure, so color is free): color -> solver.  Figure 1
# plots decomp and ipopt only; mono is listed so the arm set can be widened.
FIG1_COLORS = {"mono": "tab:blue", "decomp": "tab:red", "ipopt": "black"}
FIG1_LABELS = {"mono": "monolithic LISCO", "decomp": "decomposed LISCO",
               "ipopt": "IPOPT w. MA27"}

# The single profile measured in stage 1 and used by the figures and tables.
PROFILE_LABELS = {"cpu_all_ca": "cpu"}
PLOT_PROFILE = "cpu_all_ca"

SINGLE_SPEED_KWARGS = {
    "n_samples": 256,
    "warmup": 16,
    "max_iter": 300,
    "tol_metric": "fk_inf",
    "tol_value": 1e-5,
    "eval_seed": 42,
    "ipopt_overrides": {
        "mu_target": 5e-3,
        "tol": 1e-5,
        "linear_solver": "ma27",
        "hsllib": HSLLIB_PATH,
    },
}

OUTPUT_DIR = Path("results/solver_training/_aggregated/horizon_scaling_nr_sweep")

RUN_EVAL = False         # stage 1: expensive
EVAL_OVERWRITE = False  # stage 1: re-measure runs that already have metrics
SAVE_FIGURES = True    # export cell: write PNGs and tables


# %% Helpers -- loading and validation
def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _check_meta(meta: dict, run_dir: Path) -> None:
    """Reject cached metrics produced under different settings than requested."""
    for key, want in SINGLE_SPEED_KWARGS.items():
        if key in meta and meta[key] != want:
            raise ValueError(
                f"Cached metrics for {run_dir.name} were produced with "
                f"{key}={meta[key]!r}, but this study requests {key}={want!r}. "
                "Re-run stage 1 for that run (RUN_EVAL=True, EVAL_OVERWRITE=True)."
            )


def _stats(values, prefix: str) -> dict[str, float]:
    """{prefix}_min / _median / _p95 / _max over a value list."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("min", "median", "p95", "max")}
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_median": float(np.percentile(arr, 50)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(arr.max()),
    }


_NZ_CACHE: dict[tuple[int, int], int] = {}


def _problem_size(cfg: dict, nr: int, N: int) -> int:
    """n_z = n_w + n_h + n_g for this (nr, N).

    No run artifact records the NLP dimensions, so the NMPC is rebuilt to read
    them off -- about 0.3 s per distinct (nr, N), then memoised for the rest of
    the session so re-running the load cell stays cheap.
    """
    key = (nr, N)
    if key not in _NZ_CACHE:
        nmpc = CSTRNMPC(
            CSTRParams.from_dict(cfg["model_params"]),
            NMPCParams.from_dict(cfg["nmpc_params"]),
        )
        _NZ_CACHE[key] = (
            int(nmpc.solver_nlp["w0"].shape[0])
            + int(nmpc.nlp["n_h"])
            + int(nmpc.nlp["n_g"])
        )
    return _NZ_CACHE[key]


def _n_params(nn: dict) -> int:
    """Learned parameter count of one FeedforwardNN, from its config alone.

    Mirrors ``FeedforwardNN.__init__``: one Linear(n_in, h), then n_hidden_layers-1
    Linear(h, h), then Linear(h, n_out).  Checked against every run's ``weights.pt``
    -- exact match, so the plotting path never has to load torch weights.  For the
    decomposed arm this is the whole learned model: the net is shared across windows.
    """
    h, n_layers = int(nn["n_neurons"]), int(nn["n_hidden_layers"])
    return (
        (int(nn["n_in"]) * h + h)
        + (n_layers - 1) * (h * h + h)
        + (h * int(nn["n_out"]) + int(nn["n_out"]))
    )


def _train_time_s(run_dir: Path) -> float:
    """Total training wall-clock for this run, or NaN if not logged."""
    path = run_dir / "training_log.json"
    if not path.exists():
        return float("nan")
    value = _read_json(path).get("train_time")
    if isinstance(value, list):
        value = value[-1] if value else None
    return float(value) if value is not None else float("nan")


def _ipopt_source(nr: int, N: int) -> str | None:
    """Which arm's run carries the IPOPT measurement for this (nr, N)."""
    arms = RUNS[nr]
    for arm in ("decomp", "mono"):
        if N in (arms.get(arm) or {}):
            return arm
    return None


def iter_runs(runs: dict) -> list[tuple[int, str, int, Path]]:
    """Flatten RUNS into (nr, arm, N, run_dir), sorted."""
    out = []
    for nr in sorted(runs):
        for arm in ("mono", "decomp"):
            for N in sorted((runs[nr].get(arm) or {})):
                out.append((nr, arm, N, _resolve(runs[nr][arm][N])))
    return out


def _paired_speedups(solver_raw: list, ipopt_raw: list | None) -> dict[str, float]:
    """Per-sample IPOPT/solver time ratios, as ``speedup_*`` stats.

    The solver and IPOPT loops iterate the same ``p_eval`` in the same order,
    so sample i is the same problem on both sides and the ratio is a genuine
    per-problem speedup -- not a ratio of separately-taken percentiles.

    With no IPOPT measurement the empty input yields the all-NaN stats, which
    is also how IPOPT's own rows fill their (meaningless) speedup columns.
    """
    if not ipopt_raw or len(ipopt_raw) != len(solver_raw):
        return _stats([], "speedup")
    return _stats(
        np.asarray(ipopt_raw, dtype=float) / np.asarray(solver_raw, dtype=float),
        "speedup",
    )


def _solver_rows(payload: dict, nr: int, arm: str, N: int, run_dir: Path,
                 n_z: int, n_params: int, train_time_s: float) -> list[dict]:
    max_iter = int(payload["meta"]["max_iter"])
    ipopt_raw = payload["profiles"].get("ipopt", {}).get("raw", {}).get("solve_time_ms")
    rows = []
    for prof, pdata in payload["profiles"].items():
        if prof == "ipopt":
            continue
        # Censor non-converged samples to max_iter so the stats cover every
        # sample, not just the converged subset.
        censored = [
            float(v) if v is not None else float(max_iter)
            for v in pdata["raw"]["iter_to_tol"]
        ]
        rows.append({
            "nr": nr, "arm": arm, "N": N, "profile": prof, "run_dir": str(run_dir),
            "n_z": int(n_z), "n_params": int(n_params),
            "train_time_s": float(train_time_s),
            **_stats(pdata["raw"]["solve_time_ms"], "solve"),
            **_stats(censored, "iter"),
            **_paired_speedups(pdata["raw"]["solve_time_ms"], ipopt_raw),
            "conv_rate": float(pdata["tolerance"]["convergence_rate"]),
        })
    return rows


def _ipopt_row(payload: dict, nr: int, N: int, run_dir: Path, n_z: int) -> dict:
    ip = payload["profiles"]["ipopt"]
    return {
        "nr": nr, "arm": "ipopt", "N": N, "profile": None, "run_dir": str(run_dir),
        "n_z": int(n_z), "n_params": float("nan"), "train_time_s": float("nan"),
        **_stats(ip["raw"]["solve_time_ms"], "solve"),
        **_stats(ip["raw"]["iterations"], "iter"),
        **_stats([], "speedup"),
        "conv_rate": float(ip["success_rate"]),
    }


def build_dataframe(runs: dict) -> pd.DataFrame:
    """Load every listed run into a tidy frame, validating as we go."""
    rows: list[dict] = []
    skipped: list[str] = []
    nmpc_seen: dict[tuple[int, int], tuple[str, dict]] = {}

    for nr, arm, N, run_dir in iter_runs(runs):
        tag = f"nr={nr} {arm} N={N}"
        if not run_dir.exists():
            skipped.append(f"{tag}: run dir not found ({run_dir.name})")
            continue
        metrics_path = run_dir / "eval_single_speed_metrics.json"
        if not metrics_path.exists():
            skipped.append(f"{tag}: no eval_single_speed_metrics.json")
            continue

        cfg = _read_json(run_dir / "solver_config.json")
        cfg_nr = int(cfg["model_params"]["nr"])
        cfg_N = int(cfg["nmpc_params"]["N"])
        if cfg_nr != nr or cfg_N != N:
            raise ValueError(
                f"{tag}: run {run_dir.name} is actually nr={cfg_nr}, N={cfg_N}. "
                "Check the path in RUNS."
            )
        prev = nmpc_seen.get((nr, N))
        if prev is not None and prev[1] != cfg["nmpc_params"]:
            raise ValueError(
                f"nmpc_params differ between {prev[0]} and {arm} at nr={nr}, N={N}; "
                "the arms are not solving the same problem."
            )
        nmpc_seen[(nr, N)] = (arm, cfg["nmpc_params"])

        payload = _read_json(metrics_path)
        _check_meta(payload["meta"], run_dir)

        n_z = _problem_size(cfg, nr, N)
        rows.extend(_solver_rows(payload, nr, arm, N, run_dir, n_z,
                                 _n_params(cfg["nn_params"]), _train_time_s(run_dir)))

        if arm == _ipopt_source(nr, N):
            if "ipopt" in payload["profiles"]:
                rows.append(_ipopt_row(payload, nr, N, run_dir, n_z))
            else:
                skipped.append(f"nr={nr} ipopt N={N}: not measured in {run_dir.name}")

    df = pd.DataFrame(rows)
    if len(df):
        print(f"Loaded {len(df)} rows from {df['run_dir'].nunique()} runs.")
        found = df[df["arm"] != "ipopt"].groupby(["nr", "arm"])["N"].apply(
            lambda s: sorted(int(v) for v in s.unique())
        )
        for (nr, arm), Ns in found.items():
            print(f"  nr={nr} {arm:7s} N={Ns}")
    else:
        print("Loaded 0 rows.")
    for msg in skipped:
        print(f"  skipped: {msg}")
    return df


# %% Helpers -- selection and drawing
def _select(df: pd.DataFrame, nr: int | None = None,
            arms: tuple[str, ...] = ("decomp", "ipopt")) -> pd.DataFrame:
    """Rows for the plotted profile, given arms and optionally one nr."""
    sub = df[(df["profile"] == PLOT_PROFILE) | (df["arm"] == "ipopt")]
    sub = sub[sub["arm"].isin(arms)]
    if nr is not None:
        sub = sub[sub["nr"] == nr]
    return sub.sort_values(["nr", "arm", "N"])


def _series(sub: pd.DataFrame):
    """Yield (nr, arm, group) in legend order: nr ascending, then ARM_ORDER."""
    groups = sub.groupby(["nr", "arm"])
    for key in sorted(groups.groups, key=lambda k: (k[0], ARM_ORDER.index(k[1]))):
        yield int(key[0]), str(key[1]), groups.get_group(key).sort_values("N")


def _draw_series(ax, grp: pd.DataFrame, ycol: str, color: str, linestyle: str,
                 marker: str, label: str, linewidth: float = 2.0) -> bool:
    """Line with filled markers where converged, hollow + annotated where not.

    The marker is drawn separately from the line so a hollow marker really is
    hollow -- overlaying an outline on a filled marker just looks filled.
    Returns whether any point fell short of full convergence.
    """
    ax.plot(grp["N"], grp[ycol], color=color, linestyle=linestyle,
            linewidth=linewidth, label=label, zorder=3)
    ok = grp[grp["conv_rate"] >= 1.0]
    bad = grp[grp["conv_rate"] < 1.0]
    if len(ok):
        ax.scatter(ok["N"], ok[ycol], marker=marker, s=16, color=color, zorder=4)
    for row in bad.itertuples():
        y = float(getattr(row, ycol))
        ax.scatter([row.N], [y], marker=marker, s=20, facecolors="white",
                   edgecolors=color, linewidths=1.5, zorder=5)
        ax.annotate(f"{float(row.conv_rate):.0%}", xy=(row.N, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=6, color=color, zorder=6)
    return len(bad) > 0


def _unconverged_proxy(ax) -> None:
    ax.scatter([], [], s=20, marker="o", facecolors="white", edgecolors="gray",
               linewidths=1.0, label="hollow = conv < 100%")


# %% Figures
def plot_solve_time(df: pd.DataFrame, nr: int, show_max: bool = True):
    """Figure 1: solve time vs N for one nr -- decomposed solver vs IPOPT.

    Single nr, so colour encodes the solver: red for LISCO, black for IPOPT.
    Solid = median, dotted = max.

    Mono is excluded by construction: it does not converge at nr=5, so its
    wall-clock is time spent hitting ``max_iter``, not a solve time.  See
    ``build_training_table`` for the mono comparison.
    """
    sub = _select(df, nr=nr, arms=("decomp", "ipopt"))
    if sub.empty:
        raise ValueError(f"No decomp/ipopt rows for nr={nr}.")

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))
    any_bad = False
    for _, arm, grp in _series(sub):
        color = FIG1_COLORS[arm]
        any_bad |= _draw_series(ax, grp, "solve_median", color, "-",
                                ARM_MARKERS[arm], f"{FIG1_LABELS[arm]} (median)",
                                linewidth=1.6)
        if show_max:
            ax.plot(grp["N"], grp["solve_max"], color=color, linestyle=":",
                    linewidth=1.1, label=f"{FIG1_LABELS[arm]} (max)", zorder=2)
    if any_bad:
        _unconverged_proxy(ax)

    ax.set_xlabel(r"Prediction Horizon $N$")
    ax.set_ylabel("Solve Time [ms]")
    ax.set_title(
        f"Solve Time vs Horizon ($n_r = {nr}$, {PROFILE_LABELS[PLOT_PROFILE]})",
        fontsize=8,
    )
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    return fig, ax


# %% Paper tables
# Column specs: (DataFrame column, LaTeX header, decimals -- None = integer).
# Explicit LaTeX headers mean no escaping pass is needed anywhere below.
_PAPER1_ARMS = [("mono", "monolithic"), ("decomp", "decomposed")]
_PAPER1_METRICS = [
    ("params_M", r"$n_\mathrm{par}$~[M]", 2),
    ("train_h", r"$t_\mathrm{train}$~[h]", 2),
    # 3 decimals, not 2: mono at (3, 10) is 0.996094, which would round to 1.00
    # and read as full convergence.
    ("conv_rate", "conv.", 3),
]
# Paper table 2 is grouped: (spanning header, column specs).  An empty spanning
# header leaves its columns ungrouped.  2 decimals on the speedups to match the
# heatmap's cell labels, and because 1 decimal collapses 1.68 and 1.72 to "1.7".
_PAPER2_GROUPS = [
    ("", [("n_z", "$n_z$", None)]),
    (r"$t_\mathrm{LISCO}$~[ms]", [("solve_median", "med", 2),
                                  ("solve_p95", "p95", 2),
                                  ("solve_max", "max", 2)]),
    (r"$t_\mathrm{IPOPT}$~[ms]", [("ipopt_median", "med", 2),
                                  ("ipopt_p95", "p95", 2),
                                  ("ipopt_max", "max", 2)]),
    (r"speedup~[$\times$]", [("speedup_min", "min", 2),
                             ("speedup_median", "med", 2),
                             ("speedup_max", "max", 2)]),
]
_PAPER2_COLS = [col for _, cols in _PAPER2_GROUPS for col in cols]


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paper table 1: parameters, training time and convergence, mono vs decomp.

    Missing (nr, N) combinations stay NaN and render as dashes rather than
    vanishing.
    """
    sub = df[df["profile"] == PLOT_PROFILE].copy()
    sub["params_M"] = sub["n_params"] / 1e6
    sub["train_h"] = sub["train_time_s"] / 3600.0

    metrics = [key for key, _, _ in _PAPER1_METRICS]
    table = sub.pivot_table(index=["nr", "N"], columns="arm", values=metrics)
    # pivot_table nests as (metric, arm); the paper wants arm-major groups, and
    # reindex both fills in arms with no run at all and pins the column order.
    table = table.swaplevel(axis=1).reindex(
        columns=pd.MultiIndex.from_product([[arm for arm, _ in _PAPER1_ARMS], metrics])
    )
    table.columns.names = ["arm", "metric"]
    return table.sort_index()


def build_runtime_table(df: pd.DataFrame, arm: str = "decomp") -> pd.DataFrame:
    """Paper table 2: problem size, solve time and speedup over IPOPT."""
    solver = df[(df["arm"] == arm) & (df["profile"] == PLOT_PROFILE)]
    if solver.empty:
        raise ValueError(f"No rows for arm '{arm}'.")
    ipopt = df[df["arm"] == "ipopt"][
        ["nr", "N", "solve_median", "solve_p95", "solve_max"]
    ].rename(columns={"solve_median": "ipopt_median", "solve_p95": "ipopt_p95",
                      "solve_max": "ipopt_max"})
    merged = solver.merge(ipopt, on=["nr", "N"], how="left")
    cols = [key for key, _, _ in _PAPER2_COLS]
    return merged.set_index(["nr", "N"]).sort_index()[cols]


def _fmt_tex(value, decimals: int | None) -> str:
    """One cell.  NaN becomes a dash -- a blank cell reads as an oversight."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if decimals is None:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}"


def _latex_table(index_headers: list[str], col_headers: list[str],
                 body: list[list[str]],
                 groups: list[tuple[str, int]] | None = None) -> str:
    """booktabs tabular over pre-formatted cells -- no float, caption or label.

    The paper wraps this in its own table/table* and writes the caption there.
    `groups` adds a spanning header row above the column names; a group with an
    empty title spans its columns without a header or a rule.
    """
    n_index = len(index_headers)
    lines = [
        f"\\begin{{tabular}}{{{'c' * n_index + 'r' * len(col_headers)}}}",
        "\\toprule",
    ]
    if groups:
        cells, rules, start = [""] * n_index, [], n_index + 1
        for title, span in groups:
            cells.append(f"\\multicolumn{{{span}}}{{c}}{{{title}}}")
            if title:
                rules.append(f"\\cmidrule(lr){{{start}-{start + span - 1}}}")
            start += span
        lines.append(" & ".join(cells) + " \\\\")
        lines.append("".join(rules))
    lines.append(" & ".join(index_headers + col_headers) + " \\\\")
    lines.append("\\midrule")
    lines.extend(" & ".join(row) + " \\\\" for row in body)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def training_table_to_latex(table: pd.DataFrame) -> str:
    body = [
        [str(int(nr)), str(int(N))] + [
            _fmt_tex(table.loc[(nr, N), (arm, key)], dec)
            for arm, _ in _PAPER1_ARMS
            for key, _, dec in _PAPER1_METRICS
        ]
        for nr, N in table.index
    ]
    return _latex_table(
        ["$n_r$", "$N$"],
        [header for _, header, _ in _PAPER1_METRICS] * len(_PAPER1_ARMS),
        body,
        groups=[(title, len(_PAPER1_METRICS)) for _, title in _PAPER1_ARMS],
    )


def runtime_table_to_latex(table: pd.DataFrame) -> str:
    body = [
        [str(int(nr)), str(int(N))] + [
            _fmt_tex(table.loc[(nr, N), key], dec) for key, _, dec in _PAPER2_COLS
        ]
        for nr, N in table.index
    ]
    return _latex_table(
        ["$n_r$", "$N$"], [header for _, header, _ in _PAPER2_COLS],
        body,
        groups=[(title, len(cols)) for title, cols in _PAPER2_GROUPS],
    )


def export_all(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write the solve-time figure (one per nr) and both paper tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prof = PROFILE_LABELS[PLOT_PROFILE]
    written: list[Path] = []

    for nr in sorted(RUNS):
        if _select(df, nr=nr, arms=("decomp", "ipopt")).empty:
            continue
        fig, _ = plot_solve_time(df, nr=nr)
        path_png = output_dir / f"figure_1_{prof}_nr{nr}_solve_time_vs_N.png"
        path_pdf = output_dir / f"figure_1_{prof}_nr{nr}_solve_time_vs_N.pdf"
        fig.savefig(path_png)
        fig.savefig(path_pdf)
        plt.close(fig)
        written.append(path_png)
        written.append(path_pdf)

    training_table = build_training_table(df)
    runtime_table = build_runtime_table(df)
    for name, text in (("table_paper_1_training.tex",
                        training_table_to_latex(training_table)),
                       ("table_paper_2_runtime.tex",
                        runtime_table_to_latex(runtime_table))):
        p = output_dir / name
        p.write_text(text + "\n")
        written.append(p)

    snapshot = {
        "runs": {str(nr): {a: {str(n): path for n, path in d.items()}
                           for a, d in arms.items()} for nr, arms in RUNS.items()},
        "single_speed_kwargs": SINGLE_SPEED_KWARGS,
        "plot_profile": PLOT_PROFILE,
    }
    p = output_dir / "study_config.json"
    with open(p, "w") as f:
        json.dump(snapshot, f, indent=2)
    written.append(p)
    return written


# %% Stage 1 -- run evaluations (expensive; set RUN_EVAL = True)
if RUN_EVAL:
    for nr, arm, N, run_dir in iter_runs(RUNS):
        if not run_dir.exists():
            print(f"skip (missing): nr={nr} {arm} N={N} -> {run_dir}")
            continue
        include_ipopt = arm == _ipopt_source(nr, N)
        print(f"=== nr={nr} {arm} N={N} (ipopt={include_ipopt}) ===")
        run_eval_solver_single_speed_cstr(
            run_dir=run_dir,
            include_ipopt=include_ipopt,
            overwrite=EVAL_OVERWRITE,
            **SINGLE_SPEED_KWARGS,
        )


# %% Stage 2 -- load metrics into a DataFrame
df = build_dataframe(RUNS)
df


# %% Figure 1 -- solve time vs N for one nr (decomp vs IPOPT)
fig1, ax1 = plot_solve_time(df, nr=5)
plt.show()


# %% Paper table 1 -- training cost and convergence (mono vs decomp)
training_table = build_training_table(df)
print(training_table_to_latex(training_table))
training_table


# %% Paper table 2 -- runtime and speedup over IPOPT (decomp)
runtime_table = build_runtime_table(df)
print(runtime_table_to_latex(runtime_table))
runtime_table


# %% Export -- figures + tables (set SAVE_FIGURES = True)
if SAVE_FIGURES:
    out = _resolve(OUTPUT_DIR)
    paths = export_all(df, out)
    print(f"Wrote {len(paths)} files to {out}")
    for p in paths:
        print(f"  {p.name}")
