# Time-decomposed LISCO

Code to reproduce the computational studies: data generation, training, evaluation, figures and tables. Datasets, trained weights and results are not included; they are generated locally into `results/`.

## Setup

Requirements: Linux, an NVIDIA GPU with the CUDA toolkit (`nvcc`) and CMake, `gcc`, the HSL library (MA27) for IPOPT, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                    # Python 3.13 environment from uv.lock
git clone --branch lisco-changes https://github.com/lukaslueken/cusadi-lisco.git src/cusadi
```

Set `HSLLIB_PATH` in `src/experiments/common.py` to your `libcoinhsl` library.

## Reproduce

Run from this folder:

```bash
# 1. Train all learned solvers (45 runs from plans/sweep_solver_cstr_paper.json, ~33 GPU-hours)
uv run python -m src.experiments.train.run_sweep_solver_cstr

# 2. Sample NMPC data and train approximate MPC
uv run python -m src.experiments.train.sample_mpc_data
uv run python -m src.experiments.train.train_approx_mpc_cstr

# 3. Evaluate and plot: set RUN_EVAL = True at the top of each script, then run
uv run python -m src.experiments.visualization.visualize_horizon_scaling_runtime_cstr
uv run python -m src.experiments.visualization.visualize_eps_influence_solver_cstr
uv run python -m src.experiments.visualization.visualize_decomposition_cstr
uv run python -m src.experiments.visualization.visualize_closed_loop_comparison_cstr
```

Figures and tables are written to `results/`. Training is not seeded and timings depend on hardware, so results will not match exactly.
