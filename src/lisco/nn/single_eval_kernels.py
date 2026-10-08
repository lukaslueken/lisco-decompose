"""Speedup kernels for the CPU single-sample solve path.

Two independent mechanisms:
  * build_compiled_fk -- CasADi codegen + gcc -O3 -march=native, evaluated
    through a zero-copy Function.buffer().
  * pre / post -- fused Numba njit glue for the decomposed forward pass,
    replacing many small torch ops with two compiled loops.

Importing this module imports numba.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import casadi as ca
import numpy as np
from numba import njit


def build_compiled_fk(fk_func_ca, n_z, n_p, cache_dir=None):
    """Compile a CasADi Fk function to a .so and return a zero-copy evaluator.

    Returns (buf, eval_fn):
      buf     -- FunctionBuffer holding the fixed input/output memoryviews. MUST
                 be kept alive by the caller; if garbage-collected the eval
                 crashes the process.
      eval_fn -- eval_fn(z_np, p_np) -> np.ndarray, shape (n_z,), a fresh copy
                 (solve_fast_single retains fk_state/fk_best by reference, so the
                 shared output buffer must not be handed out directly).

    The .so is cached under cache_dir (default ~/.cache/lisco_fk), keyed by a
    hash of the generated C source, so repeated setups skip the ~8s gcc call.
    Fail-fast if gcc is unavailable.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "lisco_fk"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    name = fk_func_ca.name()

    # Generate C into cache_dir, hash it to pick the cache slot.
    tmp_c = cache_dir / f"tmp_{name}.c"
    cg = ca.CodeGenerator(tmp_c.name, {"with_header": False})
    cg.add(fk_func_ca)
    cg.generate(str(cache_dir) + "/")
    src = tmp_c.read_bytes()
    key = hashlib.sha1(src).hexdigest()[:16]
    so_path = cache_dir / f"{name}_{key}.so"
    c_path = cache_dir / f"{name}_{key}.c"

    if not so_path.exists():
        c_path.write_bytes(src)
        r = subprocess.run(
            ["gcc", "-O3", "-march=native", "-fPIC", "-shared",
             str(c_path), "-o", str(so_path)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"gcc failed compiling {c_path}:\n{r.stderr}")
    tmp_c.unlink(missing_ok=True)

    fc = ca.external(name, str(so_path))

    z_buf = np.zeros(int(n_z), dtype=np.float64)
    p_buf = np.zeros(int(n_p), dtype=np.float64)
    out_buf = np.zeros(int(n_z), dtype=np.float64)
    buf, evalf = fc.buffer()
    buf.set_arg(0, memoryview(z_buf))
    buf.set_arg(1, memoryview(p_buf))
    buf.set_res(0, memoryview(out_buf))

    def eval_fn(z_np, p_np):
        z_buf[:] = z_np
        p_buf[:] = p_np
        evalf()
        return out_buf.copy()

    # One-time fail-fast self-check vs the interpreted function. Guards against a
    # corrupt/stale cache slot silently returning wrong residuals.
    z0 = np.random.default_rng(0).standard_normal(int(n_z))
    p0 = np.random.default_rng(1).standard_normal(int(n_p))
    ref = np.asarray(fk_func_ca(z0, p0)).reshape(-1)
    if np.max(np.abs(eval_fn(z0, p0) - ref)) > 1e-10:
        raise RuntimeError("Compiled Fk disagrees with interpreted Fk (>1e-10).")

    return buf, eval_fn


@njit(cache=True, fastmath=True)
def pre(fk, decompose_idx, window_idx, offset, nn_inputs, col_p_end, col_fk_end):
    """Decompose Fk into stages, per-stage normalize, assemble window inputs.

    Writes the normalized-Fk block ([col_p_end:col_fk_end]) and the log-norm
    block ([col_fk_end:col_fk_end+n_nodes]) of nn_inputs in place. The p columns
    ([:col_p_end]) and positional columns (after the log-norm block) are assumed
    pre-filled by the caller. Returns per-stage norm (N,), float32.
    """
    N, n_z_stage = decompose_idx.shape
    n_windows, n_nodes = window_idx.shape

    fk_stages = np.empty((N, n_z_stage), dtype=np.float32)
    log_norm = np.empty(N, dtype=np.float32)
    norm = np.empty(N, dtype=np.float32)
    for s in range(N):
        acc = 0.0
        for c in range(n_z_stage):
            v = fk[decompose_idx[s, c]]
            fk_stages[s, c] = np.float32(v)
            acc += v * v
        nrm = np.sqrt(acc) + offset
        norm[s] = np.float32(nrm)
        log_norm[s] = np.float32(np.log(nrm))
        inv = np.float32(1.0 / nrm)
        for c in range(n_z_stage):
            fk_stages[s, c] = fk_stages[s, c] * inv

    for w in range(n_windows):
        for j in range(n_nodes):
            s = window_idx[w, j]
            base = col_p_end + j * n_z_stage
            for c in range(n_z_stage):
                nn_inputs[w, base + c] = fk_stages[s, c]
            nn_inputs[w, col_fk_end + j] = log_norm[s]
    return norm


@njit(cache=True, fastmath=True)
def post(dz_windows, decompose_idx, window_idx, counts, norm, norm_fk_max, gamma, dz_out):
    """Merge overlapping windows, scale by clamped per-stage norm, compose flat.

    dz_out[decompose_idx[s, c]] = (sum_windows dz / counts[s]) * min(norm[s],
    norm_fk_max) * gamma. Writes dz_out (n_z,) in place.
    """
    N, n_z_stage = decompose_idx.shape
    n_windows, n_nodes = window_idx.shape

    dz_sum = np.zeros((N, n_z_stage), dtype=np.float32)
    for w in range(n_windows):
        for j in range(n_nodes):
            s = window_idx[w, j]
            base = j * n_z_stage
            for c in range(n_z_stage):
                dz_sum[s, c] += dz_windows[w, base + c]

    for s in range(N):
        nrm = norm[s]
        if nrm > norm_fk_max:
            nrm = norm_fk_max
        scale = (nrm / counts[s]) * gamma
        for c in range(n_z_stage):
            dz_out[decompose_idx[s, c]] = dz_sum[s, c] * scale
