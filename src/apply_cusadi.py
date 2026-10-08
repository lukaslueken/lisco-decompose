# apply_cusadi.py
"""
Minimal helper to compile + use CusADi from a CasADi function.

Public API:
    apply_cusadi(casadi_function, batch_size, name=None, force_rebuild=False) -> callable

The returned callable accepts torch CUDA tensors as inputs and returns a
single torch tensor; CasADi functions with more than one output are not
supported.

Notes:
- No differentiation/autograd is provided or required.
- This uses CusADi's repo-based workflow: saves .casadi into <cusadi_repo>/src/casadi_functions/
  and runs <cusadi_repo>/run_codegen.py --fn=<name>.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import torch
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import casadi as ca

# Ensure src/ is on sys.path so that the local cusadi package (src/cusadi/) can be imported
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

TorchTensor = Any  # avoid hard import at module import time


def _locate_cusadi_repo_root() -> Path:
    import cusadi  # noqa: F401

    pkg_dir = Path(inspect.getfile(cusadi)).resolve().parent

    run_codegen = pkg_dir / "run_codegen.py"
    fn_dir = pkg_dir / "src" / "casadi_functions"

    if not run_codegen.exists():
        raise FileNotFoundError(
            f"Could not find CusADi run_codegen.py at: {run_codegen}\n"
            "This helper assumes CusADi is installed from a repo checkout (editable install recommended)."
        )
    if not fn_dir.exists():
        raise FileNotFoundError(
            f"Could not find CusADi function directory at: {fn_dir}\n"
            "Expected layout: <cusadi_repo>/src/casadi_functions/."
        )

    return pkg_dir


def _hash_casadi_function_bytes(fn: ca.Function) -> str:
    """
    Make a stable fingerprint of the function by saving it and hashing the bytes.
    This is robust against collisions when function.name() stays the same.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "tmp_fn.casadi"
        fn.save(str(tmp_path))
        data = tmp_path.read_bytes()
    return hashlib.sha1(data).hexdigest()  # stable, good enough for caching


def _clone_function_with_name(fn: ca.Function, new_name: str) -> ca.Function:
    """Create an equivalent CasADi function with a new internal name."""
    inputs = [ca.SX.sym(f"in_{i}", fn.sparsity_in(i)) for i in range(int(fn.n_in()))]
    outputs = fn(*inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return ca.Function(new_name, inputs, list(outputs), fn.name_in(), fn.name_out())


def _codegen_outputs_exist(codegen_dir: Path, fn_name: str) -> bool:
    """
    CusADi produces artifacts under <repo>/codegen.
    We use a conservative heuristic: if any file/folder mentions fn_name, assume built.
    """
    if not codegen_dir.exists():
        return False
    hits = list(codegen_dir.glob(f"**/*{fn_name}*"))
    return len(hits) > 0


def _built_library_exists(build_dir: Path, fn_name: str) -> bool:
    """Return True when expected shared library exists in CusADi build dir."""
    return (build_dir / f"lib{fn_name}.so").exists()


@dataclass
class _CusadiCallable:
    """
    Small callable wrapper around cusadi.CusadiFunction.
    """
    cusadi_fn: Any
    n_out: int

    def __call__(self, *inputs: TorchTensor) -> Union[TorchTensor, Tuple[TorchTensor, ...]]:
        # Evaluate in CusADi – inputs must be CUDA tensors (auto-migrate if needed)
        self.cusadi_fn.evaluate([inp.cuda().double().contiguous() for inp in inputs])

        # CusADi exposes outputs via outputs_sparse in the repo README examples.
        # We'll return dense-ish tensors exactly as CusADi provides them.
        outs = self.cusadi_fn.outputs_sparse

        if self.n_out == 1:
            return outs[0].clone()
        else:
            raise NotImplementedError


def apply_cusadi(
    casadi_function: ca.Function,
    *,
    batch_size: int,
    name: Optional[str] = None,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> Callable[..., Union[TorchTensor, Tuple[TorchTensor, ...]]]:
    """
    Compile and wrap a CasADi function as a batched GPU CusADi callable.

    Parameters
    ----------
    casadi_function:
        CasADi Function to compile.
    batch_size:
        Number of parallel instances (batch dimension) CusADi will run on GPU.
    name:
        Optional human-readable base name. If None, uses casadi_function.name().
    force_rebuild:
        If True, reruns codegen even if cached artifacts exist.
    verbose:
        If True, prints the codegen command and some paths.

    Returns
    -------
    callable:
        A Python callable that you can call like: y = f(x, p, ...)
        Inputs must be torch CUDA tensors (typically dtype=torch.float64).
    """
    # Import lazily so importing this file doesn't require torch/cusadi immediately.
    import cusadi
    from cusadi.src import CusadiFunction

    repo_root = _locate_cusadi_repo_root()
    run_codegen = repo_root / "run_codegen.py"
    fn_dir = repo_root / "src" / "casadi_functions"
    codegen_dir = repo_root / "codegen"
    build_dir = repo_root / "build"

    # Create stable unique name: <base>_<short-hash>_b<batch_size>
    base = (name or casadi_function.name() or "casadi_fn").strip()
    h = _hash_casadi_function_bytes(casadi_function)[:10]
    fn_name = f"{base}_{h}_b{batch_size}"
    casadi_function_named = _clone_function_with_name(casadi_function, fn_name)

    casadi_path = fn_dir / f"{fn_name}.casadi"

    # Save into CusADi function directory
    # (Overwrite is OK; name contains hash so it only changes when fn changes.)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / casadi_path.name
        casadi_function_named.save(str(tmp))
        # Move atomically-ish: copy then replace
        shutil.copy2(tmp, casadi_path)

    # Codegen if needed
    cache_hit = _codegen_outputs_exist(codegen_dir, fn_name)
    if cache_hit and build_dir.exists():
        cache_hit = _built_library_exists(build_dir, fn_name)
    need_build = force_rebuild or (not cache_hit)
    if need_build:
        cmd = [sys.executable, str(run_codegen), f"--fn={fn_name}"]
        if verbose:
            print(f"[apply_cusadi] repo_root: {repo_root}")
            print(f"[apply_cusadi] saving .casadi: {casadi_path}")
            print(f"[apply_cusadi] running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    else:
        if verbose:
            print(f"[apply_cusadi] cache hit, skipping codegen for {fn_name}")

    # Load and wrap
    loaded = ca.Function.load(str(casadi_path))
    cusadi_fn = CusadiFunction(loaded, batch_size)

    return _CusadiCallable(cusadi_fn=cusadi_fn, n_out=int(loaded.n_out()))


if __name__ == "__main__":
    # Minimal import-time smoke test (doesn't compile anything).
    repo = _locate_cusadi_repo_root()
    print(f"CusADi repo root detected at: {repo}")
