"""
NLP Handler for Solver Learning (LISCO)

Wraps an NMPC object and provides CasADi functions for learning-based solvers.
All NLP evaluations (KKT, Fischer-Burmeister, merit function) are computed in CasADi,
with batching via ca.Function.map(). PyTorch is only used for tensor conversion.

Standard Form NLP:
    min_w f(w, p)
    s.t. h(w, p) = 0
         g(w, p) <= 0

Variable convention:
- w: Primal decision variables (n_w,)
- p: Parameters (n_p,)
- nu: Lagrange multipliers for equality constraints h (n_h,)
- lam: Lagrange multipliers for inequality constraints g (n_g,), lam >= 0
- z: Stacked primal-dual vector [w, nu, lam]
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class NLPConfig:
    """Configuration for NLPHandler. Defaults match NLPHandler.__init__."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float64
    eps: float = 1e-8
    sigma: float = 1e-2       # dual regularization σ_d (−σ on constraint block)
    rho: float = 0.0          # primal regularization ρ (+ρ on Hessian block)
    regularize: bool = False  # opt-in: use convexified + σ/ρ-regularized z-gradient
    n_threads: int = 16
    distributed_KKT_functions: bool = False


# Functions
batched_dot = torch.func.vmap(torch.dot, chunk_size=None)


def casadi_to_torch(
    ca_func: ca.Function,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    output_squeeze: bool = False,
) -> callable:
    """Wrap a CasADi function as a PyTorch function.

    Handles numpy/torch conversion for seamless PyTorch integration.
    Works with both single and mapped (batched) CasADi functions.

    Args:
        ca_func: CasADi function to wrap. Can be a mapped function.
        device: PyTorch device for output tensors ('cpu' or 'cuda').
        dtype: PyTorch dtype for output tensors.
        output_squeeze: If True, flatten scalar outputs to 1D tensor.

    Returns:
        A callable that accepts PyTorch tensors (batch_size, dim) and
        returns PyTorch tensor outputs with batch dimension first.

    Example:
        >>> f_ca = ca.Function("f", [x, p], [x**2 + p])
        >>> f_mapped = f_ca.map(1024, "thread", 8)
        >>> f_torch = casadi_to_torch(f_mapped, device="cuda")
        >>> result = f_torch(x_batch, p_batch)
    """

    def torch_func(*args: torch.Tensor) -> torch.Tensor:
        # Convert torch tensors to transposed numpy arrays (CasADi expects columns)
        np_args = [arg.detach().cpu().numpy().T for arg in args]

        # Evaluate CasADi function
        result_np = np.array(ca_func(*np_args))

        # Handle output shape based on dimensions
        if result_np.ndim == 2:
            # Vector output: (n, batch_size) -> (batch_size, n)
            result_np = result_np.T
        elif result_np.ndim == 3:
            # Matrix output: (m, n, batch_size) -> (batch_size, m, n)
            result_np = np.transpose(result_np, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported output shape: {result_np.shape}")

        # Handle scalar outputs
        if output_squeeze and result_np.shape[-1] == 1:
            result_np = result_np.squeeze(-1)

        return torch.tensor(result_np, dtype=dtype, device=device)

    return torch_func


# NLP Handler Class
class NLPHandler:
    """NLP Handler for solver learning.

    Wraps an NMPC object and provides CasADi functions for learning-based
    optimization solvers. All NLP evaluations are done in CasADi with
    batch support via ca.Function.map().

    Attributes:
        n_w: Number of primal decision variables.
        n_h: Number of equality constraints.
        n_g: Number of inequality constraints.
        n_z: Total primal-dual dimension (n_w + n_g + n_h).
        n_p: Number of parameters.
        device: PyTorch device ('cpu' or 'cuda').
        dtype: PyTorch dtype (torch.float32 or torch.float64).
        eps: Smoothing parameter for Fischer-Burmeister.
    """

    def __init__(
        self,
        nmpc,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-8,
        sigma: float = 1e-2,
        rho: float = 0.0,
        regularize: bool = False,
        n_threads: int = 16,
        distributed_KKT_functions: bool = False,
    ) -> None:
        """Initialize NLPHandler from an NMPC object.

        Args:
            nmpc: NMPC instance providing nlp dict and bounds.
            device: PyTorch device ('cpu' or 'cuda').
            dtype: PyTorch dtype for tensors.
            eps: Smoothing parameter for Fischer-Burmeister.
            sigma: Regularization parameter for dual variables in constraints.
            n_threads: Number of threads for parallel CasADi evaluation.
            distributed_KKT_functions: If True, use stage-wise distributed KKT-family
                functions (requires ``nmpc.distributed_nlp``).
        """
        self.nmpc = nmpc
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.sigma = sigma
        self.rho = rho
        self.regularize = bool(regularize)
        self.batch_size = None
        self.n_threads = n_threads
        self.distributed_KKT_functions = bool(distributed_KKT_functions)
        self.distributed_nlp = getattr(self.nmpc, "distributed_nlp", None)

        # Extract standard NLP dict
        self.nlp = nmpc.nlp

        # Extract dimensions
        self._extract_dimensions()

        # Build CasADi symbolic expressions and functions
        self._build_casadi_functions()
        if self.distributed_KKT_functions:
            if self.distributed_nlp is None:
                raise ValueError(
                    "distributed_KKT_functions=True requires nmpc.distributed_nlp."
                )
            self._build_distributed_casadi_functions()

    # -------------------------------------------------------------------------
    # Dimension Extraction
    # -------------------------------------------------------------------------

    def _extract_dimensions(self) -> None:
        """Extract dimensions from nlp dict."""
        nlp = self.nlp

        self.n_w = nlp["n_w"]
        self.n_h = nlp["n_h"]
        self.n_g = nlp["n_g"]
        self.n_p = nlp["n_p"]
        self.n_z = nlp["n_z"]

        self.N = self.nmpc.N
        self.nx = self.nmpc.nx
        self.nu = self.nmpc.nu

        self.n_wx = (self.N + 1) * self.nx
        self.n_wu = self.N * self.nu

        # Per-stage dimensions for decomposition
        self.n_w_stage = self.nx + self.nu  # [u_k, x_{k+1}] per stage
        self.n_h_stage = self.nx  # dynamics constraint per stage
        # Note: n_g_stage computed in _build_stage_indices()

        # Build stage indices for decomposition
        self._build_stage_indices()

        # Optional distributed-stage metadata from NMPC model
        if self.distributed_nlp is not None:
            dn = self.distributed_nlp
            self.n_w_stage = int(dn.get("n_w_stage", self.n_w_stage))
            self.n_h_stage = int(dn.get("n_h_stage", self.n_h_stage))
            self.n_g_stage = int(dn.get("n_g_stage", self.n_g_stage))
            self.n_ub_per_stage = int(dn.get("n_g_upper_stage", self.n_ub_per_stage))
            self.n_lb_per_stage = int(dn.get("n_g_lower_stage", self.n_lb_per_stage))
            self.n_g_upper = self.N * self.n_ub_per_stage
            self.n_g_lower = self.N * self.n_lb_per_stage
            self.n_z_stage = self.n_w_stage + self.n_h_stage + self.n_g_stage
            self.n_p_stage_start = int(dn.get("n_p_stage_start", self.nx))
            self.n_p_stage_i = int(dn.get("n_p_stage_i", self.nx))
            self.n_p_stage_end = int(dn.get("n_p_stage_end", self.nx))

            if self.n_w != self.N * self.n_w_stage:
                raise ValueError(
                    f"Inconsistent n_w ({self.n_w}) vs N*n_w_stage ({self.N * self.n_w_stage})."
                )
            if self.n_h != self.N * self.n_h_stage:
                raise ValueError(
                    f"Inconsistent n_h ({self.n_h}) vs N*n_h_stage ({self.N * self.n_h_stage})."
                )
            if self.n_g != self.N * self.n_g_stage:
                raise ValueError(
                    f"Inconsistent n_g ({self.n_g}) vs N*n_g_stage ({self.N * self.n_g_stage})."
                )

    def _build_stage_indices(self) -> None:
        """Build index mappings for decomposing z into per-stage z_k vectors.
        
        Assumes uniform bound structure per stage (same bounds on u_k and x_{k+1} for all k).
        This allows decomposition via simple reshapes.
        
        g is structured as [all_upper_bounds, all_lower_bounds] where each half
        is ordered by stage: [bounds_stage_0, bounds_stage_1, ...].
        """
        N, nx, nu, n_w = self.N, self.nx, self.nu, self.n_w
        
        # Count finite bounds per stage (using stage 0 as reference)
        x_lb, x_ub = self.nmpc.x_lb, self.nmpc.x_ub
        u_lb, u_ub = self.nmpc.u_lb, self.nmpc.u_ub
        
        self.n_ub_per_stage = int(np.isfinite(u_ub).sum() + np.isfinite(x_ub).sum())
        self.n_lb_per_stage = int(np.isfinite(u_lb).sum() + np.isfinite(x_lb).sum())
        self.n_g_upper = N * self.n_ub_per_stage
        self.n_g_lower = N * self.n_lb_per_stage
        self.n_g_stage = self.n_ub_per_stage + self.n_lb_per_stage
        self.n_z_stage = self.n_w_stage + self.n_h_stage + self.n_g_stage

    # -------------------------------------------------------------------------
    # CasADi Function Building
    # -------------------------------------------------------------------------

    def _build_casadi_functions(self) -> None:
        """Build CasADi symbolic expressions and functions."""
        nlp = self.nlp

        # Extract symbolic expressions from NMPC
        w_sym = nlp["w"]
        p_sym = nlp["p"]
        f_sym = nlp["f"]
        h_sym = nlp["h"]
        g_sym = nlp["g"]

        # Primal-dual symbolic variables
        lam_sym = ca.SX.sym("lam", self.n_g)
        nu_sym = ca.SX.sym("nu", self.n_h)
        z_sym = ca.vertcat(w_sym, nu_sym, lam_sym)        

        # =====================================================================
        # Lagrangian, Gradient, and KKT Conditions
            
        # Lagrangian: L = f + nu^T h + lam^T g
        L_sym = f_sym + ca.dot(nu_sym, h_sym) + ca.dot(lam_sym, g_sym)
        
        # Gradient of Lagrangian w.r.t. w
        grad_L_sym = ca.gradient(L_sym, w_sym)

        # KKT Conditions        
        g_pos = ca.fmax(0, g_sym)
        lam_neg = ca.fmin(0, lam_sym)
        comp_slack = lam_sym * g_sym
        KKT_sym = ca.vertcat(grad_L_sym, h_sym, g_pos, lam_neg, comp_slack)

        # Regularized/penalized Fischer-Burmeister
        # Standard FB: lam - g - sqrt(lam² + g² + eps²)
        fb_standard = lam_sym - g_sym - ca.sqrt(lam_sym**2 + g_sym**2 + self.eps**2)
        # Penalization: max(0, lam) * max(0, -g)
        fb_penalty = ca.fmax(0, lam_sym) * ca.fmax(0, -g_sym)
        # Combined: 80% standard + 20% penalty
        fb_sym = 0.8 * fb_standard + 0.2 * fb_penalty
        Fk_sym = ca.vertcat(grad_L_sym, h_sym, fb_sym)

        # Merit Function Tk = 0.5 * ||Fk||²
        Tk_sym = 0.5 * ca.dot(Fk_sym, Fk_sym)

        grad_Tk_z_sym = ca.gradient(Tk_sym, z_sym)

        # Create CasADi functions
        self.f_func_ca = ca.Function("f_fn", [w_sym, p_sym], [f_sym])
        self.h_func_ca = ca.Function("h_fn", [w_sym, p_sym], [h_sym])
        self.g_func_ca = ca.Function("g_fn", [w_sym, p_sym], [g_sym])
        self.comp_slack_func_ca = ca.Function("comp_slack_fn", [z_sym, p_sym], [comp_slack])
        self.grad_L_func_ca = ca.Function("grad_L_fn", [z_sym, p_sym], [grad_L_sym])
        self.KKT_func_ca = ca.Function("KKT_fn", [z_sym, p_sym], [KKT_sym])
        self.Fk_func_ca = ca.Function("Fk_fn", [z_sym, p_sym], [Fk_sym])
        self.Tk_nodiff_func_ca = ca.Function("Tk_nodiff_fn", [z_sym, p_sym], [Tk_sym])
        self.grad_Tk_z_func_ca = ca.Function("grad_Tk_z_fn", [z_sym, p_sym], [grad_Tk_z_sym])

    # -------------------------------------------------------------------------
    # Distributed CasADi Function Building
    # -------------------------------------------------------------------------

    def _build_distributed_casadi_functions(self) -> None:
        """Build distributed stage-local CasADi functions from nmpc.distributed_nlp."""
        dn = self.distributed_nlp
        required_keys = (
            "w_stage",
            "p_stage_start",
            "p_stage_i",
            "p_stage_end",
            "f_start",
            "f_i",
            "f_end",
            "h_start",
            "h_i",
            "h_end",
            "g_i",
        )
        missing = [k for k in required_keys if k not in dn]
        if missing:
            raise ValueError(f"distributed_nlp missing keys: {missing}")

        if self.n_p_stage_start != self.nx or self.n_p_stage_i != self.nx or self.n_p_stage_end != self.nx:
            raise NotImplementedError(
                "Current distributed handler expects stage parameters p_stage_* to be x_k (size nx)."
            )

        w_stage_base = dn["w_stage"]
        p_stage_start_base = dn["p_stage_start"]
        p_stage_i_base = dn["p_stage_i"]
        p_stage_end_base = dn["p_stage_end"]
        f_start_base = dn["f_start"]
        f_i_base = dn["f_i"]
        f_end_base = dn["f_end"]
        h_start_base = dn["h_start"]
        h_i_base = dn["h_i"]
        h_end_base = dn["h_end"]
        g_i_base = dn["g_i"]

        self.n_kkt_stage = self.n_w_stage + self.n_h_stage + 3 * self.n_g_stage
        self._dist_stage_roles = ("start", "mid", "end")
        self._dist_output_meta: dict[tuple[str, str], dict] = {}
        self._dist_role_p_dim_model = {
            "start": self.n_p_stage_start,
            "mid": self.n_p_stage_i,
            "end": self.n_p_stage_end,
        }
        self._dist_role_has_nu_next = {"start": True, "mid": True, "end": False}

        role_data = {
            "start": (f_start_base, h_start_base, p_stage_start_base),
            "mid": (f_i_base, h_i_base, p_stage_i_base),
            "end": (f_end_base, h_end_base, p_stage_end_base),
        }

        for role in self._dist_stage_roles:
            f_base, h_base, p_base = role_data[role]
            p_dim_model = self._dist_role_p_dim_model[role]
            has_nu_next = self._dist_role_has_nu_next[role]
            p_dim_total = p_dim_model + (self.n_h_stage if has_nu_next else 0)

            z_stage_sym = ca.SX.sym(f"z_stage_{role}", self.n_z_stage)
            p_stage_sym = ca.SX.sym(f"p_stage_{role}", p_dim_total)

            w_stage_sym = z_stage_sym[: self.n_w_stage]
            x_next_sym = w_stage_sym[self.nu : self.nu + self.nx]
            nu_stage_sym = z_stage_sym[self.n_w_stage : self.n_w_stage + self.n_h_stage]
            lam_stage_sym = z_stage_sym[self.n_w_stage + self.n_h_stage :]

            p_model_sym = p_stage_sym[:p_dim_model]
            nu_next_sym = (
                p_stage_sym[p_dim_model : p_dim_model + self.n_h_stage]
                if has_nu_next
                else ca.SX.zeros(0, 1)
            )

            f_stage_sym = ca.substitute(f_base, w_stage_base, w_stage_sym)
            h_stage_sym = ca.substitute(
                h_base,
                ca.vertcat(w_stage_base, p_base),
                ca.vertcat(w_stage_sym, p_model_sym),
            )
            g_stage_sym = ca.substitute(g_i_base, w_stage_base, w_stage_sym)

            L_stage_sym = (
                f_stage_sym
                + ca.dot(nu_stage_sym, h_stage_sym)
                + ca.dot(lam_stage_sym, g_stage_sym)
            )
            if has_nu_next:
                L_stage_sym = L_stage_sym - ca.dot(nu_next_sym, x_next_sym)

            grad_L_stage_sym = ca.gradient(L_stage_sym, w_stage_sym)

            g_pos_sym = ca.fmax(0, g_stage_sym)
            lam_neg_sym = ca.fmin(0, lam_stage_sym)
            comp_slack_stage_sym = lam_stage_sym * g_stage_sym
            KKT_stage_sym = ca.vertcat(
                grad_L_stage_sym, h_stage_sym, g_pos_sym, lam_neg_sym, comp_slack_stage_sym
            )

            fb_standard = (
                lam_stage_sym
                - g_stage_sym
                - ca.sqrt(lam_stage_sym**2 + g_stage_sym**2 + self.eps**2)
            )
            fb_penalty = ca.fmax(0, lam_stage_sym) * ca.fmax(0, -g_stage_sym)
            fb_stage_sym = 0.8 * fb_standard + 0.2 * fb_penalty
            Fk_stage_sym = ca.vertcat(grad_L_stage_sym, h_stage_sym, fb_stage_sym)
            Tk_stage_sym = 0.5 * ca.dot(Fk_stage_sym, Fk_stage_sym)

            y_stage_sym = ca.vertcat(z_stage_sym, p_stage_sym)
            grad_Tk_stage_sym = ca.gradient(Tk_stage_sym, y_stage_sym)

            if self.regularize:
                w_bar_stage   = ca.SX.sym(f"w_bar_stage_{role}", self.n_w_stage)
                nu_bar_stage  = ca.SX.sym(f"nu_bar_stage_{role}", self.n_h_stage)
                lam_bar_stage = ca.SX.sym(f"lam_bar_stage_{role}", self.n_g_stage)

                grad_L_stage_bar = ca.substitute(grad_L_stage_sym, w_stage_sym, w_bar_stage)
                H_bar_stage = ca.substitute(ca.hessian(f_stage_sym, w_stage_sym)[0], w_stage_sym, w_bar_stage)

                r_stat_stage = grad_L_stage_bar + ca.mtimes(
                    H_bar_stage + self.rho * ca.DM.eye(self.n_w_stage), w_stage_sym - w_bar_stage)
                r_eq_stage  = h_stage_sym - self.sigma * (nu_stage_sym - nu_bar_stage)
                g_reg_stage = g_stage_sym - self.sigma * (lam_stage_sym - lam_bar_stage)
                fb_reg_stage = (0.8 * (lam_stage_sym - g_reg_stage
                                       - ca.sqrt(lam_stage_sym**2 + g_reg_stage**2 + self.eps**2))
                                + 0.2 * ca.fmax(0, lam_stage_sym) * ca.fmax(0, -g_reg_stage))
                Fk_reg_stage_sym = ca.vertcat(r_stat_stage, r_eq_stage, fb_reg_stage)
                grad_Tk_reg_stage_sym = ca.gradient(0.5 * ca.dot(Fk_reg_stage_sym, Fk_reg_stage_sym), y_stage_sym)
                setattr(self, f"grad_Tk_reg_stage_{role}_func_ca", ca.Function(
                    f"grad_Tk_reg_stage_{role}_fn",
                    [z_stage_sym, p_stage_sym, w_bar_stage, nu_bar_stage, lam_bar_stage],
                    [grad_Tk_reg_stage_sym]))

            kkt_fn = ca.Function(
                f"KKT_stage_{role}_fn", [z_stage_sym, p_stage_sym], [KKT_stage_sym]
            )
            fk_fn = ca.Function(
                f"Fk_stage_{role}_fn", [z_stage_sym, p_stage_sym], [Fk_stage_sym]
            )
            tk_fn = ca.Function(
                f"Tk_stage_{role}_fn", [z_stage_sym, p_stage_sym], [Tk_stage_sym]
            )
            grad_fn = ca.Function(
                f"grad_Tk_stage_{role}_fn", [z_stage_sym, p_stage_sym], [grad_Tk_stage_sym]
            )

            setattr(self, f"KKT_stage_{role}_func_ca", kkt_fn)
            setattr(self, f"Fk_stage_{role}_func_ca", fk_fn)
            setattr(self, f"Tk_stage_{role}_func_ca", tk_fn)
            setattr(self, f"grad_Tk_stage_{role}_func_ca", grad_fn)

            for fam_name, fn, dense_size in (
                ("KKT", kkt_fn, self.n_kkt_stage),
                ("Fk", fk_fn, self.n_z_stage),
                ("Tk", tk_fn, 1),
                ("grad_Tk", grad_fn, self.n_z_stage + p_dim_total),
            ):
                rows = np.array(fn.sparsity_out(0).get_triplet()[0], dtype=np.int64)
                self._dist_output_meta[(fam_name, role)] = {
                    "dense_size": int(dense_size),
                    "row_idx": torch.tensor(rows, dtype=torch.long, device=self.device),
                }

    def _distributed_stage_counts(self, batch_size: int) -> dict[str, int]:
        if self.N == 1:
            return {"start": 0, "mid": 0, "end": batch_size}
        return {
            "start": batch_size,
            "mid": batch_size * max(self.N - 2, 0),
            "end": batch_size,
        }

    def _normalize_distributed_output(
        self, raw: torch.Tensor, fam_name: str, role: str, n_rows: int
    ) -> torch.Tensor:
        raw_t = raw.to(dtype=self.dtype, device=self.device)
        if raw_t.ndim == 1:
            raw_t = raw_t.unsqueeze(-1)
        elif raw_t.ndim != 2:
            raise ValueError(
                f"Unsupported distributed tensor rank for {fam_name}/{role}: "
                f"shape={tuple(raw_t.shape)}"
            )

        if raw_t.shape[0] != n_rows:
            if raw_t.shape[1] == n_rows:
                raw_t = raw_t.T
            else:
                raise ValueError(
                    f"Distributed output row mismatch for {fam_name}/{role}: "
                    f"got {tuple(raw_t.shape)}, expected first dim {n_rows}"
                )

        meta = self._dist_output_meta[(fam_name, role)]
        dense_size = int(meta["dense_size"])
        row_idx = meta["row_idx"].to(device=raw_t.device)

        if raw_t.shape[1] == dense_size:
            return raw_t
        if raw_t.shape[1] == row_idx.numel():
            dense = torch.zeros(n_rows, dense_size, dtype=raw_t.dtype, device=raw_t.device)
            dense[:, row_idx] = raw_t
            return dense

        raise ValueError(
            f"Unsupported distributed output shape for {fam_name}/{role}: "
            f"got {tuple(raw_t.shape)}, dense_size={dense_size}, nnz={row_idx.numel()}"
        )

    def _distributed_prepare_stage_context(
        self, z: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size = z.shape[0]
        if self.batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size}. "
                "Call setup_for_batch with the desired batch size first."
            )
        if p.shape[1] != self.nx:
            raise ValueError(
                f"Distributed stage context currently expects p dimension nx={self.nx}, "
                f"got p.shape[1]={p.shape[1]}."
            )

        z_stages = self.decompose_z_batch(z)
        x_next = z_stages[:, :, self.nu : self.nu + self.nx]
        nu_dual = z_stages[:, :, self.n_w_stage : self.n_w_stage + self.n_h_stage]

        x_prev = torch.cat([p.unsqueeze(1), x_next[:, :-1, :]], dim=1)
        nu_next = torch.cat(
            [
                nu_dual[:, 1:, :],
                torch.zeros(
                    batch_size, 1, self.n_h_stage, dtype=self.dtype, device=self.device
                ),
            ],
            dim=1,
        )
        return z_stages, x_prev, nu_next, batch_size

    def _distributed_eval_role(
        self, fam_name: str, role: str, z_in: torch.Tensor, p_in: torch.Tensor
    ) -> torch.Tensor:
        if self.regularize and fam_name == "grad_Tk":
            eval_fn = getattr(self, f"_grad_Tk_reg_stage_{role}_eval")
            if eval_fn is None:
                raise RuntimeError(f"Regularized distributed eval not initialized for grad_Tk/{role}")
            w_bar = z_in[:, :self.n_w_stage]
            nu_bar = z_in[:, self.n_w_stage:self.n_w_stage + self.n_h_stage]
            lam_bar = z_in[:, self.n_w_stage + self.n_h_stage:]
            raw = eval_fn(z_in, p_in, w_bar, nu_bar, lam_bar)
        else:
            eval_fn = getattr(self, f"_{fam_name}_stage_{role}_eval")
            if eval_fn is None:
                raise RuntimeError(f"Distributed eval function not initialized for {fam_name}/{role}")
            raw = eval_fn(z_in, p_in)
        return self._normalize_distributed_output(raw, fam_name, role, z_in.shape[0])

    def _distributed_eval_stage_family(
        self, z: torch.Tensor, p: torch.Tensor, fam_name: str, out_dim: int
    ) -> torch.Tensor:
        z_stages, x_prev, nu_next, batch_size = self._distributed_prepare_stage_context(z, p)
        out_stages = torch.zeros(batch_size, self.N, out_dim, dtype=self.dtype, device=self.device)

        if self.N == 1:
            z_end = z_stages[:, 0, :]
            p_end = x_prev[:, 0, :]
            out_stages[:, 0, :] = self._distributed_eval_role(fam_name, "end", z_end, p_end)
            return out_stages

        z_start = z_stages[:, 0, :]
        p_start = torch.cat([x_prev[:, 0, :], nu_next[:, 0, :]], dim=-1)
        out_stages[:, 0, :] = self._distributed_eval_role(fam_name, "start", z_start, p_start)

        if self.N > 2:
            z_mid = z_stages[:, 1:-1, :].reshape(batch_size * (self.N - 2), self.n_z_stage)
            p_mid = torch.cat([x_prev[:, 1:-1, :], nu_next[:, 1:-1, :]], dim=-1).reshape(
                batch_size * (self.N - 2), self.n_p_stage_i + self.n_h_stage
            )
            mid_out = self._distributed_eval_role(fam_name, "mid", z_mid, p_mid)
            out_stages[:, 1:-1, :] = mid_out.reshape(batch_size, self.N - 2, out_dim)

        z_end = z_stages[:, -1, :]
        p_end = x_prev[:, -1, :]
        out_stages[:, -1, :] = self._distributed_eval_role(fam_name, "end", z_end, p_end)
        return out_stages

    def _compose_stage_g_block_batch(self, block_stages: torch.Tensor) -> torch.Tensor:
        """Compose one stage-wise g-like block to global [upper_all, lower_all] ordering."""
        upper = block_stages[:, :, : self.n_ub_per_stage].reshape(block_stages.shape[0], -1)
        lower = block_stages[:, :, self.n_ub_per_stage :].reshape(block_stages.shape[0], -1)
        return torch.cat([upper, lower], dim=-1)

    def _assemble_distributed_kkt(self, KKT_stage_flat: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Assemble full KKT vector from stage-wise KKT outputs."""
        KKT_stages = KKT_stage_flat.reshape(batch_size, self.N, self.n_kkt_stage)

        i0 = self.n_w_stage
        i1 = i0 + self.n_h_stage
        i2 = i1 + self.n_g_stage
        i3 = i2 + self.n_g_stage

        grad_stages = KKT_stages[:, :, :i0]
        h_stages = KKT_stages[:, :, i0:i1]
        g_pos_stages = KKT_stages[:, :, i1:i2]
        lam_neg_stages = KKT_stages[:, :, i2:i3]
        comp_stages = KKT_stages[:, :, i3:]

        grad_global = grad_stages.reshape(batch_size, -1)
        h_global = h_stages.reshape(batch_size, -1)
        g_pos_global = self._compose_stage_g_block_batch(g_pos_stages)
        lam_neg_global = self._compose_stage_g_block_batch(lam_neg_stages)
        comp_global = self._compose_stage_g_block_batch(comp_stages)

        return torch.cat(
            [grad_global, h_global, g_pos_global, lam_neg_global, comp_global], dim=-1
        )

    def _assemble_distributed_grad_Tk(
        self, z: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Assemble global grad_Tk from stage-local gradients."""
        z_stages, x_prev, nu_next, batch_size = self._distributed_prepare_stage_context(z, p)
        grad_z_stages = torch.zeros(batch_size, self.N, self.n_z_stage, dtype=self.dtype, device=self.device)

        x_slice = slice(self.nu, self.nu + self.nx)
        nu_slice = slice(self.n_w_stage, self.n_w_stage + self.n_h_stage)

        if self.N == 1:
            grad_end = self._distributed_eval_role(
                "grad_Tk", "end", z_stages[:, 0, :], x_prev[:, 0, :]
            )
            grad_z_stages[:, 0, :] += grad_end[:, : self.n_z_stage]
            return self.compose_z_batch(grad_z_stages)

        grad_start = self._distributed_eval_role(
            "grad_Tk",
            "start",
            z_stages[:, 0, :],
            torch.cat([x_prev[:, 0, :], nu_next[:, 0, :]], dim=-1),
        )
        grad_z_stages[:, 0, :] += grad_start[:, : self.n_z_stage]
        grad_nu_next_start = grad_start[:, self.n_z_stage + self.n_p_stage_start :]
        grad_z_stages[:, 1, nu_slice] += grad_nu_next_start

        if self.N > 2:
            z_mid = z_stages[:, 1:-1, :].reshape(batch_size * (self.N - 2), self.n_z_stage)
            p_mid = torch.cat([x_prev[:, 1:-1, :], nu_next[:, 1:-1, :]], dim=-1).reshape(
                batch_size * (self.N - 2), self.n_p_stage_i + self.n_h_stage
            )
            grad_mid = self._distributed_eval_role("grad_Tk", "mid", z_mid, p_mid).reshape(
                batch_size, self.N - 2, self.n_z_stage + self.n_p_stage_i + self.n_h_stage
            )
            grad_z_stages[:, 1:-1, :] += grad_mid[:, :, : self.n_z_stage]

            grad_x_prev_mid = grad_mid[:, :, self.n_z_stage : self.n_z_stage + self.n_p_stage_i]
            grad_nu_next_mid = grad_mid[:, :, self.n_z_stage + self.n_p_stage_i :]
            grad_z_stages[:, 0 : self.N - 2, x_slice] += grad_x_prev_mid
            grad_z_stages[:, 2:self.N, nu_slice] += grad_nu_next_mid

        grad_end = self._distributed_eval_role(
            "grad_Tk", "end", z_stages[:, -1, :], x_prev[:, -1, :]
        )
        grad_z_stages[:, -1, :] += grad_end[:, : self.n_z_stage]
        grad_x_prev_end = grad_end[:, self.n_z_stage : self.n_z_stage + self.n_p_stage_end]
        grad_z_stages[:, self.N - 2, x_slice] += grad_x_prev_end

        return self.compose_z_batch(grad_z_stages)

    def _bind_distributed_interfaces(self) -> None:
        """Bind distributed batch/single-sample callable interfaces."""

        def Fk_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            Fk_stages = self._distributed_eval_stage_family(
                z, p, fam_name="Fk", out_dim=self.n_z_stage
            )
            return self.compose_Fk_batch(Fk_stages)

        def KKT_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            KKT_stages = self._distributed_eval_stage_family(
                z, p, fam_name="KKT", out_dim=self.n_kkt_stage
            )
            batch_size = z.shape[0]
            KKT_stage_flat = KKT_stages.reshape(batch_size * self.N, self.n_kkt_stage)
            return self._assemble_distributed_kkt(KKT_stage_flat, batch_size)

        def Tk_nodiff_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            Tk_stages = self._distributed_eval_stage_family(
                z, p, fam_name="Tk", out_dim=1
            )
            return Tk_stages[:, :, 0].sum(dim=1)

        def grad_Tk_z_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return self._assemble_distributed_grad_Tk(z, p)

        def Tk_batch_func(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            Tk_nodiff_batch = Tk_nodiff_batch_func(z, p)
            grad_Tk_batch = grad_Tk_z_batch_func(z, p)
            return Tk_nodiff_batch + batched_dot(grad_Tk_batch, z - z.detach())

        self.Fk_batch_func = Fk_batch_func
        self.KKT_batch_func = KKT_batch_func
        self.Tk_batch_nodiff_func = Tk_nodiff_batch_func
        self.grad_Tk_z_batch_func = grad_Tk_z_batch_func
        self.Tk_batch_func = Tk_batch_func

    def _setup_distributed_batch_functions(self, batch_size: int, n_threads: int = 1) -> None:
        """Setup distributed stage-mapped CasADi functions for batch evaluation."""
        self.batch_size = batch_size
        self.n_threads = n_threads

        stage_counts = self._distributed_stage_counts(batch_size)
        for role, n_instances in stage_counts.items():
            for fam in ("KKT", "Fk", "Tk", "grad_Tk"):
                setattr(self, f"_{fam}_stage_{role}_eval", None)
                setattr(self, f"_{fam}_stage_{role}_batch_ca", None)
            if self.regularize:
                setattr(self, f"_grad_Tk_reg_stage_{role}_eval", None)
            if n_instances <= 0:
                continue

            kkt_map = getattr(self, f"KKT_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )
            fk_map = getattr(self, f"Fk_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )
            tk_map = getattr(self, f"Tk_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )
            grad_map = getattr(self, f"grad_Tk_stage_{role}_func_ca").map(
                n_instances, "thread", n_threads
            )

            setattr(self, f"_KKT_stage_{role}_batch_ca", kkt_map)
            setattr(self, f"_Fk_stage_{role}_batch_ca", fk_map)
            setattr(self, f"_Tk_stage_{role}_batch_ca", tk_map)
            setattr(self, f"_grad_Tk_stage_{role}_batch_ca", grad_map)

            setattr(
                self,
                f"_KKT_stage_{role}_eval",
                casadi_to_torch(kkt_map, device=self.device, dtype=self.dtype),
            )
            setattr(
                self,
                f"_Fk_stage_{role}_eval",
                casadi_to_torch(fk_map, device=self.device, dtype=self.dtype),
            )
            setattr(
                self,
                f"_Tk_stage_{role}_eval",
                casadi_to_torch(tk_map, device=self.device, dtype=self.dtype),
            )
            setattr(
                self,
                f"_grad_Tk_stage_{role}_eval",
                casadi_to_torch(grad_map, device=self.device, dtype=self.dtype),
            )

            if self.regularize:
                grad_reg_map = getattr(self, f"grad_Tk_reg_stage_{role}_func_ca").map(
                    n_instances, "thread", n_threads
                )
                setattr(
                    self,
                    f"_grad_Tk_reg_stage_{role}_eval",
                    casadi_to_torch(grad_reg_map, device=self.device, dtype=self.dtype),
                )

        self._bind_distributed_interfaces()

    def _setup_distributed_cusadi_functions(
        self, batch_size: int = 1024, force_rebuild: bool = False
    ) -> None:
        """Override distributed stage evaluators with GPU CusADi callables."""
        import sys

        if sys.platform != "linux":
            return
        if not str(self.device).startswith("cuda"):
            return

        print("Setting up distributed Cusadi functions for batch evaluation...")
        from src.apply_cusadi import apply_cusadi

        stage_counts = self._distributed_stage_counts(batch_size)
        for role, n_instances in stage_counts.items():
            if n_instances <= 0:
                continue

            setattr(
                self,
                f"_KKT_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"KKT_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"KKT_stage_{role}_batch",
                ),
            )
            setattr(
                self,
                f"_Fk_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"Fk_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"Fk_stage_{role}_batch",
                ),
            )
            setattr(
                self,
                f"_Tk_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"Tk_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"Tk_stage_{role}_batch",
                ),
            )
            setattr(
                self,
                f"_grad_Tk_stage_{role}_eval",
                apply_cusadi(
                    getattr(self, f"grad_Tk_stage_{role}_func_ca"),
                    batch_size=n_instances,
                    force_rebuild=force_rebuild,
                    name=f"grad_Tk_stage_{role}_batch",
                ),
            )
            if self.regularize:
                setattr(
                    self,
                    f"_grad_Tk_reg_stage_{role}_eval",
                    apply_cusadi(
                        getattr(self, f"grad_Tk_reg_stage_{role}_func_ca"),
                        batch_size=n_instances,
                        force_rebuild=force_rebuild,
                        name=f"grad_Tk_reg_stage_{role}_batch",
                    ),
                )
        self.g_batch_func = apply_cusadi(
            self.g_func_ca,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
            name="g_batch",
        )

    # -------------------------------------------------------------------------
    # Batch Function Setup
    # -------------------------------------------------------------------------

    def _setup_batch_functions(self, batch_size: int, n_threads: int = 1) -> None:
        """Setup batch-mapped CasADi functions for parallel evaluation."""
        self.batch_size = batch_size
        self.n_threads = n_threads

        self._KKT_batch_ca = self.KKT_func_ca.map(batch_size, "thread", n_threads)
        self._Fk_batch_ca = self.Fk_func_ca.map(batch_size, "thread", n_threads)
        self._Tk_batch_nodiff_ca = self.Tk_nodiff_func_ca.map(batch_size, "thread", n_threads)
        self._grad_Tk_z_batch_ca = self.grad_Tk_z_func_ca.map(batch_size, "thread", n_threads)

        self.f_batch_ca = self.f_func_ca.map(batch_size, "thread", n_threads)
        self.h_batch_func_ca = self.h_func_ca.map(batch_size, "thread", n_threads)
        self.g_batch_func_ca = self.g_func_ca.map(batch_size, "thread", n_threads)
        self.comp_slack_batch_func_ca = self.comp_slack_func_ca.map(batch_size, "thread", n_threads)
        self.grad_L_batch_func_ca = self.grad_L_func_ca.map(batch_size, "thread", n_threads)

    # -------------------------------------------------------------------------
    # PyTorch Interface: Batch Functions
    # -------------------------------------------------------------------------

    def _setup_torch_functions(self) -> None:
        self.f_func = casadi_to_torch(self.f_func_ca, device=self.device, dtype=self.dtype)
        self.f_batch_func = casadi_to_torch(self.f_batch_ca, device=self.device, dtype=self.dtype)
        self.h_func = casadi_to_torch(self.h_func_ca, device=self.device, dtype=self.dtype)
        self.g_func = casadi_to_torch(self.g_func_ca, device=self.device, dtype=self.dtype)
        self.grad_L_func = casadi_to_torch(self.grad_L_func_ca, device=self.device, dtype=self.dtype)
        self.KKT_func = casadi_to_torch(self.KKT_func_ca, device=self.device, dtype=self.dtype)
        self.Fk_func = casadi_to_torch(self.Fk_func_ca, device=self.device, dtype=self.dtype)
        self.Tk_nodiff_func = casadi_to_torch(self.Tk_nodiff_func_ca, device=self.device, dtype=self.dtype)
        self.grad_Tk_z_func = casadi_to_torch(self.grad_Tk_z_func_ca, device=self.device, dtype=self.dtype)
        self.KKT_batch_func = casadi_to_torch(self._KKT_batch_ca, device=self.device, dtype=self.dtype)
        self.Fk_batch_func = casadi_to_torch(self._Fk_batch_ca, device=self.device, dtype=self.dtype)
        
        self.grad_L_batch_func = casadi_to_torch(self.grad_L_batch_func_ca, device=self.device, dtype=self.dtype)
        self.h_batch_func = casadi_to_torch(self.h_batch_func_ca, device=self.device, dtype=self.dtype)
        self.g_batch_func = casadi_to_torch(self.g_batch_func_ca, device=self.device, dtype=self.dtype)
        self.comp_slack_batch_func = casadi_to_torch(self.comp_slack_batch_func_ca, device=self.device, dtype=self.dtype)

        self.Tk_batch_nodiff_func = casadi_to_torch(self._Tk_batch_nodiff_ca, device=self.device, dtype=self.dtype)
        self.grad_Tk_z_batch_func = casadi_to_torch(self._grad_Tk_z_batch_ca, device=self.device, dtype=self.dtype)

    def setup_for_batch(self, batch_size: int, force_rebuild: bool = False) -> None:
        """Single entry point for (re-)configuring batch evaluation functions.

        Sets up CPU-mapped CasADi batch functions and their PyTorch wrappers,
        then overrides the hot-path functions with GPU CusADi callables on Linux.
        On Windows the CusADi step is a no-op.

        Args:
            batch_size: Number of parallel instances for batch evaluation.
            force_rebuild: If True, force CUDA codegen even if cached artifacts exist.
        """
        if self.distributed_KKT_functions:
            self._setup_batch_functions(batch_size=batch_size, n_threads=self.n_threads)
            self._setup_torch_functions()
            self._setup_distributed_batch_functions(
                batch_size=batch_size, n_threads=self.n_threads
            )
            self._setup_distributed_cusadi_functions(
                batch_size=batch_size, force_rebuild=force_rebuild
            )
            return
        raise NotImplementedError(
            "Only distributed_KKT_functions=True is used by the paper runs."
        )

    # -------------------------------------------------------------------------
    # Primal-Dual Extraction and Stacking
    # -------------------------------------------------------------------------

    def extract_primal_dual_batch(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract primal and dual variables from batched z."""
        w = z[:, : self.n_w]
        nu = z[:, self.n_w : self.n_w + self.n_h]
        lam = z[:, self.n_w + self.n_h :]
        return w, nu, lam

    def stack_primal_dual_batch(
        self, w: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Stack primal and dual variables into batched z."""
        return torch.hstack([w, nu, lam])

    # -------------------------------------------------------------------------
    # Time-Stage Decomposition
    # -------------------------------------------------------------------------

    def decompose_z_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Decompose batched primal-dual vectors z into per-stage vectors.
        
        Args:
            z: Batched primal-dual vectors of shape (B, n_z)
            
        Returns:
            z_stages: Tensor of shape (B, N, n_z_stage)
        """
        B = z.shape[0]
        w, nu, lam = self.extract_primal_dual_batch(z)
        N, nx, nu_dim = self.N, self.nx, self.nu
        
        w_stages = w.view(B, N, nu_dim + nx)
        nu_stages = nu.view(B, N, nx)
        lam_upper, lam_lower = lam.split([self.n_g_upper, self.n_g_lower], dim=-1)
        lam_stages = torch.cat([lam_upper.view(B, N, -1), lam_lower.view(B, N, -1)], dim=-1)
        
        return torch.cat([w_stages, nu_stages, lam_stages], dim=-1)

    def compose_z_batch(self, z_stages: torch.Tensor) -> torch.Tensor:
        """Compose batched per-stage vectors z_k back into full primal-dual vectors.
        
        Args:
            z_stages: Tensor of shape (B, N, n_z_stage)
            
        Returns:
            z: Batched primal-dual vectors of shape (B, n_z)
        """
        B = z_stages.shape[0]
        N, nx, nu_dim = self.N, self.nx, self.nu
        n_w_s, n_h_s = self.n_w_stage, self.n_h_stage
        
        w = z_stages[:, :, :n_w_s].reshape(B, -1)
        nu = z_stages[:, :, n_w_s:n_w_s + n_h_s].reshape(B, -1)
        lam_stages = z_stages[:, :, n_w_s + n_h_s:]
        lam_upper = lam_stages[:, :, :self.n_ub_per_stage].reshape(B, -1)
        lam_lower = lam_stages[:, :, self.n_ub_per_stage:].reshape(B, -1)
        lam = torch.cat([lam_upper, lam_lower], dim=-1)
        
        return self.stack_primal_dual_batch(w, nu, lam)

    # Aliases for Fk decomposition (same structure as z)
    decompose_Fk_batch = decompose_z_batch
    compose_Fk_batch = compose_z_batch

    def _build_decompose_index(self) -> torch.LongTensor:
        """Build a (N, n_z_stage) index tensor mapping stage positions to flat z positions.

        Usage::

            z_stages = z_flat[idx]  # equivalent to decompose_z_batch(z_flat[None])[0]

        The mapping is a permutation: every flat position appears exactly once.
        """
        N = self.N
        n_w_s = self.n_w_stage
        n_h_s = self.n_h_stage
        n_ub = self.n_ub_per_stage
        n_lb = self.n_lb_per_stage

        idx = torch.empty(N, self.n_z_stage, dtype=torch.long)
        for k in range(N):
            col = 0
            # w block
            idx[k, col:col + n_w_s] = torch.arange(k * n_w_s, (k + 1) * n_w_s)
            col += n_w_s
            # nu block
            base_nu = self.n_w
            idx[k, col:col + n_h_s] = torch.arange(base_nu + k * n_h_s, base_nu + (k + 1) * n_h_s)
            col += n_h_s
            # lam_upper block
            base_lam = self.n_w + self.n_h
            idx[k, col:col + n_ub] = torch.arange(base_lam + k * n_ub, base_lam + (k + 1) * n_ub)
            col += n_ub
            # lam_lower block
            base_lam_lower = self.n_w + self.n_h + self.n_g_upper
            idx[k, col:col + n_lb] = torch.arange(
                base_lam_lower + k * n_lb, base_lam_lower + (k + 1) * n_lb
            )

        assert idx.reshape(-1).unique().numel() == self.n_z, (
            f"Decompose index is not a valid permutation: expected {self.n_z} unique entries, "
            f"got {idx.reshape(-1).unique().numel()}"
        )
        return idx

    def _build_compose_index(self) -> torch.LongTensor:
        """Build a (N * n_z_stage,) index tensor for scatter-based compose.

        Usage::

            dz_flat = torch.zeros(n_z)
            dz_flat.scatter_(0, compose_idx, dz_stages.reshape(-1))
        """
        return self._build_decompose_index().reshape(-1)

    # -------------------------------------------------------------------------
    # Batch Generation for Training
    # -------------------------------------------------------------------------

    def batch_gen_x0(self, n_batch: int, offset: float = 0.0) -> torch.Tensor:
        """Generate random initial states in [0, 1] (scaled).
        
        Args:
            n_batch: Number of samples to generate.
            offset: Fraction to extend sampling range beyond [0, 1].
                    With offset=0.1, samples from [-0.1, 1.1].
        """
        lb = -offset
        ub = 1.0 + offset
        x0 = lb + (ub - lb) * torch.rand(n_batch, self.nx, device=self.device, dtype=self.dtype)
        return x0

    def batch_gen_p(self, n_batch: int, offset: float = 0.0) -> torch.Tensor:
        """Generate random parameter samples (p = x0 for NMPC)."""
        return self.batch_gen_x0(n_batch, offset=offset)

    def batch_gen_w(self, n_batch: int, offset: float = 0.0) -> torch.Tensor:
        """Generate random primal variable samples in [0, 1] (scaled).
        
        Structure: w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        Note: x_0 is a parameter, not a decision variable.
        
        Args:
            n_batch: Number of samples to generate.
            offset: Fraction to extend sampling range beyond [0, 1].
        """
        lb = -offset
        ub = 1.0 + offset

        # U: N control inputs, X: N states (x_1, ..., x_N)
        U = lb + (ub - lb) * torch.rand(n_batch, self.N, self.nu, device=self.device, dtype=self.dtype)
        X = lb + (ub - lb) * torch.rand(n_batch, self.N, self.nx, device=self.device, dtype=self.dtype)

        # Stack into w = [u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N]
        w_list = [U[:, 0, :]]  # u_0
        for k in range(1, self.N):
            w_list.append(X[:, k - 1, :])  # x_k
            w_list.append(U[:, k, :])      # u_k
        w_list.append(X[:, self.N - 1, :])  # x_N
        w = torch.cat(w_list, dim=1)
        return w

    def batch_gen_z(self, n_batch: int, offset: float = 0.0) -> torch.Tensor:
        """Generate random primal-dual variable samples."""
        w = self.batch_gen_w(n_batch, offset=-0.5)
        nu = torch.zeros(n_batch, self.n_h, device=self.device, dtype=self.dtype)
        lam = torch.zeros(n_batch, self.n_g, device=self.device, dtype=self.dtype)
        return self.stack_primal_dual_batch(w, nu, lam)
