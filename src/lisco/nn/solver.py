import math
import copy
import json
import time
from pathlib import Path

import torch
import numpy as np
from src.lisco.training.training_utils import adaptive_clip_grad
from src.lisco.nn.nn_models import FeedforwardNN


class Solver(torch.nn.Module):
    def __init__(self,model,nlp,convexification=False,gamma=0.1,jvp_linesearch: bool = False):
        super().__init__()
        self.model = model #step model
        self.nlp = nlp
        self.gamma = gamma # solver_cfg
        self.convexification = convexification # solver_cfg
        self.jvp_linesearch = bool(jvp_linesearch)
        self.offset = 1e-16
        self.norm_Fk_max = 100.0
        self._single_eval_ready = False
        self._single_model = None
        self._single_nn_device = None
        self._single_kkt_backend = None
        self._single_kkt_device = None
        self._single_nn_compile_mode = "none"
        self._single_fk_eval = None
        self._single_fk_buf = None
        self._single_nn_dtype = torch.float32
        self._single_diverge_factor = 10.0
        self._single_alpha_decay = 0.95
        self._single_alpha_growth = 1.0 / 0.95
        self.initialize_functions()
        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def initialize_functions(self):
        # Select Tk function based on convexification flag
        if self.convexification:
            raise NotImplementedError("Convexified Tk not yet implemented in NLPHandler")

    def _get_solver_params_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "convexification": self.convexification,
            "jvp_linesearch": self.jvp_linesearch,
        }

    def _compute_residual_features(self, zk, p):
        """Compute KKT residual features for monolithic solvers.

        Returns:
            Fk_batch: Raw KKT residual, shape (B, n_z).
            Fk_normalized: Residual normalized by its 2-norm, shape (B, n_z).
            norm_Fk: 2-norm of residual + offset, shape (B, 1).
            norm_Fk_log: Log of norm_Fk, shape (B, 1).
        """
        Fk_batch = self.nlp.Fk_batch_func(zk, p)
        norm_Fk = torch.linalg.vector_norm(Fk_batch, ord=2, dim=1, keepdim=True) + self.offset
        norm_Fk_log = torch.log(norm_Fk)
        Fk_normalized = Fk_batch / norm_Fk
        return Fk_batch, Fk_normalized, norm_Fk, norm_Fk_log
    
    def forward(self,zk,p):
        """Predict step direction for given batch of primal-dual iterates and parameters."""
        with torch.no_grad():
            Fk_batch, Fk_normalized, norm_Fk, norm_Fk_log = self._compute_residual_features(zk, p)

        nn_inputs_batch = torch.hstack((p, Fk_normalized, norm_Fk_log))

        norm_Fk_clamped = torch.clamp(norm_Fk, max=self.norm_Fk_max)
        dzk = self.model(nn_inputs_batch) * norm_Fk_clamped

        dzk = dzk * self.gamma

        return dzk

    def training_step(self, zk_batch, p_batch, optim, use_adaptive_clip_grad=False):
        # Zero Grads
        optim.zero_grad(set_to_none=True)

        # Predict step and compute new iterates
        zk_new = zk_batch
        for i in range(1):
            dzk_batch = self.forward(zk_new, p_batch)
            zk_new = zk_new + dzk_batch

        # Evaluate per-sample loss
        Tk_batch = self.nlp.Tk_batch_func(zk_new, p_batch)

        # Compute loss
        loss = torch.mean(torch.log(Tk_batch + self.offset**2))

        # Backprop
        loss.backward()

        # Gradient clipping
        if use_adaptive_clip_grad:
            adaptive_clip_grad(self.model.parameters(), clip_factor=0.01, eps=1e-3)

        # Step
        optim.step()

        return loss, zk_new, Tk_batch

    def sample_z_p(self,n_samples):
        p_batch = self.nlp.batch_gen_p(n_samples,offset=0.0)
        zk_batch = self.nlp.batch_gen_z(n_samples)
        return zk_batch, p_batch

    def _as_numpy_1d(self, x, expected_size: int, name: str) -> np.ndarray:
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        x_np = np.asarray(x_np, dtype=np.float64).reshape(-1)
        if x_np.shape[0] != int(expected_size):
            raise ValueError(
                f"{name} must have size {expected_size}, got {x_np.shape[0]}"
            )
        return x_np

    def _clone_state(self, z):
        if torch.is_tensor(z):
            return z.detach().clone()
        return np.array(z, dtype=np.float64, copy=True)

    def _state_axpy(self, z, dz, alpha: float):
        if torch.is_tensor(z):
            return z + float(alpha) * dz
        return z + float(alpha) * dz

    def _fk_inf_from_residual(self, fk) -> float:
        if torch.is_tensor(fk):
            return float(torch.linalg.vector_norm(fk, ord=float("inf")).item())
        return float(np.max(np.abs(fk)))

    def _to_output_tensor(self, z):
        if torch.is_tensor(z):
            return z.detach().clone()
        return torch.from_numpy(np.asarray(z, dtype=np.float64)).to(dtype=self.nlp.dtype)

    def _setup_single_eval_cache(self) -> None:
        """Hook for subclasses to prepare additional cached tensors."""
        return

    def _ensure_single_eval_ready(self) -> None:
        if not self._single_eval_ready:
            raise RuntimeError(
                "Single-eval fast path not set up. Call setup_for_single_eval(...) first."
            )

    def setup_for_single_eval(
        self,
        nn_device: str = "cpu",
        cpu_num_threads: int | None = 8,
        fk_compile: bool = True,
        fk_cache_dir=None,
        nn_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Prepare a fast single-sample solve path.

        This path is intentionally minimal:
        - no warm-start/predictor logic
        - line-search only
        - no projection/clipping
        """
        nn_device = str(nn_device)
        kkt_backend = "casadi_single"
        kkt_device = "cpu"
        nn_compile_mode = "none"

        if nn_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"Unsupported nn_dtype {nn_dtype}. Expected float32 or bfloat16."
            )
        if nn_device == "cpu" and cpu_num_threads is not None:
            torch.set_num_threads(int(cpu_num_threads))

        model_fast = copy.deepcopy(self.model)
        model_fast = model_fast.to(device=nn_device, dtype=nn_dtype)
        model_fast.nn_dtype = nn_dtype  # so FeedforwardNN.forward casts inputs to nn_dtype
        model_fast.eval()

        self._single_model = model_fast
        self._single_nn_device = torch.device(nn_device)
        self._single_kkt_backend = kkt_backend
        self._single_kkt_device = torch.device(kkt_device)
        self._single_nn_compile_mode = nn_compile_mode
        self._single_nn_dtype = nn_dtype

        self._single_fk_buf = None
        self._single_fk_eval = None
        if fk_compile:
            from src.lisco.nn.single_eval_kernels import build_compiled_fk
            self._single_fk_buf, self._single_fk_eval = build_compiled_fk(
                self.nlp.Fk_func_ca, self.nlp.n_z, self.nlp.n_p, fk_cache_dir,
            )

        self._setup_single_eval_cache()
        self._single_eval_ready = True

    def _eval_fk_single(self, z, p):
        self._ensure_single_eval_ready()
        if self._single_kkt_backend == "casadi_single":
            if self._single_fk_eval is not None:
                return self._single_fk_eval(z, p)
            return np.asarray(self.nlp.Fk_func_ca(z, p), dtype=np.float64).reshape(-1)
        raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

    @torch.no_grad()
    def _forward_fast_single(self, z, p, fk=None):
        self._ensure_single_eval_ready()
        if fk is None:
            fk = self._eval_fk_single(z, p)

        if self._single_kkt_backend == "casadi_single":
            norm_fk = float(np.linalg.norm(fk, ord=2) + self.offset)
            fk_normalized = fk / norm_fk
            nn_input_np = np.concatenate(
                [p, fk_normalized, np.array([math.log(norm_fk)], dtype=np.float64)]
            )
            nn_input = torch.from_numpy(nn_input_np).to(
                device=self._single_nn_device, dtype=torch.float32
            )
            nn_input = nn_input.unsqueeze(0)
            norm_fk_clamped = min(norm_fk, self.norm_Fk_max)
            dz = self._single_model(nn_input).squeeze(0).float()
            dz = dz * float(norm_fk_clamped)
            dz = dz * float(self.gamma)
            return dz.detach().cpu().numpy().astype(np.float64, copy=False)

        raise ValueError(f"Unsupported kkt backend '{self._single_kkt_backend}'.")

    @torch.no_grad()
    def solve_fast_single(
        self,
        p,
        z0,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 1.0,
        update_mode: str = "line_search",
    ):
        """Fast single-sample solve with best-so-far line search.

        Returns:
            (z_best, n_iter, success, fk_inf_best)
        """
        self._ensure_single_eval_ready()

        if update_mode != "line_search":
            raise ValueError("solve_fast_single only supports update_mode='line_search'.")
        if z0 is None:
            raise ValueError("z0 is mandatory for solve_fast_single.")

        p_state = self._as_numpy_1d(p, self.nlp.n_p, "p")
        z_state = self._as_numpy_1d(z0, self.nlp.n_z, "z0")

        z_best = self._clone_state(z_state)
        fk_state = self._eval_fk_single(z_state, p_state)
        fk_best = fk_state
        fk_inf_best = self._fk_inf_from_residual(fk_best)

        if fk_inf_best < float(tol):
            return self._to_output_tensor(z_best), 0, True, float(fk_inf_best)

        alpha_k = float(alpha)
        success = False
        n_iter = 0

        for i in range(int(max_iter)):
            dz = self._forward_fast_single(z_state, p_state, fk=fk_state)
            z_candidate = self._state_axpy(z_state, dz, alpha_k)

            fk_candidate = self._eval_fk_single(z_candidate, p_state)
            fk_inf_candidate = self._fk_inf_from_residual(fk_candidate)

            if fk_inf_candidate < fk_inf_best:
                z_best = self._clone_state(z_candidate)
                fk_best = fk_candidate
                fk_inf_best = fk_inf_candidate
                alpha_k = min(float(alpha), alpha_k * self._single_alpha_growth)

            if fk_inf_candidate > self._single_diverge_factor * fk_inf_best:
                z_state = self._clone_state(z_best)
                fk_state = fk_best
                alpha_k = alpha_k * self._single_alpha_decay
            else:
                z_state = z_candidate
                fk_state = fk_candidate

            n_iter = i + 1
            if fk_inf_best < float(tol):
                success = True
                break

        return self._to_output_tensor(z_best), n_iter, success, float(fk_inf_best)

    @torch.no_grad()
    def solve_batch(
        self,
        p_batch,
        max_iter=100,
        alpha=1.0,
        update_mode="line_search",
        return_trajectory=False,
        z0_batch=None,
    ):
        if update_mode != "line_search":
            raise ValueError("Update mode not implemented.")
        Tk_batch_func = self.nlp.Tk_batch_func
        step_func = self.forward

        alpha_batch = torch.ones(p_batch.shape[0],device=p_batch.device)*alpha
        # initial batch
        n_batch = p_batch.shape[0]
        step_time_list = []
        if z0_batch is not None:
            zk_batch = z0_batch.to(device=p_batch.device, dtype=p_batch.dtype).detach().clone()
            if zk_batch.shape != (n_batch, self.nlp.n_z):
                raise ValueError(
                    f"z0_batch must have shape ({n_batch}, {self.nlp.n_z}), got {tuple(zk_batch.shape)}"
                )
        else:
            zk_batch = self.nlp.batch_gen_z(n_batch)

        T0_batch = Tk_batch_func(zk_batch,p_batch)
        zk_best = zk_batch.detach().clone()
        Tk_best = T0_batch.detach().clone()

        # Pre-allocate trajectory buffer if needed (GPU-friendly)
        if return_trajectory:
            zk_traj_buffer = torch.empty((max_iter + 1, n_batch, zk_batch.shape[1]),
                                         dtype=zk_batch.dtype, device=zk_batch.device)
            zk_traj_buffer[0] = zk_batch.detach().clone()

        # LOOP
        for i in range(max_iter):
            print(f"Solve iteration {i+1}/{max_iter}")
            step_tic = time.perf_counter()

            dzk_batch = self.forward(zk_batch, p_batch)
            alpha_eff = alpha_batch

            # update
            zk_batch = zk_batch + alpha_eff[:,None]*dzk_batch
            Tk_batch = Tk_batch_func(zk_batch,p_batch)
            # update best iterates
            mask_best = Tk_batch < Tk_best
            zk_best[mask_best] = zk_batch[mask_best].detach().clone()
            Tk_best[mask_best] = Tk_batch[mask_best].detach().clone()
            # diverging
            mask_diverged = Tk_batch > 10*Tk_best
            # reset to best + reduce alpha
            zk_batch[mask_diverged] = zk_best[mask_diverged].detach().clone()
            alpha_batch[mask_diverged] = alpha_batch[mask_diverged]*0.95
            alpha_batch[mask_best] = torch.clip(alpha_batch[mask_best]/0.95,0.0,alpha)

            step_toc = time.perf_counter()
            step_time = step_toc - step_tic
            step_time_list.append(step_time)

            if return_trajectory:
                zk_traj_buffer[i+1] = zk_best.detach().clone()

        # END OF SOLVE
        zk_batch = zk_best.detach().clone()

        if return_trajectory:
            # Convert buffer to list only after loop completes
            zk_traj = [zk_traj_buffer[j] for j in range(max_iter + 1)]
            return zk_batch, zk_traj, step_time_list
        else:
            return zk_batch, step_time_list

    def save(self, save_dir):
        """Save solver config (JSON) and model weights (.pt) to save_dir.

        Writes:
            solver_config.json  — all hyperparameters, human-readable
            weights.pt          — FeedforwardNN state dict

        The full Solver can be reconstructed with Solver.load(save_dir)
        (or the appropriate subclass, e.g. SolverDecomposed.load(save_dir)).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "solver_class":  type(self).__name__,
            "model_params":  self.nlp.nmpc.model_params.to_dict(),
            "nmpc_params":   self.nlp.nmpc.nmpc_params.to_dict(),
            "nlp_params": {
                "device":     self.nlp.device,
                "dtype":      str(self.nlp.dtype).split(".")[-1],
                "eps":        self.nlp.eps,
                "sigma":      self.nlp.sigma,
                "n_threads":  self.nlp.n_threads,
                "distributed_KKT_functions": self.nlp.distributed_KKT_functions,
            },
            "nn_params":    self.model.to_dict(),
            "solver_params": self._get_solver_params_dict(),
            "has_predictor": False,
        }

        with open(save_dir / "solver_config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.model.save_weights(save_dir)

    @classmethod
    def load(cls, load_dir) -> "Solver":
        """Reconstruct a Solver from a directory written by save().

        Rebuilds the full chain: CSTRParams → NMPCParams → CSTRNMPC →
        NLPHandler → FeedforwardNN (with loaded weights) → Solver.

        Call on the correct subclass to get the right type back, e.g.:
            SolverDecomposed.load(path)
        A ValueError is raised if the stored solver_class does not match cls.

        Note: hardcoded to CSTRNMPC; generalise if other NMPC types are added.
        """
        from src.lisco.nlp_handler import NLPHandler
        from src.models.cstr_cascade import CSTRParams
        from src.models.cstr_nmpc import CSTRNMPC, NMPCParams

        load_dir = Path(load_dir)

        with open(load_dir / "solver_config.json") as f:
            config = json.load(f)

        if cls.__name__ != config["solver_class"]:
            raise ValueError(
                f"Class mismatch: called {cls.__name__}.load() but "
                f"saved solver_class is '{config['solver_class']}'"
            )

        model_params = CSTRParams.from_dict(config["model_params"])
        nmpc_params  = NMPCParams.from_dict(config["nmpc_params"])
        nmpc = CSTRNMPC(model_params, nmpc_params)

        nlp_cfg = dict(config["nlp_params"])
        nlp_cfg["dtype"] = getattr(torch, nlp_cfg["dtype"])
        nlp_cfg.pop("batch_size", None)   # compat: old files may contain this key
        nlp = NLPHandler(nmpc, **nlp_cfg)

        solver_params = dict(config.get("solver_params", {}))
        if cls.__name__ == "SolverDecomposed":
            required = ("n_nodes", "n_slide_steps", "overlap_merge_mode")
            missing = [key for key in required if key not in solver_params]
            if missing:
                raise ValueError(
                    "Checkpoint is missing decomposed solver parameters "
                    f"{missing}. This checkpoint was likely saved before "
                    "SolverDecomposed-specific serialization was added."
                )

        nn_cfg = dict(config["nn_params"])
        nn_cfg["nn_dtype"] = getattr(torch, nn_cfg["nn_dtype"])
        model = FeedforwardNN(**nn_cfg)
        model.load_weights(load_dir)
        model.to(nlp.device)

        return cls(model, nlp, **solver_params)


class SolverDecomposed(Solver):
    """Windowed decomposed solver with configurable overlap and merge strategy.

    The shared NN is evaluated in parallel over sliding windows of `n_nodes`
    stages. Consecutive windows advance by `n_slide_steps`.

    NN input per window:
        [p,
         Fk_stages_normalized (covered stages, flattened),
         log(||Fk_stage||) (covered stages, flattened),
         positional encoding per covered stage (flattened)]

    NN output per window:
        [dz_stage_0, ..., dz_stage_{n_nodes-1}] (flattened)

    Overlapping stage predictions are merged using `overlap_merge_mode` and then
    scaled by the clamped per-stage residual norm.
    """

    def __init__(
        self,
        model,
        nlp,
        n_nodes: int,
        n_slide_steps: int,
        overlap_merge_mode: str = "average",
        convexification: bool = False,
        gamma: float = 0.1,
        jvp_linesearch: bool = False,
    ):
        self.n_nodes = int(n_nodes)
        self.n_slide_steps = int(n_slide_steps)
        self.overlap_merge_mode = str(overlap_merge_mode)

        super().__init__(
            model=model,
            nlp=nlp,
            convexification=convexification,
            gamma=gamma,
            jvp_linesearch=jvp_linesearch,
        )

        self._validate_config(self.nlp.N)
        self.n_windows = 1 + (self.nlp.N - self.n_nodes) // self.n_slide_steps
        starts = torch.arange(self.n_windows, device=self.nlp.device, dtype=torch.long) * self.n_slide_steps
        offsets = torch.arange(self.n_nodes, device=self.nlp.device, dtype=torch.long)
        self._window_stage_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        self._window_stage_indices_flat = self._window_stage_indices.reshape(-1)
        self._avg_merge_counts = torch.bincount(
            self._window_stage_indices_flat,
            minlength=self.nlp.N,
        )

    def _validate_config(self, N: int) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {self.n_nodes}")
        if self.n_slide_steps < 1:
            raise ValueError(f"n_slide_steps must be >= 1, got {self.n_slide_steps}")
        if self.n_slide_steps > self.n_nodes:
            raise ValueError(
                f"n_slide_steps must be <= n_nodes to avoid gaps, got "
                f"n_slide_steps={self.n_slide_steps}, n_nodes={self.n_nodes}"
            )
        if N < self.n_nodes:
            raise ValueError(
                f"Invalid decomposition for horizon N={N}: n_nodes={self.n_nodes} exceeds N."
            )
        if (N - self.n_nodes) % self.n_slide_steps != 0:
            raise ValueError(
                f"Invalid decomposition for horizon N={N}: "
                f"(N - n_nodes) % n_slide_steps must be 0, got "
                f"({N} - {self.n_nodes}) % {self.n_slide_steps} = "
                f"{(N - self.n_nodes) % self.n_slide_steps}."
            )

    def _get_solver_params_dict(self) -> dict:
        params = super()._get_solver_params_dict()
        params.update(
            {
                "n_nodes": self.n_nodes,
                "n_slide_steps": self.n_slide_steps,
                "overlap_merge_mode": self.overlap_merge_mode,
            }
        )
        return params

    def _setup_single_eval_cache(self) -> None:
        self._single_window_stage_indices = self._window_stage_indices.to(
            device=self._single_nn_device
        )

        if self.nlp.N == 1:
            pos_all = torch.tensor(
                [[0.0, 1.0]], device=self._single_nn_device, dtype=torch.float32
            )
        else:
            k = torch.arange(self.nlp.N, device=self._single_nn_device, dtype=torch.float32).view(
                self.nlp.N, 1
            )
            pos_forward = k / float(self.nlp.N - 1)
            pos_backward = 1.0 - pos_forward
            pos_all = torch.cat([pos_forward, pos_backward], dim=1)

        pos_windows = pos_all[self._single_window_stage_indices]
        self._single_pos_windows_flat = pos_windows.reshape(self.n_windows, 2 * self.n_nodes)

        n_z_stage = self.nlp.n_z_stage

        # --- Column offsets for in-place nn_inputs writes ---
        n_p = self.nlp.n_p
        fk_cols = self.n_nodes * n_z_stage
        norm_cols = self.n_nodes
        self._single_col_p_end = n_p
        self._single_col_fk_end = n_p + fk_cols
        self._single_col_norm_end = n_p + fk_cols + norm_cols

        # --- Numba fused-glue buffers (CPU casadi_single path) ---
        self._np_decompose_idx = self.nlp._build_decompose_index().cpu().numpy()
        self._np_window_idx = self._window_stage_indices.cpu().numpy()
        self._np_counts = self._avg_merge_counts.cpu().numpy().astype(np.float32)
        self._single_nn_inputs_np = np.zeros(
            (self.n_windows, self.model.n_in), dtype=np.float32
        )
        self._single_nn_inputs_t = torch.from_numpy(self._single_nn_inputs_np)
        self._single_nn_inputs_np[:, self._single_col_norm_end:] = (
            self._single_pos_windows_flat.cpu().numpy()
        )
        self._single_dz_out_np = np.zeros(self.nlp.n_z, dtype=np.float32)

        from src.lisco.nn.single_eval_kernels import pre, post
        self._pre_kernel = pre
        self._post_kernel = post

    def _compute_stage_residual_features(self, zk, p, compute_global_norm=False, clamp_log=None):
        """Compute KKT residual features decomposed into MPC stages.

        Args:
            zk: Current primal-dual variables, shape (B, n_z).
            p: Parameters, shape (B, n_p).
            compute_global_norm: Whether to also compute global (non-decomposed) norm.
            clamp_log: Optional (min, max) tuple to clamp log-norm values.

        Returns:
            Fk_batch: Raw full KKT residual, shape (B, n_z).
            Fk_stages: Decomposed residual, shape (B, N, n_z_stage).
            Fk_stages_normalized: Per-stage normalized residual, shape (B, N, n_z_stage).
            norm_Fk_stages: Per-stage 2-norm + offset, shape (B, N, 1).
            norm_Fk_stages_log: Log of per-stage norm, shape (B, N, 1).
            norm_Fk_global: Global 2-norm + offset, shape (B, 1) or None.
            norm_Fk_global_log: Log of global norm, shape (B, 1) or None.
        """
        Fk_batch = self.nlp.Fk_batch_func(zk, p)
        Fk_stages = self.nlp.decompose_Fk_batch(Fk_batch)

        norm_Fk_stages = torch.linalg.vector_norm(Fk_stages, ord=2, dim=-1, keepdim=True) + self.offset
        norm_Fk_stages_log = torch.log(norm_Fk_stages)
        if clamp_log is not None:
            norm_Fk_stages_log = norm_Fk_stages_log.clamp(*clamp_log)
        Fk_stages_normalized = Fk_stages / norm_Fk_stages

        norm_Fk_global = norm_Fk_global_log = None
        if compute_global_norm:
            norm_Fk_global = torch.linalg.vector_norm(Fk_batch, ord=2, dim=-1, keepdim=True) + self.offset
            norm_Fk_global_log = torch.log(norm_Fk_global)
            if clamp_log is not None:
                norm_Fk_global_log = norm_Fk_global_log.clamp(*clamp_log)

        return Fk_batch, Fk_stages, Fk_stages_normalized, norm_Fk_stages, norm_Fk_stages_log, norm_Fk_global, norm_Fk_global_log

    def _map_window_inputs(self, zk: torch.Tensor, p: torch.Tensor):
        B = zk.shape[0]
        N = self.nlp.N
        n_z_stage = self.nlp.n_z_stage

        (
            _Fk_batch,
            _Fk_stages,
            Fk_stages_normalized,
            norm_Fk_stages,
            norm_Fk_stages_log,
            _norm_Fk_global,
            _norm_Fk_global_log,
        ) = self._compute_stage_residual_features(zk, p, compute_global_norm=False)

        window_stage_indices = self._window_stage_indices
        n_windows = self.n_windows

        Fk_windows = Fk_stages_normalized[:, window_stage_indices, :]
        norm_windows = norm_Fk_stages_log[:, window_stage_indices, :]

        Fk_windows_flat = Fk_windows.reshape(B, n_windows, self.n_nodes * n_z_stage)
        norm_windows_flat = norm_windows.reshape(B, n_windows, self.n_nodes)

        if N == 1:
            pos_all = torch.tensor([[0.0, 1.0]], device=zk.device, dtype=zk.dtype)
        else:
            k = torch.arange(N, device=zk.device, dtype=zk.dtype).view(N, 1)
            pos_forward = k / (N - 1)
            pos_backward = 1 - pos_forward
            pos_all = torch.cat([pos_forward, pos_backward], dim=1)

        pos_windows = pos_all[window_stage_indices]
        pos_windows = pos_windows.unsqueeze(0).expand(B, -1, -1, -1)
        pos_windows_flat = pos_windows.reshape(B, n_windows, 2 * self.n_nodes)

        p_expanded = p.unsqueeze(1).expand(B, n_windows, -1)
        nn_inputs = torch.cat(
            [p_expanded, Fk_windows_flat, norm_windows_flat, pos_windows_flat],
            dim=-1,
        )
        nn_inputs_flat = nn_inputs.reshape(B * n_windows, -1)

        return nn_inputs_flat, norm_Fk_stages

    def _merge_window_outputs_average(self, dz_windows: torch.Tensor) -> torch.Tensor:
        N = self.nlp.N
        n_z_stage = self.nlp.n_z_stage
        B = dz_windows.shape[0]
        dz_flat = dz_windows.reshape(B, -1, n_z_stage)
        dz_sum = torch.zeros(B, N, n_z_stage, device=dz_windows.device, dtype=dz_windows.dtype)

        stage_idx = self._window_stage_indices_flat.to(device=dz_windows.device)
        stage_idx = stage_idx.view(1, -1, 1).expand(B, -1, n_z_stage)
        dz_sum.scatter_add_(dim=1, index=stage_idx, src=dz_flat)

        count = self._avg_merge_counts.to(device=dz_windows.device, dtype=dz_windows.dtype)
        return dz_sum / count.view(1, N, 1)

    def forward(self, zk, p):
        nn_inputs_flat, norm_Fk_stages = self._map_window_inputs(zk, p)
        B = zk.shape[0]
        n_z_stage = self.nlp.n_z_stage

        dz_windows_flat = self.model(nn_inputs_flat)
        dz_windows = dz_windows_flat.reshape(
            B,
            self.n_windows,
            self.n_nodes,
            n_z_stage,
        )

        if self.overlap_merge_mode == "average":
            dz_stages = self._merge_window_outputs_average(dz_windows=dz_windows)
        else:
            raise NotImplementedError

        norm_Fk_clamped = torch.clamp(norm_Fk_stages, max=self.norm_Fk_max)
        dz_stages = dz_stages * norm_Fk_clamped

        dzk = self.nlp.compose_z_batch(dz_stages)
        dzk = dzk * self.gamma

        return dzk

    @torch.no_grad()
    def _forward_fast_single(self, z, p, fk=None):
        self._ensure_single_eval_ready()
        if fk is None:
            fk = self._eval_fk_single(z, p)

        # --- Fused Numba glue path (CPU casadi_single) ---
        nn_np = self._single_nn_inputs_np
        nn_np[:, :self._single_col_p_end] = p.astype(np.float32)
        norm = self._pre_kernel(
            fk, self._np_decompose_idx, self._np_window_idx,
            np.float32(self.offset), nn_np,
            self._single_col_p_end, self._single_col_fk_end,
        )
        dz_windows_np = self._single_model(self._single_nn_inputs_t).float().numpy()
        self._post_kernel(
            dz_windows_np, self._np_decompose_idx, self._np_window_idx,
            self._np_counts, norm, np.float32(self.norm_Fk_max),
            np.float32(self.gamma), self._single_dz_out_np,
        )
        return self._single_dz_out_np
