"""
Economic NMPC for CSTR Cascade Model (Simplified, No Initial Value Embedding)

Standard Form NLP:
    min_w  f(w, p)
    s.t.   h(w, p) = 0          (equality: dynamics only)
           g(w, p) <= 0         (inequality: box bounds)
    
    where:
        w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]  (decision variables)
        p = x0  (parameters: initial state)
        
    The initial state x0 enters the dynamics directly at the first time step:
        h = [x_1 - F(x0, u_0); x_2 - F(x_1, u_1); ...; x_N - F(x_{N-1}, u_{N-1})]
"""

from dataclasses import dataclass, field, fields
import casadi as ca
import numpy as np


def _serialize_scalar(v):
    """Encode ±inf as a string for JSON compatibility."""
    if isinstance(v, float) and np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _deserialize_scalar(v):
    """Decode inf strings back to float."""
    if v == "inf":  return np.inf
    if v == "-inf": return -np.inf
    return v

from src.models.cstr_cascade import CSTRCascade


@dataclass
class NMPCParams:
    """Parameters for the economic NMPC formulation.
    
    Scaling vs Bounds:
        - Scaling ranges (x_scaling/u_scaling): Fixed ranges for min-max normalization.
          All variables are scaled to [0, 1] using these ranges. These should not change.
        - Bounds (x_bounds/u_bounds): Decision variable bounds for the NLP.
          By default equal to scaling ranges, but can be different.
          Use +inf/-inf to disable specific bound constraints.
    """
    N: int = 10
    w_R: float = 10.0
    w_S: float = 1.0
    w_uA: float = 0.1
    w_uB: float = 0.0
    objective: str = "economic"  # only the economic objective is implemented
    setpoint_cR: float = None
    setpoint_cS: float = None
    print_level: int = 5
    max_iter: int = 5000
    ipopt_tol: float = 1e-10
    
    # Scaling ranges: dict with keys matching state/input names, values are (min, max) tuples
    # Used for min-max normalization to [0, 1]
    x_scaling: dict = field(default_factory=lambda: {
        "cA": (0.0, 0.1),
        "cB": (0.0, 1.0),
        "cR": (0.0, 4.0),
        "cS": (0.0, 1.0),
        "Tr": (20.0, 80.0),
    })
    u_scaling: dict = field(default_factory=lambda: {
        "uA": (0.0, 1.5),
        "uB": (0.0, 1.5),
    })
    
    # Decision variable bounds: dict with keys matching state/input names, values are (lb, ub) tuples
    # Use np.inf/-np.inf to disable bounds.
    # Example: {"cS": (0.0, 0.5)} to tighten cS bound, {"cA": (-np.inf, np.inf)} to remove cA bounds
    x_bounds: dict = field(default_factory=lambda: {
        "cA": (0.0, np.inf),
        "cB": (-np.inf, np.inf),
        "cR": (-np.inf, np.inf),
        "cS": (-np.inf, np.inf),
        "Tr": (20.0, 80.0),
    })
    u_bounds: dict = field(default_factory=lambda: {
        "uA": (0.0, 1.5),
        "uB": (0.0, 1.5),
    })

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict.

        Dict fields (x_scaling, u_scaling, x_bounds, u_bounds): tuples → lists, ±inf → strings.
        Scalar fields: passed through unchanged.
        """
        result = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                result[f.name] = {
                    k: [_serialize_scalar(t[0]), _serialize_scalar(t[1])]
                    for k, t in v.items()
                }
            else:
                result[f.name] = v
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "NMPCParams":
        """Reconstruct from a dict produced by to_dict().

        Converts lists back to tuples and restores ±inf from strings.
        """
        kwargs = {}
        for k, v in d.items():
            if isinstance(v, dict):
                kwargs[k] = {
                    kk: (_deserialize_scalar(vv[0]), _deserialize_scalar(vv[1]))
                    for kk, vv in v.items()
                }
            else:
                kwargs[k] = v
        return cls(**kwargs)


class CSTRNMPC:
    """Economic NMPC controller for the CSTR cascade model.
    
    No initial value embedding: x0 enters dynamics directly at first time step.
    """

    def __init__(self, model_params, nmpc_params=None):
        self.model_params = model_params
        self.nmpc_params = nmpc_params if nmpc_params is not None else NMPCParams()

        self.model = CSTRCascade(model_params, integrator="implicit_euler")

        self.nx = model_params.nx
        self.nu = model_params.nu
        self.N = self.nmpc_params.N

        self._build_scaling_ranges()
        self._build_bounds()
        self._build_nlp()
        self._build_distributed_nlp()
        self._ipopt_overrides = None
        self._build_solver()

    def _dict_to_arrays(self, d: dict, size: int, names_and_slices: list) -> tuple:
        """Convert {name: (lo, hi)} dict to (lo_array, hi_array)."""
        arr_lo = np.zeros(size)
        arr_hi = np.zeros(size)
        for name, (start, end) in names_and_slices:
            arr_lo[start:end], arr_hi[start:end] = d[name]
        return arr_lo, arr_hi

    def _build_scaling_ranges(self):
        """Build scaling ranges for min-max normalization."""
        nr = self.model_params.nr

        x_scaling = self.nmpc_params.x_scaling
        u_scaling = self.nmpc_params.u_scaling
        self.scaling_ranges = {**x_scaling, **u_scaling}

        # State scaling: names → array slices
        x_slices = [("cA", (0, nr)), ("cB", (nr, 2*nr)), ("cR", (2*nr, 3*nr)),
                    ("cS", (3*nr, 4*nr)), ("Tr", (4*nr, 5*nr))]
        self.x_min, self.x_max = self._dict_to_arrays(x_scaling, self.nx, x_slices)

        # Input scaling (Tj is a fixed parameter, not a control input)
        u_slices = [("uA", (0, nr)), ("uB", (nr, 2*nr))]
        self.u_min, self.u_max = self._dict_to_arrays(u_scaling, self.nu, u_slices)

        # Compute scaling factors: scale = max - min
        self.x_scale = self.x_max - self.x_min
        self.u_scale = self.u_max - self.u_min
        self.x_scale[self.x_scale == 0.0] = 1.0
        self.u_scale[self.u_scale == 0.0] = 1.0

        # Build w_scale and w_offset for the structure: [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        w_scale_list = [self.u_scale]  # u_0
        w_offset_list = [self.u_min]
        for k in range(1, self.N):
            w_scale_list.append(self.x_scale)  # x_k
            w_scale_list.append(self.u_scale)  # u_k
            w_offset_list.append(self.x_min)
            w_offset_list.append(self.u_min)
        w_scale_list.append(self.x_scale)  # x_N
        w_offset_list.append(self.x_min)

        self.w_scale = np.concatenate(w_scale_list)
        self.w_offset = np.concatenate(w_offset_list)

    def _build_bounds(self):
        """Build decision variable bounds."""
        nr = self.model_params.nr
        
        x_bounds = self.nmpc_params.x_bounds
        u_bounds = self.nmpc_params.u_bounds
        self.bounds = {**x_bounds, **u_bounds}
        
        # State bounds
        x_slices = [("cA", (0, nr)), ("cB", (nr, 2*nr)), ("cR", (2*nr, 3*nr)),
                    ("cS", (3*nr, 4*nr)), ("Tr", (4*nr, 5*nr))]
        self.x_lb, self.x_ub = self._dict_to_arrays(x_bounds, self.nx, x_slices)
        
        # Input bounds (Tj is a fixed parameter, not a control input)
        u_slices = [("uA", (0, nr)), ("uB", (nr, 2*nr))]
        self.u_lb, self.u_ub = self._dict_to_arrays(u_bounds, self.nu, u_slices)

    def _unscale_x(self, x_scaled):
        """Unscale state from [0,1] to physical units: x = x_scaled * x_scale + x_min"""
        return x_scaled * self.x_scale + self.x_min
    
    def _unscale_u(self, u_scaled):
        """Unscale input from [0,1] to physical units: u = u_scaled * u_scale + u_min"""
        return u_scaled * self.u_scale + self.u_min
    
    def _scale_x(self, x):
        """Scale state to [0,1]: x_scaled = (x - x_min) / x_scale"""
        return (x - self.x_min) / self.x_scale
    
    def _scale_u(self, u):
        """Scale input to [0,1]: u_scaled = (u - u_min) / u_scale"""
        return (u - self.u_min) / self.u_scale

    def _x_idx(self, k: int) -> tuple:
        """Index range for x_k in w (k >= 1). Returns (start, end)."""
        # Structure: [u_0, x_1, u_1, x_2, u_2, ..., x_{N-1}, u_{N-1}, x_N]
        # x_k starts after: u_0 + (k-1) * (nx + nu)
        start = self.nu + (k - 1) * (self.nx + self.nu)
        return start, start + self.nx

    def _u_idx(self, k: int) -> tuple:
        """Index range for u_k in w (k >= 0). Returns (start, end)."""
        # u_0 at 0, then u_k starts after: u_0 + k * (nx + nu) - nx = nu + (k-1)*(nx+nu) + nx
        if k == 0:
            return 0, self.nu
        start = self.nu + (k - 1) * (self.nx + self.nu) + self.nx
        return start, start + self.nu

    def _get_last_reactor_outputs(self, x_scaled):
        """Return scaled [0, 1] values of (cR, cS) for the last reactor.

        Args:
            x_scaled: Slice of the scaled state vector for one time step.
        """
        nr = self.model_params.nr
        cR = x_scaled[3*nr - 1]
        cS = x_scaled[4*nr - 1]
        return cR, cS

    def _build_economic_cost(self, w):
        """Build economic cost: maximize c_R, minimize c_S of last reactor."""
        J = ca.SX(0.0)
        w_R = self.nmpc_params.w_R
        w_S = self.nmpc_params.w_S
        N = self.N

        # Stage costs for k=1,...,N
        for k in range(0, N):
            i_start, i_end = self._x_idx(k+1)
            cR, cS = self._get_last_reactor_outputs(w[i_start:i_end])
            J += -w_R * cR + w_S * cS

        # Control effort costs for k=0,...,N-1 (scaled [0, 1] inputs)
        for k in range(0,N):
            u_start, u_end = self._u_idx(k)
            u_k = w[u_start:u_end]
            uA_k = u_k[0:self.model_params.nr]
            uB_k = u_k[self.model_params.nr:]
            J += self.nmpc_params.w_uA * ca.sumsqr(uA_k) + self.nmpc_params.w_uB * ca.sumsqr(uB_k)

        return J*self.model_params.dt

    def _build_nlp(self):
        """Build the NLP in standard form.
        
        Decision variables: w = [u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        Parameters: p = x0 (initial state, scaled to [0,1])
        
        All variables in w are scaled to [0, 1] using min-max normalization.
        """
        N = self.N
        n_x = self.nx
        n_u = self.nu

        w_list = []
        # States and controls for k=0,...,N-1 (controls) and k=1,...,N (states)
        for k in range(0, N):
            u_k = ca.SX.sym(f"u_{k}", n_u)
            w_list.append(u_k)

            x_k = ca.SX.sym(f"x_{k+1}", n_x)
            w_list.append(x_k)
        # Stack decision variables
        w = ca.vertcat(*w_list)
        n_w = w.size()[0]

        # default initial guess
        w0 = []
        x0_default_scaled = self._scale_x(self.model_params.get_default_x0())
        for k in range(0, N):
            # u_k initial guess: mid of scaling range
            u0_k = 0.5 * np.ones(n_u)
            w0.extend(u0_k.tolist())

            # x_{k+1} initial guess: default x0
            w0.extend(x0_default_scaled.tolist())
        w0 = np.array(w0)

        # Parameters: x0 (initial state, scaled to [0,1])
        x0_param = ca.SX.sym("x0", n_x)
        p = x0_param
        n_p = p.size()[0]

        # Objective
        if self.nmpc_params.objective == "economic":
            f = self._build_economic_cost(w)
        else:
            raise ValueError(f"Unknown objective type: {self.nmpc_params.objective}")

        # Equality constraints h(w, p) = 0 (dynamics only)
        h_list = []
        dt = self.model_params.dt

        # First dynamics constraint: x_1 - F(x0_param, u_0) = 0
        u_0_s, u_0_e = self._u_idx(0)
        x_1_s, x_1_e = self._x_idx(1)
        u_0 = w[u_0_s:u_0_e]
        x_1 = w[x_1_s:x_1_e]
        
        f_eval = self.model.f_ode(
            x_1 * self.x_scale + self.x_min,
            u_0 * self.u_scale + self.u_min
        )
        h_list.append(x_1 - x0_param - dt * f_eval/self.x_scale) # scaled dynamics

        # Dynamics constraints for k=1,...,N-1
        for k in range(1, N):
            x_k_s, x_k_e = self._x_idx(k)
            u_k_s, u_k_e = self._u_idx(k)
            x_kp1_s, x_kp1_e = self._x_idx(k + 1)
            
            x_k = w[x_k_s:x_k_e]
            u_k = w[u_k_s:u_k_e]
            x_k_next = w[x_kp1_s:x_kp1_e]

            f_eval = self.model.f_ode(
                x_k_next * self.x_scale + self.x_min,
                u_k * self.u_scale + self.u_min
            )
            h_list.append(x_k_next - x_k - dt * f_eval/self.x_scale)

        h = ca.vertcat(*h_list)
        n_h = h.size()[0]

        # Inequality constraints g(w, p) <= 0 (from box bounds)
        g_list = []
        
        # Build bound vectors matching w structure
        lbw_list = [self.u_lb]  # u_0
        ubw_list = [self.u_ub]
        for k in range(1, N):
            lbw_list.append(self.x_lb)
            ubw_list.append(self.x_ub)
            lbw_list.append(self.u_lb)
            ubw_list.append(self.u_ub)
        lbw_list.append(self.x_lb)  # x_N
        ubw_list.append(self.x_ub)
        
        lbw = np.concatenate(lbw_list)
        ubw = np.concatenate(ubw_list)
        
        # Upper bounds: w_phys - ub <= 0
        for i in range(n_w):
            if np.isfinite(ubw[i]):
                g_list.append(w[i] + self.w_offset[i]/self.w_scale[i] - ubw[i]/self.w_scale[i])

        # Lower bounds: lb - w_phys <= 0
        for i in range(n_w):
            if np.isfinite(lbw[i]):
                g_list.append(lbw[i]/self.w_scale[i]  - (w[i] + self.w_offset[i]/self.w_scale[i]))

        if len(g_list) > 0:
            g = ca.vertcat(*g_list)
            n_g = g.size()[0]
        else:
            g = ca.SX(0, 1)
            n_g = 0

        self.nlp = {
            "w": w,
            "p": p,
            "f": f,
            "h": h,
            "g": g,
            "n_w": int(n_w),
            "n_p": int(n_p),
            "n_h": int(n_h),
            "n_g": int(n_g),
            "n_z": int(n_w + n_h + n_g),
            "w0": w0,
            "n_x": n_x,
            "n_u": n_u,
            "N": N,
        }

    def _build_distributed_nlp(self):
        """Build stage-local NLP components for distributed formulations."""
        n_x = self.nx
        n_u = self.nu
        nr = self.model_params.nr

        w_stage = ca.SX.sym("w_stage", n_u + n_x)  # [u_k, x_{k+1}]
        x_prev = ca.SX.sym("x_prev", n_x)          # x_k (or x0 for first stage)

        u_stage = w_stage[:n_u]
        x_next = w_stage[n_u:]

        if self.nmpc_params.objective == "economic":
            cR, cS = self._get_last_reactor_outputs(x_next)
            stage_cost = -self.nmpc_params.w_R * cR + self.nmpc_params.w_S * cS
        else:
            raise ValueError(f"Unknown objective type: {self.nmpc_params.objective}")

        uA_stage = u_stage[:nr]
        uB_stage = u_stage[nr:]
        control_cost = (
            self.nmpc_params.w_uA * ca.sumsqr(uA_stage)
            + self.nmpc_params.w_uB * ca.sumsqr(uB_stage)
        )
        f_stage = self.model_params.dt*(stage_cost + control_cost)

        f_eval = self.model.f_ode(
            x_next * self.x_scale + self.x_min,
            u_stage * self.u_scale + self.u_min,
        )
        h_stage = x_next - x_prev - self.model_params.dt * f_eval / self.x_scale

        stage_lb = np.concatenate([self.u_lb, self.x_lb])
        stage_ub = np.concatenate([self.u_ub, self.x_ub])
        stage_scale = np.concatenate([self.u_scale, self.x_scale])
        stage_offset = np.concatenate([self.u_min, self.x_min])

        g_upper = []
        g_lower = []
        for i in range(n_u + n_x):
            if np.isfinite(stage_ub[i]):
                g_upper.append(
                    w_stage[i] + stage_offset[i] / stage_scale[i] - stage_ub[i] / stage_scale[i]
                )
        for i in range(n_u + n_x):
            if np.isfinite(stage_lb[i]):
                g_lower.append(
                    stage_lb[i] / stage_scale[i] - (w_stage[i] + stage_offset[i] / stage_scale[i])
                )

        g_upper_stage = ca.vertcat(*g_upper) if len(g_upper) > 0 else ca.SX.zeros(0, 1)
        g_lower_stage = ca.vertcat(*g_lower) if len(g_lower) > 0 else ca.SX.zeros(0, 1)
        g_stage = ca.vertcat(g_upper_stage, g_lower_stage)

        self.distributed_nlp = {
            "w_stage": w_stage,
            "p_stage_start": x_prev,
            "p_stage_i": x_prev,
            "p_stage_end": x_prev,
            "f_start": f_stage,
            "f_i": f_stage,
            "f_end": f_stage,
            "h_start": h_stage,
            "h_i": h_stage,
            "h_end": h_stage,
            "g_i": g_stage,
            "f_start_fn": ca.Function("f_start_fn", [w_stage], [f_stage]),
            "f_i_fn": ca.Function("f_i_fn", [w_stage], [f_stage]),
            "f_end_fn": ca.Function("f_end_fn", [w_stage], [f_stage]),
            "h_start_fn": ca.Function("h_start_fn", [w_stage, x_prev], [h_stage]),
            "h_i_fn": ca.Function("h_i_fn", [w_stage, x_prev], [h_stage]),
            "h_end_fn": ca.Function("h_end_fn", [w_stage, x_prev], [h_stage]),
            "g_i_fn": ca.Function("g_i_fn", [w_stage], [g_stage]),
            "n_w_stage": int(n_u + n_x),
            "n_h_stage": int(n_x),
            "n_g_stage": int(g_stage.size1()),
            "n_p_stage_start": int(n_x),
            "n_p_stage_i": int(n_x),
            "n_p_stage_end": int(n_x),
            "n_g_upper_stage": int(g_upper_stage.size1()),
            "n_g_lower_stage": int(g_lower_stage.size1()),
            "N": int(self.N),
        }

    def _build_solver(self):
        """Build IPOPT solver."""
        n_h = self.nlp["n_h"]
        n_g = self.nlp["n_g"]
        w = self.nlp["w"]
        p = self.nlp["p"]
        f = self.nlp["f"]
        h = self.nlp["h"]
        g = self.nlp["g"]
        w0 = self.nlp["w0"]

        g_solver_list = []
        lbg_solver = []
        ubg_solver = []

        if n_h > 0:
            g_solver_list.append(h)
            lbg_solver.extend([0.0] * n_h)
            ubg_solver.extend([0.0] * n_h)

        if n_g > 0:
            g_solver_list.append(g)
            lbg_solver.extend([-np.inf] * n_g)
            ubg_solver.extend([0.0] * n_g)

        if len(g_solver_list) > 0:
            g_solver = ca.vertcat(*g_solver_list)
        else:
            g_solver = ca.SX(0, 1)

        lbg_solver = np.array(lbg_solver)
        ubg_solver = np.array(ubg_solver)

        solver_nlp = {"x": w, "f": f, "g": g_solver, "p": p}
        opts = {
            "ipopt.print_level": self.nmpc_params.print_level,
            "ipopt.max_iter": self.nmpc_params.max_iter,
            "ipopt.tol": self.nmpc_params.ipopt_tol,
            "print_time": False,
        }
        if self._ipopt_overrides:
            for k, v in self._ipopt_overrides.items():
                print(f"Override {k} with {v}")
                opts[f"ipopt.{k}"] = v
        self.solver = ca.nlpsol("nmpc_solver", "ipopt", solver_nlp, opts)

        solver_nlp.update({"w0": w0, "lbg": lbg_solver, "ubg": ubg_solver})
        self.solver_nlp = solver_nlp

    def override_ipopt_options(self, ipopt_opts: dict) -> None:
        """Rebuild IPOPT solver with additional/overridden options."""
        self._ipopt_overrides = ipopt_opts
        self._build_solver()

    def extract_states_controls(self, w):
        """Extract states X and controls U from decision vector w.
        
        Note: x_0 is not in w; caller should prepend x0 if full trajectory needed.
        Returns:
            X: States for x_1,...,x_N ((N)*nx,)
            U: Controls for u_0,...,u_{N-1} (N*nu,)
        """
        X_list = []
        U_list = []

        # States x_1, ..., x_N
        for k in range(1, self.N + 1):
            i_start, i_end = self._x_idx(k)
            X_list.append(w[i_start:i_end])

        # Controls u_0, ..., u_{N-1}
        for k in range(self.N):
            i_start, i_end = self._u_idx(k)
            U_list.append(w[i_start:i_end])

        X = np.concatenate(X_list)
        U = np.concatenate(U_list)
        return X, U

    def solve(self, x0):
        """Solve the NMPC problem.
        
        Args:
            x0: Current state (nx,), scaled.
            return_primal_dual: If True, return full primal-dual solution vector z_opt.
        
        Returns:
            z_opt: Optimal primal-dual solution vector [w_opt, nu_opt, lam_opt], primal variables scaled
            info: Solver information dict
        """
        import time

        x0 = np.atleast_1d(x0).flatten()
        p = x0

        w0 = self.solver_nlp["w0"]
        lbg = self.solver_nlp["lbg"]
        ubg = self.solver_nlp["ubg"]

        t_start = time.perf_counter()
        sol = self.solver(x0=w0, lbg=lbg, ubg=ubg, p=p)
        t_solve = time.perf_counter() - t_start

        w_opt = np.array(sol["x"]).flatten()
        lam_g_opt = np.array(sol["lam_g"]).flatten()
        obj_val = float(sol["f"])

        stats = self.solver.stats()
        info = {
            "objective": obj_val,
            "iterations": stats.get("iter_count", -1),
            "success": stats.get("success", False),
            "return_status": stats.get("return_status", "unknown"),
            "solve_time": t_solve,
        }

        n_h = self.nlp["n_h"]
        n_g = self.nlp["n_g"]
        nu_opt = np.array(lam_g_opt[:n_h]) if n_h > 0 else np.array([])
        lam_opt = np.array(lam_g_opt[n_h:n_h + n_g]) if n_g > 0 else np.array([])
        z_opt = np.hstack([w_opt, nu_opt, lam_opt])

        return z_opt, info

    def mpc_step(self, x0_unscaled):
        """Execute one MPC step: scale input, solve, extract and unscale control action.
        
        Args:
            x0_unscaled: Current state in physical units (nx,).
        
        Returns:
            u0: First optimal control action in physical units (nu,).
            info: Solver information dict.
        """
        x0_unscaled = np.atleast_1d(x0_unscaled).flatten()
        
        # Scale initial state
        x0_scaled = self._scale_x(x0_unscaled)
        
        # Solve NMPC
        z_opt, info = self.solve(x0_scaled)

        # Extract w_opt
        w_opt = z_opt[:self.nlp["n_w"]]
        
        # Extract x_1,...,x_N and u_0,...,u_{N-1}
        _, U_opt = self.extract_states_controls(w_opt)

        # Unscale controls
        U_opt_unscaled = U_opt.reshape((self.N, self.nu)) * self.u_scale + self.u_min
        u_opt = U_opt_unscaled

        return u_opt[0], info
