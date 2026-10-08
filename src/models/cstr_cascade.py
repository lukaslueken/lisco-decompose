"""CSTR Cascade Model

Continuous-time dynamics of a cascade of CSTRs and their implicit-Euler
discretization, used as the NMPC benchmark.

The model implements a cascade of nr reactors with:
- States (per reactor): cA, cB, cR, cS (concentrations) and Tr (temperature)
- Inputs (per reactor): uA, uB (feed rates)
- Fixed parameter: Tj (jacket temperature, same for all reactors)

Reactions:
- A + B -> R (desired product), rate k1 * cA * cB
- 2A -> S (undesired side product), rate k2 * cA^2

Flow Structure:
    The first reactor receives an implicit solvent feed stream at volumetric rate
    V_out [L/s] with zero concentrations (cA=cB=cR=cS=0) and temperature Tr_in.
    Pure components A and B are added at molar rates uA, uB [mol/s]. This maintains
    constant volume in all reactors.

    Schematic:
    
                        uA[0], uB[0] (mol/s)
                             |
        Solvent feed    +---------+         +---------+
        (V_out, Tr_in)  | Reactor |  V_out  | Reactor |  V_out
        cA=cB=cR=cS=0 ->|    1    |-------->|    2    |--------> ...
                        |         |         |         |
                        +----+----+         +----+----+
                             |                   |
                            Tj                  Tj

Unit System:
    The model uses Liters as the base volume unit, making parameter values
    physically consistent:

    | Parameter | Value | Unit        | Interpretation                    |
    |-----------|-------|-------------|-----------------------------------|
    | V_R       | 15    | L           | Total reactor volume              |
    | V_out     | 1     | L/s         | Volumetric flow rate              |
    | rho       | 1     | kg/L        | Density (= 1000 kg/m³ for water)  |
    | cp        | 0.5   | kJ/(kg·K)   | Heat capacity (organic solvent)   |
    | kA        | 2     | kW/K        | Heat transfer coeff. × area       |
    | uA, uB    | -     | mol/s       | Molar feed rates                  |
    | cA, cB... | -     | mol/L       | Concentrations                    |
    | Tr, Tj    | -     | °C          | Temperatures                      |
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

import casadi as ca
import numpy as np


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class CSTRParams:
    """Parameters for the simplified CSTR cascade model.

    The model uses Liters as the base volume unit. With rho=1 kg/L, this
    corresponds to a density of 1000 kg/m³ (water/aqueous solution).

    Attributes:
        nr: Number of reactors in the cascade.
        dt: Discretization time step [s].
        V_R: Total reactor volume [L].
        V_out: Volumetric outflow rate [L/s].
        k1: Reaction rate constant for A + B -> R [L/(mol·s)].
        k2: Reaction rate constant for 2A -> S [L/(mol·s)].
        delH1: Reaction enthalpy for reaction 1 [kJ/mol] (negative = exothermic).
        delH2: Reaction enthalpy for reaction 2 [kJ/mol] (negative = exothermic).
        Tr_in: Inlet temperature of solvent feed [°C].
        kA: Heat transfer coefficient × area [kW/K].
        rho: Density [kg/L] (1 kg/L = 1000 kg/m³).
        cp: Heat capacity [kJ/(kg·K)].
        Ea1: Activation energy for reaction 1 [J/mol].
        Ea2: Activation energy for reaction 2 [J/mol].
        R_gas: Universal gas constant [J/(mol·K)].
    """

    # Cascade structure
    nr: int = 1

    # Discretization
    dt: float = 1.0  # [s]

    # Physical parameters
    V_R: float = 15.0  # [L] Total reactor volume
    V_out: float = 1.0  # [L/s] Volumetric outflow rate
    k1: float = 2.0  # [L/(mol·s)] Reaction rate constant for A + B -> R
    k2: float = 2.0  # [L/(mol·s)] Reaction rate constant for 2A -> S
    delH1: float = -100.0  # [kJ/mol] Reaction enthalpy (exothermic)
    delH2: float = -50.0  # [kJ/mol] Reaction enthalpy (exothermic)
    Tr_in: float = 60.0  # [°C] Inlet temperature
    kA: float = 2.0  # [kW/K] Heat transfer coefficient * area
    rho: float = 1.0  # [kg/L] Density (= 1000 kg/m³)
    cp: float = 0.5  # [kJ/(kg·K)] Heat capacity
    Ea1: float = 500.0  # [J/mol] Activation energy for reaction 1
    Ea2: float = 600.0  # [J/mol] Activation energy for reaction 2
    R_gas: float = 8.3145  # [J/(mol·K)] Universal gas constant
    Tj: float = 50.0  # [°C] Fixed jacket temperature (same for all reactors)

    # Derived quantities (computed in __post_init__)
    V_i: float = field(init=False)  # [L] Volume per reactor
    nx: int = field(init=False)  # Total state dimension
    nu: int = field(init=False)  # Total input dimension

    def __post_init__(self) -> None:
        """Compute derived quantities and validate parameters."""
        self.V_i = self.V_R / self.nr
        self.nx = 5 * self.nr  # 5 states per reactor: cA, cB, cR, cS, Tr
        self.nu = 2 * self.nr  # 2 inputs per reactor: uA, uB (Tj is fixed)

    def to_dict(self) -> dict:
        """Serialize init fields to a plain dict (excludes derived V_i, nx, nu)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "CSTRParams":
        """Reconstruct a CSTRParams from a dict produced by to_dict()."""
        return cls(**d)

    def get_default_x0(self) -> np.ndarray:
        """Get default initial state (near-zero concentrations, inlet temperature)."""
        x0 = 1e-3 * np.ones(self.nx)
        x0[4 * self.nr :] = self.Tr_in  # Initial temperature = inlet temperature
        return x0


# =============================================================================
# CSTR Cascade Model Class
# =============================================================================


class CSTRCascade:
    """Simplified CSTR Cascade model for simulation.

    This class provides:
    - Continuous-time ODE dynamics (CasADi symbolic)
    - Discrete-time integrator (implicit Euler)
    - One-step simulation

    Example:
        >>> model = CSTRCascade(CSTRParams(nr=3))
        >>> x0 = model.params.get_default_x0()
        >>> u = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 50, 50, 50])
        >>> x_next = model.simulate_step(x0, u)
    """

    def __init__(
        self,
        params: CSTRParams | None = None,
        integrator: Literal["implicit_euler"] = "implicit_euler",
    ) -> None:
        """Initialize the CSTR cascade model.

        Args:
            params: Model parameters. If None, uses defaults.
            integrator: Integration method ('implicit_euler').
        """
        self.params = params if params is not None else CSTRParams()
        self._integrator_method = integrator

        # Build CasADi functions
        self._build_symbolic_variables()
        self._build_dynamics()
        self._build_integrator(integrator)

    # -------------------------------------------------------------------------
    # Symbolic Variables
    # -------------------------------------------------------------------------

    def _build_symbolic_variables(self) -> None:
        """Create symbolic variables for states and inputs."""
        nr = self.params.nr

        # States (per reactor, then stacked)
        self.sym_cA = ca.SX.sym("cA", nr)
        self.sym_cB = ca.SX.sym("cB", nr)
        self.sym_cR = ca.SX.sym("cR", nr)
        self.sym_cS = ca.SX.sym("cS", nr)
        self.sym_Tr = ca.SX.sym("Tr", nr)
        self.sym_x = ca.vertcat(
            self.sym_cA, self.sym_cB, self.sym_cR, self.sym_cS, self.sym_Tr
        )

        # Inputs (per reactor, then stacked)
        self.sym_uA = ca.SX.sym("uA", nr)
        self.sym_uB = ca.SX.sym("uB", nr)
        self.sym_u = ca.vertcat(self.sym_uA, self.sym_uB)

    # -------------------------------------------------------------------------
    # Dynamics
    # -------------------------------------------------------------------------

    def _build_dynamics(self) -> None:
        """Build the continuous-time ODE dynamics as a CasADi function.

        Flow structure:
        - First reactor receives an implicit solvent feed at rate V_out [L/s]
          with zero concentrations and temperature Tr_in. Pure components A and B
          are added at molar rates uA, uB [mol/s].
        - Subsequent reactors receive the effluent from the previous reactor.
        - All reactors have constant volume V_i = V_R / nr.
        """
        p = self.params
        nr = p.nr

        # Temperature in Kelvin, clipped from below at T_min. The clip is a
        # numerical safeguard for the Arrhenius terms, which are singular at
        # 0 K, against non-physical iterates; it is inactive in the operating
        # range (Tr in [20, 80] °C, i.e. above 293 K) and not part of the
        # model in the paper.
        T_kelvin_raw = self.sym_Tr + 273.15
        T_min = 150.0  # Minimum temperature [K]
        T_kelvin = T_min + ca.fmax(0, T_kelvin_raw - T_min)

        # Reaction rates with Arrhenius kinetics
        # k_eff = k * exp(-Ea / (R * T))
        k1_eff = p.k1 * ca.exp(-p.Ea1 / (p.R_gas * T_kelvin))
        k2_eff = p.k2 * ca.exp(-p.Ea2 / (p.R_gas * T_kelvin))

        # Reaction rates [mol/(L·s)]
        r1 = k1_eff * self.sym_cA * self.sym_cB  # A + B -> R
        r2 = k2_eff * self.sym_cA**2  # 2A -> S

        # Heat generation coefficients
        # Q = -delH / (rho * cp) * k
        Q1 = -p.delH1 / (p.rho * p.cp) * p.k1
        Q2 = -p.delH2 / (p.rho * p.cp) * p.k2

        # Build ODEs reactor by reactor
        rhs_cA = []
        rhs_cB = []
        rhs_cR = []
        rhs_cS = []
        rhs_Tr = []

        for i in range(nr):
            # Flow terms
            if i == 0:
                # First reactor: outflow only (feed via uA, uB)
                flow_cA = -p.V_out / p.V_i * self.sym_cA[i]
                flow_cB = -p.V_out / p.V_i * self.sym_cB[i]
                flow_cR = -p.V_out / p.V_i * self.sym_cR[i]
                flow_cS = -p.V_out / p.V_i * self.sym_cS[i]
                flow_Tr = p.V_out / p.V_i * (p.Tr_in - self.sym_Tr[i])
            else:
                # Subsequent reactors: inflow from previous reactor
                flow_cA = p.V_out / p.V_i * (self.sym_cA[i - 1] - self.sym_cA[i])
                flow_cB = p.V_out / p.V_i * (self.sym_cB[i - 1] - self.sym_cB[i])
                flow_cR = p.V_out / p.V_i * (self.sym_cR[i - 1] - self.sym_cR[i])
                flow_cS = p.V_out / p.V_i * (self.sym_cS[i - 1] - self.sym_cS[i])
                flow_Tr = p.V_out / p.V_i * (self.sym_Tr[i - 1] - self.sym_Tr[i])

            # Concentration ODEs
            # dcA/dt = flow - r1 - 2*r2 + feed_A/V
            rhs_cA.append(flow_cA - r1[i] - 2 * r2[i] + self.sym_uA[i] / p.V_i)

            # dcB/dt = flow - r1 + feed_B/V
            rhs_cB.append(flow_cB - r1[i] + self.sym_uB[i] / p.V_i)

            # dcR/dt = flow + r1 (product)
            rhs_cR.append(flow_cR + r1[i])

            # dcS/dt = flow + r2 (side product)
            rhs_cS.append(flow_cS + r2[i])

            # Temperature ODE
            # dTr/dt = flow_T + heat_transfer + reaction_heat
            heat_transfer = p.kA / (p.rho * p.cp * p.V_i) * (p.Tj - self.sym_Tr[i])
            reaction_heat = (
                Q1 * ca.exp(-p.Ea1 / (p.R_gas * T_kelvin[i])) * self.sym_cA[i] * self.sym_cB[i]
                + Q2 * ca.exp(-p.Ea2 / (p.R_gas * T_kelvin[i])) * self.sym_cA[i] ** 2
            )
            rhs_Tr.append(flow_Tr + heat_transfer + reaction_heat)

        # Stack into ODE right-hand side
        rhs = ca.vertcat(
            ca.vertcat(*rhs_cA),
            ca.vertcat(*rhs_cB),
            ca.vertcat(*rhs_cR),
            ca.vertcat(*rhs_cS),
            ca.vertcat(*rhs_Tr),
        )

        # Symbolic expression for ODE
        self.ode = rhs

        # Create CasADi function: f(x, u) -> dx/dt
        self.f_ode = ca.Function(
            "f_ode",
            [self.sym_x, self.sym_u],
            [rhs],
            ["x", "u"],
            ["xdot"],
        )

    # -------------------------------------------------------------------------
    # Integrator
    # -------------------------------------------------------------------------

    def _build_integrator(self, method: Literal["implicit_euler"] = "implicit_euler") -> None:
        """Build a discrete-time integrator.

        Args:
            method: Integration method ('implicit_euler').
        """
        p = self.params

        if method == "implicit_euler":
            # Implicit Euler: solve x_next = x + dt * f(x_next, u)
            # Use CasADi rootfinder to solve the implicit equation
            x_next_sym = ca.SX.sym("x_next", p.nx)
            x_curr_sym = ca.SX.sym("x_curr", p.nx)
            u_param_sym = ca.SX.sym("u_param", p.nu)
            
            # Residual: g(x_next; x_curr, u) = x_next - x_curr - dt * f(x_next, u) = 0
            residual = x_next_sym - x_curr_sym - p.dt * self.f_ode(x_next_sym, u_param_sym)
            
            # Create rootfinder function
            # g takes (x_next, p) where p = (x_curr, u)
            g_func = ca.Function(
                "g_implicit_euler",
                [x_next_sym, ca.vertcat(x_curr_sym, u_param_sym)],
                [residual],
                ["x", "p"],
                ["g"],
            )
            
            # Newton solver for the implicit equation
            self._rootfinder = ca.rootfinder(
                "implicit_euler_solver", "newton", g_func,
                {"error_on_fail": False}
            )
            
            # Evaluate: use current state as initial guess
            x_next = self._rootfinder(self.sym_x, ca.vertcat(self.sym_x, self.sym_u))

        else:
            raise ValueError(f"Unknown integrator method: {method}")

        self.f_d_sym = x_next

        self.f_d_func = ca.Function(
            "f_d",
            [self.sym_x, self.sym_u],
            [x_next],
            ["x", "u"],
            ["x_next"],
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate one time step.

        Args:
            x: Current state (nx,).
            u: Input (nu,).

        Returns:
            Next state (nx,).
        """
        x = np.atleast_1d(x).flatten()
        u = np.atleast_1d(u).flatten()
        x_next = np.array(self.f_d_func(x, u)).flatten()
        return x_next
