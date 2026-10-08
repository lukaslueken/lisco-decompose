"""LISCO - Learning-based Iterative Solver for Constraint Optimization.

Reduced to the components used by the paper experiments:
- FeedforwardNN: the network used by all learned components
- Solver / SolverDecomposed: monolithic and time-decomposed learned solvers
- NLPConfig: NLP handler configuration
- SolverTrainConfig / train_solver: solver training
- ApproxMPC / ApproxMPCTrainConfig / train_approx_mpc: approximate MPC baseline
"""

from src.lisco.nn import (
    FeedforwardNN,
    FeedforwardNNConfig,
    Solver,
    SolverDecomposed,
    ApproxMPC,
)
from src.lisco.nlp_handler import NLPConfig
from src.lisco.training import (
    TrainLogger,
    SolverTrainConfig,
    train_solver,
    ApproxMPCTrainConfig,
    train_approx_mpc,
)

__all__ = [
    "FeedforwardNN",
    "FeedforwardNNConfig",
    "Solver",
    "SolverDecomposed",
    "ApproxMPC",
    "NLPConfig",
    "TrainLogger",
    "SolverTrainConfig",
    "train_solver",
    "ApproxMPCTrainConfig",
    "train_approx_mpc",
]
