"""Solver infrastructure for nonlinear and block-structured systems."""

from .block_system import BlockStructuredSystem, SchurComplementSolver
from .nonlinear_solvers import ImplicitEquationSolver, PETSC_AVAILABLE, UMFPACK_AVAILABLE
from .pcr import pcr_solve

__all__ = [
    "BlockStructuredSystem",
    "SchurComplementSolver",
    "ImplicitEquationSolver",
    "PETSC_AVAILABLE",
    "UMFPACK_AVAILABLE",
    "pcr_solve",
]
