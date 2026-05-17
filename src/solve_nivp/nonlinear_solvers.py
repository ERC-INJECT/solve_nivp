"""Backward-compatible import path for nonlinear solver infrastructure."""

from __future__ import annotations

import sys as _sys

from .solvers import nonlinear_solvers as _impl

_sys.modules[__name__] = _impl
