"""Backward-compatible import path for the PCR linear solver."""

from __future__ import annotations

import sys as _sys

from .solvers import pcr as _impl

_sys.modules[__name__] = _impl
