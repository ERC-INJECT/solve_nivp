"""Backward-compatible import path for block-structured solver infrastructure."""

from __future__ import annotations

import sys as _sys

from .solvers import block_system as _impl

_sys.modules[__name__] = _impl
