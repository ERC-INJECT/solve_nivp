"""Optional dependency helpers for the Solve_IVP_NS RL subpackage."""

from __future__ import annotations

import importlib
from typing import Final

__all__ = ["REQUIRED_PACKAGES", "ensure_rl_dependencies"]

REQUIRED_PACKAGES: Final[tuple[str, ...]] = (
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
)


def ensure_rl_dependencies() -> None:
    """Raise ``ImportError`` if any optional RL dependency is missing.

    This helper lets modules delay dependency checks until runtime so the
    base package remains importable without ``extras`` installed.
    """

    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if missing:
        extras = "Solve_IVP_NS[rl]"
        missing_str = ", ".join(sorted(missing))
        raise ImportError(
            "The Solve_IVP_NS RL utilities require optional packages that are not installed. "
            f"Missing: {missing_str}. Install them via 'pip install {extras}'."
        )
