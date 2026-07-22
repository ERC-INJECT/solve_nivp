"""Run-time control utilities for the MJF drivers: progress and checkpoints.

Long Moreau-Jean-Fremond marches (fixed-step or adaptive) previously ran
silently and kept every accepted state in memory until the driver returned.
This module provides the shared machinery the three drivers
(:func:`~solve_nivp.mjf_integration.solve_mjf_fixed_step`,
:func:`~solve_nivp.moreau_jean_fremond.solve_mjf_adaptive`,
:func:`~solve_nivp.moreau_jean_fremond.solve_mjf_adaptive_ratio`) use to

* report progress during the march (a wall-clock-throttled heartbeat print
  and/or a user ``on_step`` callback), and
* periodically write a restart checkpoint to disk, so a multi-day (or
  multi-year simulated time) run can be killed and resumed.

The restart datum of an MJF march is ``(t, y, aux)``: the stepper
externalises all correctness-bearing cross-step state (friction law state
such as ``mu`` / ``cum_slip``, the contact warm-start impulse, the endpoint
contact velocity) through the ``aux`` dict it returns from ``step``.  Hidden
stepper attributes are performance caches (theta factorisations, solver warm
starts) that rebuild transparently after a reload.  A checkpoint therefore
stores exactly ``(t, y, aux)`` plus a little driver bookkeeping.

Public API
----------
MJFRunMonitor        : per-run progress + checkpoint coordinator
save_mjf_checkpoint  : atomic ``.npz`` checkpoint writer
load_mjf_checkpoint  : checkpoint reader returning a plain dict
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import numpy as np

CHECKPOINT_SCHEMA = "mjf-checkpoint-v1"

# Prefixes used to flatten the aux / extras dicts into flat npz keys.
_AUX_PREFIX = "aux_"
_EXTRA_PREFIX = "extra_"


def _as_savable_array(name: str, value: Any) -> np.ndarray:
    """Convert one aux/extras entry to a numeric/bool array or raise.

    Checkpoints are written with ``allow_pickle=False`` so they stay portable
    and safe to load; anything that does not round-trip through a plain numpy
    array (nested dicts, objects, strings) is rejected with the offending key
    named, rather than silently producing an unloadable file.
    """
    arr = np.asarray(value)
    if arr.dtype == object or arr.dtype.kind not in "fiub":
        raise TypeError(
            f"checkpoint entry {name!r} has dtype {arr.dtype}; only numeric "
            f"or boolean values can be checkpointed (allow_pickle=False)"
        )
    return arr


def _from_saved_array(arr: np.ndarray) -> Any:
    """Undo :func:`_as_savable_array`: 0-d arrays come back as python scalars.

    Scalar aux entries (e.g. a step-size scale) are stored as 0-d arrays by
    ``np.savez``; returning them as plain python scalars keeps them behaving
    exactly as they did before the save (the drivers' ``_copy_aux`` helpers
    pass scalars through untouched).
    """
    a = np.asarray(arr)
    if a.ndim == 0:
        return a[()].item()
    return a.copy()


def save_mjf_checkpoint(path, *, t: float, y: np.ndarray, aux: dict,
                        h: float, step_index: int, driver: str = "mjf",
                        label: str = "MJF",
                        extras: Optional[dict] = None) -> str:
    """Write an MJF restart checkpoint atomically as an ``.npz`` file.

    The file is first written to ``<path>.tmp`` and then moved into place
    with :func:`os.replace`, so a crash mid-write can never corrupt an
    existing checkpoint: the previous complete checkpoint (if any) survives.

    Parameters
    ----------
    path : str or os.PathLike
        Destination file.  A ``.npz`` suffix is appended if missing.
    t : float
        Simulation time of the checkpointed state.
    y : (n,) ndarray
        Full state vector at time ``t``.
    aux : dict
        The MJF auxiliary state as returned by the stepper for this step
        (friction state, warm-start impulse, ...).  Values must be numeric
        or boolean arrays/scalars.
    h : float
        Step size to resume with (the fixed step, or the committed/held
        adaptive step size).
    step_index : int
        Number of accepted steps taken up to this state (bookkeeping only).
    driver : str
        Identifier of the driver that wrote the checkpoint
        (``"fixed"`` / ``"adaptive"`` / ``"ratio"``).
    label : str
        The run label, echoed on resume for log continuity.
    extras : dict, optional
        Additional numeric entries (e.g. the last reported reaction row, the
        controller's next-step proposal).  Restored under ``"extras"``.

    Returns
    -------
    str
        The final checkpoint path.
    """
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path = path + ".npz"
    payload: dict = {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "driver": np.asarray(str(driver)),
        "label": np.asarray(str(label)),
        "t": _as_savable_array("t", float(t)),
        "y": _as_savable_array("y", np.asarray(y, dtype=float)),
        "h": _as_savable_array("h", float(h)),
        "step_index": _as_savable_array("step_index", int(step_index)),
    }
    for key, val in (aux or {}).items():
        payload[_AUX_PREFIX + str(key)] = _as_savable_array(f"aux[{key!r}]", val)
    for key, val in (extras or {}).items():
        payload[_EXTRA_PREFIX + str(key)] = _as_savable_array(
            f"extras[{key!r}]", val)

    # PID-unique temp name so concurrent runs mistakenly sharing a
    # checkpoint_path cannot interleave writes into each other's temp file;
    # flush+fsync before the rename so the replacement is atomic against
    # machine crashes too (rename-before-writeback would otherwise be able
    # to publish a truncated file over the previous good checkpoint).
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(fh, **payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def load_mjf_checkpoint(path) -> dict:
    """Load a checkpoint written by :func:`save_mjf_checkpoint`.

    Returns
    -------
    dict
        ``{"t", "y", "h", "step_index", "driver", "label", "aux", "extras",
        "schema"}`` with ``aux`` and ``extras`` restored as dicts.  Scalar
        entries come back as python scalars, arrays as fresh ndarray copies.
    """
    path = os.fspath(path)
    if not path.endswith(".npz") and not os.path.exists(path):
        path = path + ".npz"
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"][()]) if "schema" in data else ""
        if schema != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"{path}: not an MJF checkpoint (schema {schema!r}, "
                f"expected {CHECKPOINT_SCHEMA!r})"
            )
        out = {
            "schema": schema,
            "driver": str(data["driver"][()]),
            "label": str(data["label"][()]),
            "t": float(data["t"][()]),
            "y": np.asarray(data["y"], dtype=float).copy(),
            "h": float(data["h"][()]),
            "step_index": int(data["step_index"][()]),
            "aux": {},
            "extras": {},
        }
        for key in data.files:
            if key.startswith(_AUX_PREFIX):
                out["aux"][key[len(_AUX_PREFIX):]] = _from_saved_array(data[key])
            elif key.startswith(_EXTRA_PREFIX):
                out["extras"][key[len(_EXTRA_PREFIX):]] = _from_saved_array(
                    data[key])
    return out


def resolve_mjf_checkpoint(resume_from) -> dict:
    """Accept a path or an already-loaded checkpoint dict and return the dict.

    Drivers take ``resume_from`` as either form; a dict is validated to look
    like a loaded checkpoint (must carry ``t``, ``y`` and ``aux``).
    """
    if isinstance(resume_from, dict):
        missing = [k for k in ("t", "y", "aux") if k not in resume_from]
        if missing:
            raise ValueError(
                f"resume_from dict is missing checkpoint keys {missing}; "
                f"pass a path or the dict returned by load_mjf_checkpoint"
            )
        return resume_from
    return load_mjf_checkpoint(resume_from)


class MJFRunMonitor:
    """Progress heartbeat, user callback, and periodic checkpointing.

    One instance accompanies one driver run.  The driver calls
    :meth:`after_step` after every *accepted* step, :meth:`after_reject`
    after every rejected attempt (adaptive drivers only), and
    :meth:`finish` once when the march ends.  Everything is opt-in:

    * ``progress_interval_s`` -- when set, a single status line is printed at
      most every that-many wall-clock seconds (``flush=True`` so it appears
      immediately in notebooks and redirected logs), plus one final line from
      :meth:`finish`.  The line reports accepted/rejected step counts,
      simulation time, percent of the horizon, current step size, elapsed
      wall time and a wall-time ETA extrapolated from the simulated-time
      rate so far.  On a resumed run the percentage refers to the CURRENT
      SEGMENT's horizon (``t_start`` is the checkpointed time), while the
      step index continues globally across segments.
    * ``on_step`` -- ``on_step(t, y, aux, info)`` called after every accepted
      step, for custom progress bars, live plots, or user-side snapshots.
      Exceptions propagate and abort the march (they are user code).
    * ``checkpoint_path`` (+ ``checkpoint_every_steps`` /
      ``checkpoint_every_walltime_s``) -- periodic restart checkpoints via
      :func:`save_mjf_checkpoint`; whichever trigger fires first wins, and
      both counters reset after each write.  With a path but no trigger, a
      single checkpoint is written by :meth:`finish` at the end of the march.
      Each write replaces the previous file atomically.

    The monitor is deliberately stateless about the physics: the driver hands
    it ``(t, y, aux)`` and it never mutates them.
    """

    def __init__(self, *, t_start: float, t_end: float, label: str = "MJF",
                 driver: str = "mjf",
                 progress_interval_s: Optional[float] = None,
                 on_step: Optional[Callable[..., None]] = None,
                 checkpoint_path=None,
                 checkpoint_every_steps: Optional[int] = None,
                 checkpoint_every_walltime_s: Optional[float] = None,
                 step_index0: int = 0):
        if checkpoint_path is None and (checkpoint_every_steps is not None
                                        or checkpoint_every_walltime_s is not None):
            raise ValueError(
                "checkpoint_every_steps / checkpoint_every_walltime_s require "
                "checkpoint_path"
            )
        if progress_interval_s is not None and not (progress_interval_s > 0.0):
            raise ValueError("progress_interval_s must be positive")
        if checkpoint_every_steps is not None and int(checkpoint_every_steps) < 1:
            raise ValueError("checkpoint_every_steps must be >= 1")
        if (checkpoint_every_walltime_s is not None
                and not (checkpoint_every_walltime_s > 0.0)):
            raise ValueError("checkpoint_every_walltime_s must be positive")

        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.label = str(label)
        self.driver = str(driver)
        self.progress_interval_s = (
            None if progress_interval_s is None else float(progress_interval_s))
        self.on_step = on_step
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every_steps = (
            None if checkpoint_every_steps is None else int(checkpoint_every_steps))
        self.checkpoint_every_walltime_s = (
            None if checkpoint_every_walltime_s is None
            else float(checkpoint_every_walltime_s))

        # counters (step_index0 lets a resumed run continue its numbering)
        self.n_accepted = 0
        self.n_rejected = 0
        self._step_index0 = int(step_index0)
        self._wall_start = time.monotonic()
        self._wall_last_beat = self._wall_start
        self._wall_last_ckpt = self._wall_start
        self._steps_since_ckpt = 0
        self.n_checkpoints_written = 0

    # -- public counters ------------------------------------------------------
    @property
    def step_index(self) -> int:
        """Global accepted-step index (continues across resumed segments)."""
        return self._step_index0 + self.n_accepted

    # -- driver hooks ----------------------------------------------------------
    def after_step(self, t: float, y: np.ndarray, aux: dict, info: dict,
                   *, h: Optional[float] = None,
                   extras: Optional[dict] = None) -> None:
        """Record one accepted step: callback, heartbeat, maybe checkpoint."""
        self.n_accepted += 1
        if self.on_step is not None:
            self.on_step(t, y, aux, info)
        if self.progress_interval_s is not None:
            now = time.monotonic()
            if now - self._wall_last_beat >= self.progress_interval_s:
                self._print_beat(t, h, now)
                self._wall_last_beat = now
        if self.checkpoint_path is not None:
            self._steps_since_ckpt += 1
            if self._checkpoint_due():
                self._write_checkpoint(t, y, aux, h, extras)

    def after_reject(self, t: float, h: Optional[float] = None) -> None:
        """Record one rejected attempt (heartbeat bookkeeping only)."""
        self.n_rejected += 1
        if self.progress_interval_s is not None:
            now = time.monotonic()
            if now - self._wall_last_beat >= self.progress_interval_s:
                self._print_beat(t, h, now)
                self._wall_last_beat = now

    def finish(self, t: float, y: Optional[np.ndarray] = None,
               aux: Optional[dict] = None, *, h: Optional[float] = None,
               extras: Optional[dict] = None, status: str = "done") -> None:
        """Close out the run: final heartbeat line and final checkpoint.

        The final checkpoint is written whenever a ``checkpoint_path`` is
        configured and the driver supplies the state, regardless of the
        periodic triggers -- so the file on disk always ends at the final
        state of a completed (or cleanly aborted) march.  ``status`` is shown
        on the final heartbeat line (drivers pass ``"aborted"`` when the
        march ended on a step failure rather than at the horizon).
        """
        if self.progress_interval_s is not None:
            self._print_beat(t, h, time.monotonic(), final_status=status)
        if self.checkpoint_path is not None and y is not None and aux is not None:
            self._write_checkpoint(t, y, aux, h, extras)

    # -- internals --------------------------------------------------------------
    def _checkpoint_due(self) -> bool:
        if (self.checkpoint_every_steps is not None
                and self._steps_since_ckpt >= self.checkpoint_every_steps):
            return True
        if self.checkpoint_every_walltime_s is not None:
            if (time.monotonic() - self._wall_last_ckpt
                    >= self.checkpoint_every_walltime_s):
                return True
        return False

    def _write_checkpoint(self, t, y, aux, h, extras) -> None:
        save_mjf_checkpoint(
            self.checkpoint_path,
            t=float(t), y=y, aux=aux,
            h=float(h) if h is not None else 0.0,
            step_index=self.step_index,
            driver=self.driver, label=self.label,
            extras=extras,
        )
        self.n_checkpoints_written += 1
        self._steps_since_ckpt = 0
        self._wall_last_ckpt = time.monotonic()

    def _print_beat(self, t, h, now, final_status: Optional[str] = None) -> None:
        elapsed = now - self._wall_start
        span = self.t_end - self.t_start
        frac = (float(t) - self.t_start) / span if span > 0.0 else 1.0
        frac = min(max(frac, 0.0), 1.0)
        if final_status is not None:
            eta_txt = final_status
        elif frac >= 1.0:
            eta_txt = "done"
        elif frac > 0.0:
            eta_txt = f"ETA {elapsed * (1.0 - frac) / frac:.0f}s"
        else:
            eta_txt = "ETA --"
        h_txt = f", h={float(h):.3e}" if h is not None else ""
        rej_txt = f" (+{self.n_rejected} rej)" if self.n_rejected else ""
        ckpt_txt = (f", ckpt#{self.n_checkpoints_written}"
                    if self.n_checkpoints_written else "")
        print(
            f"{self.label}: step {self.step_index}{rej_txt}, "
            f"t={float(t):.6e} ({100.0 * frac:5.1f}%){h_txt}, "
            f"wall {elapsed:.0f}s, {eta_txt}{ckpt_txt}",
            flush=True,
        )
