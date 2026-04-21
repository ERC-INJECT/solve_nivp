"""Tests for Macklin (2019) velocity-level contact backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def _bouncing_ball_setup():
    """2D bouncing ball: y = [vx, vy, qx, qy], 1 normal + 1 tangential contact."""
    mass = 1.0
    g_acc = 9.81
    gravity = np.array([0.0, -g_acc])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y):
        return np.concatenate([mass * gravity, y[:2]])

    def gap_func(y, t):
        return np.array([y[3]])

    drop_height = 0.5
    y0 = np.array([0.0, 0.0, 0.0, drop_height])
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.0)]
    return A, rhs, y0, contacts, gap_func, drop_height, g_acc


def test_macklin_block_shapes():
    from solve_nivp.macklin_contact import build_macklin_contact_blocked

    A, rhs, y0, contacts, gap_func, _, _ = _bouncing_ball_setup()
    bs = build_macklin_contact_blocked(
        A=A, rhs_smooth=rhs, y0=y0, contacts=contacts, gap_func=gap_func,
    )
    assert bs.n_phys == 4
    assert bs.n_react == 2

    n_p, n_r = bs.n_phys, bs.n_react
    y_aug = np.zeros(n_p + n_r)
    y_aug[:n_p] = y0
    h = 0.01
    blocks = bs.assemble_blocks(y_aug, t=h, h=h, y_prev=y_aug)

    assert blocks["H"].shape == (n_p, n_p)
    assert blocks["B_top"].shape == (n_p, n_r)
    assert blocks["B_bot"].shape == (n_r, n_p)
    assert blocks["C"].shape == (n_r, n_r)
    assert blocks["g"].shape == (n_p,)
    assert blocks["h_c"].shape == (n_r,)
    assert blocks["precond_diag"].shape == (n_r,)
