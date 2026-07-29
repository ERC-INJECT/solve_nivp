"""Open-contact gating of slip-weakening accumulation.

With ``weaken_while_open=False`` the ``h * |v_t|`` slip increment is gated on
the step's total normal reaction (``info['p_contact_effective']``, falling
back to ``info['p_contact']`` for steppers without offset forces), so
separated faces do not weaken mu.  The default ``True`` keeps the ungated
accumulation (slip history == theta-point quadrature of |v_T|).
"""
import numpy as np

from solve_nivp.mjf_integration import MJFIntegrationMethod


class _StubStepper:
    """Constant-velocity 2-contact stepper: contact 0 closed, contact 1 open."""

    theta = 0.5
    n_react = 4
    block_slices = [slice(0, 2), slice(2, 4)]

    def __init__(self, info_key="p_contact_effective"):
        self.info_key = info_key

    def step(self, t, y, aux, h):
        # state layout: [v_t(2), slip(2)]; velocities stay constant
        y1 = np.asarray(y, float).copy()
        info = {self.info_key: np.array([1.0, 0.3, 0.0, 0.0])}
        return y1, dict(aux), info


def _method(stepper, **kw):
    return MJFIntegrationMethod(
        stepper, {"mu": np.array([0.6, 0.6]), "cum_slip": np.zeros(2)},
        n_c=2, reaction_scale=0.5,
        slip_slice=slice(2, 4), mu_from_slip=lambda s: 0.6 - np.asarray(s, float),
        vel_t_extract=np.eye(2), n_phys=2, **kw,
    )


def test_gated_open_contact_accumulates_no_slip():
    m = _method(_StubStepper(), weaken_while_open=False)
    y0 = np.array([1.0, 1.0, 0.0, 0.0])
    y1, aux1, info1 = m._step_weakening(0.0, y0, 0.1)
    np.testing.assert_allclose(y1[2:], [0.1, 0.0])
    np.testing.assert_allclose(aux1["cum_slip"], [0.1, 0.0])
    np.testing.assert_allclose(aux1["mu"], [0.5, 0.6])


def test_default_preserves_ungated_accumulation():
    m = _method(_StubStepper())
    y0 = np.array([1.0, 1.0, 0.0, 0.0])
    y1, aux1, _ = m._step_weakening(0.0, y0, 0.1)
    np.testing.assert_allclose(y1[2:], [0.1, 0.1])
    np.testing.assert_allclose(aux1["mu"], [0.5, 0.5])


def test_gating_falls_back_to_p_contact_without_offsets():
    m = _method(_StubStepper(info_key="p_contact"), weaken_while_open=False)
    y0 = np.array([1.0, 1.0, 0.0, 0.0])
    y1, aux1, _ = m._step_weakening(0.0, y0, 0.1)
    np.testing.assert_allclose(aux1["cum_slip"], [0.1, 0.0])
