import numpy as np

from solve_nivp.mjf_integration import MJFIntegrationMethod


class _FakeStepper:
    def __init__(self, n_react, converged):
        self.theta = 0.5
        self.n_react = n_react
        self._converged = converged

    def step(self, t, y, aux, h):
        info = {
            "soccp_converged": self._converged,
            "soccp_residual": 0.0,
            "p_contact": np.zeros(self.n_react),
        }
        return np.asarray(y, dtype=float).copy(), dict(aux), info


def test_reaction_history_sized_from_stepper_n_react():
    # n_react = 3 (e.g. 3D contact) differs from 2 * n_c = 4.
    stepper = _FakeStepper(n_react=3, converged=True)
    mjf = MJFIntegrationMethod(stepper, aux0={}, n_c=2, reaction_scale=1.0)
    assert mjf.reaction_history[0].shape[0] == 3


def test_step_fails_when_soccp_not_converged():
    stepper = _FakeStepper(n_react=2, converged=False)
    mjf = MJFIntegrationMethod(stepper, aux0={}, n_c=1, reaction_scale=1.0)
    y = np.array([0.0, 0.0])
    _y1, _fk, _err, success, _it = mjf.step(lambda *a: None, 0.0, y, 0.1)
    assert success is False
