import numpy as np


def test_package_import_and_basic_api():
    import Solve_IVP_NS as sivp

    # Simple linear ODE: dy/dt = -y
    def fun(t, y):
        return -y

    y0 = np.array([1.0, -2.0, 0.5])
    t_span = (0.0, 0.1)

    # Use identity projection and semismooth newton with fixed step
    result = sivp.solve_ivp_ns(
        fun=fun,
        t_span=t_span,
        y0=y0,
        method='backward_euler',
        projection='identity',
        solver='semismooth_newton',
        adaptive=False,
        h0=0.05,
    )

    t_values, y_values, h_values, fk_values, errors = result

    # Basic shape checks
    assert t_values.ndim == 1 and y_values.ndim == 2
    assert y_values.shape[1] == y0.size
    # Monotonic time and at least one step
    assert t_values[0] == t_span[0]
    assert t_values[-1] == t_span[1]

    # Backward Euler should be close to exact solution over small horizon
    y_exact = y0 * np.exp(-(t_span[1] - t_span[0]))
    np.testing.assert_allclose(y_values[-1], y_exact, rtol=5e-2, atol=1e-6)
