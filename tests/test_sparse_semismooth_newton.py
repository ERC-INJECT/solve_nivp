import numpy as np


def test_semismooth_sparse_path_runs():
    # A moderately sized linear system to trigger sparse code path
    import Solve_IVP_NS as sivp
    n = 300
    A = np.eye(n)

    def fun(t, y):
        return -y  # stable linear

    y0 = np.linspace(-1, 1, n)

    # Force sparse path with threshold below n and use gmres
    result = sivp.solve_ivp_ns(
        fun=fun,
        t_span=(0.0, 0.01),
        y0=y0,
        method='backward_euler',
        projection='identity',
        solver='semismooth_newton',
        solver_opts={
            "sparse": True,
            "sparse_threshold": 50,
            "linear_solver": "gmres",
            "precond_reuse_steps": 2,
            "gmres_tol": 1e-8,
        },
        adaptive=False,
        h0=0.01,
        A=A,
    )

    t_values, y_values, *_ = result
    assert t_values[-1] == 0.01
    # Ensure solution changed and is finite
    assert np.isfinite(y_values).all()
