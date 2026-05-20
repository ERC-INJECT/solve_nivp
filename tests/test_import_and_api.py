import numpy as np


def test_package_import_and_basic_api():
    import solve_nivp as sivp

    # Simple linear ODE: dy/dt = -y
    def fun(t, y):
        return -y

    y0 = np.array([1.0, -2.0, 0.5])
    t_span = (0.0, 0.1)

    # Use identity projection and semismooth newton with fixed step
    result = sivp.solve_nivp(
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


def test_package_import_and_default_solve_api_runs():
    import solve_nivp as sivp

    t_values, y_values, h_values, fk_values, errors = sivp.solve_nivp(
        fun=lambda t, y: -y,
        t_span=(0.0, 0.1),
        y0=np.array([1.0]),
        adaptive=False,
        h0=0.05,
    )

    assert t_values[-1] == 0.1
    assert y_values.shape == (3, 1)
    assert h_values.shape == t_values.shape
    assert len(fk_values) == len(errors) == len(t_values) - 1


def test_solver_subpackage_imports_and_compatibility_paths():
    import solve_nivp.block_system as old_block_system
    import solve_nivp.nonlinear_solvers as old_nonlinear_solvers
    import solve_nivp.pcr as old_pcr
    import solve_nivp.solvers.block_system as new_block_system
    import solve_nivp.solvers.nonlinear_solvers as new_nonlinear_solvers
    import solve_nivp.solvers.pcr as new_pcr
    from solve_nivp.solvers import (
        BlockStructuredSystem,
        ImplicitEquationSolver,
        SchurComplementSolver,
        pcr_solve,
    )

    assert old_nonlinear_solvers is new_nonlinear_solvers
    assert old_block_system is new_block_system
    assert old_pcr is new_pcr
    assert ImplicitEquationSolver is new_nonlinear_solvers.ImplicitEquationSolver
    assert SchurComplementSolver is new_block_system.SchurComplementSolver
    assert BlockStructuredSystem is new_block_system.BlockStructuredSystem
    assert pcr_solve is new_pcr.pcr_solve
