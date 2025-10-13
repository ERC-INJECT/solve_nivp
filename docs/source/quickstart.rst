Quickstart
==========

Install
-------

- Stable: ``pip install Solve_IVP_NS``
- From source (editable): ``pip install -e .[test]``

Minimal example
---------------

.. code-block:: python

    import numpy as np
    from Solve_IVP_NS import solve_ivp_ns

    # y' = -y with identity projection
    rhs = lambda t, y: -y
    t_span = (0.0, 1.0)
    y0 = np.array([1.0])

    t, y, h, fk, info = solve_ivp_ns(
        fun=rhs,
        t_span=t_span,
        y0=y0,
        method='composite',
        projection='identity',
        solver='VI',
    )

    print(t[:5], y[:5])

Options
-------
- Projections: ``'identity'``, ``'sign'``, ``'coulomb'`` (see API docs for arguments like ``y_indices``, ``w_indices``, ``con_force_func``, ``conf_jacobian_mode``)
- Solvers: ``'VI'`` and ``'semismooth_newton'`` (globalization ``'none'`` or ``'linesearch'``)
- Integrators: BackwardEuler, Trapezoidal, ThetaMethod, CompositeMethod, EmbeddedBETR

Numba acceleration
------------------
Some projector internals can use numba if available. To disable, pass ``use_numba=False`` to the relevant projection. To install: ``pip install numba``.
