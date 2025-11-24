def main():
    """
    Run the project's test suite with visible output and a brief environment summary.

    Usage:
        solve_nivp-selftest
    """
    import sys
    import os
    import platform
    from importlib import import_module

    print("\n=== solve_nivp self-test ===")
    try:
        import solve_nivp as sivp
        print(f"Package: solve_nivp {getattr(sivp, '__version__', '0.1.0')} @ {os.path.dirname(sivp.__file__)}")
    except Exception as e:
        print("Could not import solve_nivp:", e)
        return 1

    try:
        import numpy as np  # noqa: F401
        import scipy as sp  # noqa: F401
        print("Python:", platform.python_version(), "| Platform:", platform.platform())
        print("NumPy:", import_module('numpy').__version__, "| SciPy:", import_module('scipy').__version__)
    except Exception as e:
        print("Warning: failed to import NumPy/SciPy:", e)

    try:
        import pytest  # type: ignore
    except Exception:
        print("pytest is required. Install with: pip install -e .[test]", file=sys.stderr)
        return 1

    tests_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
    if not os.path.isdir(tests_path):
        print(f"Tests directory not found: {tests_path}")
        print("If you installed non-editable, clone the repo and run tests from source.")
        return 1

    print(f"Running pytest in: {tests_path}")
    print("(Showing test names and print output; this may be verbose)\n")
    # -v: verbose (show test names), -s: no capture (show prints)
    exit_code = pytest.main(["-v", "-s", tests_path])
    print("\n=== Self-test", "PASSED" if exit_code == 0 else "FAILED", f"(exit code {exit_code}) ===")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
