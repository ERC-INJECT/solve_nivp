from pathlib import Path

import pytest

from solve_nivp import _selftest


def test_selftest_uses_top_level_repository_tests(monkeypatch):
    captured = {}

    def fake_pytest_main(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(pytest, "main", fake_pytest_main)

    assert _selftest.main() == 0
    assert Path(captured["args"][-1]).resolve() == Path(__file__).resolve().parent
