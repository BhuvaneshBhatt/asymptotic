"""Keep introductory limit examples executable."""

import runpy
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[3]


def test_multivariate_limits_example():
    namespace = runpy.run_path(ROOT / "examples" / "multivariate_limits.py")
    assert namespace["results"]["cylindrical"].value == sp.pi / 2
