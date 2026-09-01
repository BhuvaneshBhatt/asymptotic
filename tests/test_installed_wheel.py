"""Release-only smoke test against an actually installed wheel.

Set ASYMPTOTIC_WHEEL to the wheel path in the release job.  Ordinary source-tree
runs skip this test because building a wheel is intentionally outside pytest.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_installed_wheel_smoke(tmp_path):
    configured = os.environ.get("ASYMPTOTIC_WHEEL")
    if not configured:
        pytest.skip("set ASYMPTOTIC_WHEEL in the release-artifact job")
    wheel = Path(configured).resolve()
    assert wheel.is_file()
    target = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    script = r"""
import sympy as sp
import asymptotic
assert asymptotic.__version__ == "0.53.2"
assert not hasattr(asymptotic, "certify_product_remainder")
from asymptotic.remainder_theorems import certify_product_remainder
n=sp.symbols("n", positive=True); mu=sp.symbols("mu", real=True); x=sp.symbols("x", real=True)
r=asymptotic.airy_uniform_saddle_asymptotic(sp.exp(sp.I*n*(x**3/3+mu*x)), x, (-sp.oo,sp.oo), parameter=n, control_parameter=mu)
assert r.status == "FORMAL"
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(target)!r});\n{script}"],
        check=True,
        env=env,
    )
