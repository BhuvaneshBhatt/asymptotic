"""Release-only smoke test against an actually installed wheel.

Set ASYMPTOTIC_WHEEL to the wheel path in the release job.  Ordinary source-tree
runs skip this test because building a wheel is outside pytest.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


def test_installed_wheel_smoke(tmp_path):
    configured = os.environ.get("ASYMPTOTIC_WHEEL")
    if not configured:
        pytest.skip("set ASYMPTOTIC_WHEEL in the release-artifact job")
    python = os.environ.get("ASYMPTOTIC_PYTHON", sys.executable)
    wheel = Path(configured).resolve()
    assert wheel.is_file()
    target = tmp_path / "site"
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    expected_version = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["version"]
    script = r"""
from importlib.metadata import version
from pathlib import Path
import sympy as sp
import asymptotic
from asymptotic.limit_models import LimitStatus
assert asymptotic.__version__ == version("asymptotic") == __EXPECTED_VERSION__
assert Path(asymptotic.__file__).resolve().is_relative_to(Path(__TARGET__))
x, y = sp.symbols("x y", real=True)
z = sp.Symbol("z")
assert asymptotic.limit(x*y/sp.sqrt(x*x+y*y), (x,y), (0,0)) == 0
assert asymptotic.one_sided_limit(1/x, x, 0, direction="+") is sp.oo
assert asymptotic.complex_limit(1/(z-sp.I)**2, z, sp.I) is sp.zoo
assert asymptotic.limit(sp.Function("f")(x), x, 0, return_result=True).status is LimitStatus.UNKNOWN
result = asymptotic.limit(x*y/(x*x+y*y), (x,y), (0,0), return_result=True)
assert result.status is LimitStatus.DOES_NOT_EXIST
assert len({item.value for item in result.evidence}) >= 2
for item in result.evidence:
    chart = dict(item.substitutions)
    assert set(chart) == {x, y}
    indices = set().union(*(value.free_symbols for value in chart.values()))
    assert len(indices) == 1
    index = next(iter(indices))
    assert index.is_integer is True and index.is_positive is True
    assert all(sp.limit(value, index, sp.oo) == 0 for value in chart.values())
    denominator = (x*x+y*y).subs(chart, simultaneous=True)
    assert denominator.is_positive is True
    attained = (x*y/(x*x+y*y)).subs(chart, simultaneous=True)
    assert sp.limit(attained, index, sp.oo) == item.value
t = sp.Symbol("t", positive=True)
truncation = asymptotic.multiseries(sp.exp(1/t), t, scale=[1/t], terms=5, return_result=True).as_element().truncation(3)
assert truncation.prefix == 1+1/t+1/(2*t*t)
assert truncation.remainder.check() is True
""".replace("__EXPECTED_VERSION__", repr(expected_version)).replace(
        "__TARGET__", repr(str(target))
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    subprocess.run(
        [
            python,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(target)!r});\n{script}",
        ],
        check=True,
        env=env,
    )
