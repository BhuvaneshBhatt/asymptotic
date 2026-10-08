"""Fast end-to-end checks for the primary installed-package workflows."""

from __future__ import annotations

import sympy as sp

import asymptotic
from asymptotic import (
    RemainderKind,
    equivalent,
    sum,
)
from asymptotic.transseries import (
    transseries_from_expression,
)


def test_primary_package_smoke():
    """Exercise representative root-namespace workflows without expensive search."""
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", integer=True)

    assert asymptotic.__version__
    assert equivalent(x + 1, x, x, sp.oo) is True

    expansion = transseries_from_expression(
        1 / x + 2 / x**2 + 3 / x**3,
        x,
        point=sp.oo,
        complete=True,
    )
    truncation = expansion.truncation(1)
    assert sp.simplify(truncation.prefix - 1 / x) == 0
    assert truncation.remainder.kind is RemainderKind.BIG_O
    assert truncation.remainder.check() is True

    summation = sum(
        (k / x) ** 2,
        k,
        1,
        x,
        parameter=x,
        terms=2,
        method="riemann",
        return_result=True,
    )
    assert summation.method == "riemann-sum"
    assert summation.status == "FORMAL"
    assert sp.simplify(summation.expression - x / 3) == 0
