"""Permanent adversarial limit corpus derived from public SymPy issues.

These are mathematical contracts, not tests of SymPy's current behavior.  The
issue metadata records why each case entered the corpus; upstream fixes must not
change asymptotic's proof/status contract accidentally.
"""

from dataclasses import dataclass

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limits import LimitStatus

x = sp.symbols("x", real=True)
a, b = sp.symbols("a b", positive=True)


@dataclass(frozen=True)
class GitHubLimitCase:
    issue: int
    name: str
    expression: sp.Expr
    target: sp.Expr
    status: LimitStatus
    value: sp.Expr | None
    failure_mode: str
    evidence: str | None = None


CASES = (
    GitHubLimitCase(
        30581,
        "lambertw_exponential_infinity",
        sp.exp(sp.LambertW(x)),
        sp.oo,
        LimitStatus.PROVED,
        sp.oo,
        "upstream wrong result (1)",
        "principal_lambert_exact_positive_chart",
    ),
    GitHubLimitCase(
        30580,
        "nested_log_power",
        (1 / (x * sp.log(sp.log(x)))) ** (sp.log(sp.log(x)) / sp.log(x)),
        sp.oo,
        LimitStatus.PROVED,
        sp.S.Zero,
        "upstream RecursionError",
        "positive_power_log_normalization",
    ),
    GitHubLimitCase(
        30568,
        "min_polynomial",
        sp.Min(x, x**2),
        0,
        LimitStatus.PROVED,
        sp.S.Zero,
        "upstream NotImplementedError for Min/Max",
    ),
    GitHubLimitCase(
        30568,
        "max_sine",
        sp.Max(x, sp.sin(x)),
        0,
        LimitStatus.PROVED,
        sp.S.Zero,
        "upstream NotImplementedError for Min/Max",
    ),
    GitHubLimitCase(
        30568,
        "min_exponentials",
        sp.Min(sp.exp(x), sp.exp(2 * x)),
        sp.oo,
        LimitStatus.PROVED,
        sp.oo,
        "upstream NotImplementedError for Min/Max",
        "eventual_exp_minmax_dominance",
    ),
    GitHubLimitCase(
        30545,
        "sinc_origin",
        sp.sinc(x),
        0,
        LimitStatus.PROVED,
        sp.S.One,
        "upstream NotImplementedError for sinc",
    ),
    GitHubLimitCase(
        22893,
        "parameter_dependent_exponential_rate",
        (a * sp.exp(-a * x) + b * sp.exp(-b * x)) * sp.exp(b * x),
        sp.oo,
        LimitStatus.UNKNOWN,
        None,
        "upstream unconditional oo although the answer depends on sign(b-a)",
        "parameter_exponential_rate_unresolved",
    ),
    GitHubLimitCase(
        27236,
        "piecewise_two_sided_jump",
        sp.Piecewise((1, x < 0), (-1, True)),
        0,
        LimitStatus.DOES_NOT_EXIST,
        None,
        "upstream two-sided limit selected one branch",
        "piecewise_conflicting_germs",
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"sympy-{c.issue}:{c.name}")
def test_sympy_github_limit_regression(case):
    result = limit(case.expression, x, case.target, return_result=True)
    assert result.status is case.status, case.failure_mode
    if case.status is LimitStatus.PROVED:
        if case.value in (sp.oo, -sp.oo):
            assert result.value is case.value
        else:
            assert sp.simplify(result.value - case.value) == 0
    else:
        assert result.value is None
    if case.evidence is not None:
        assert case.evidence in {item.method for item in result.evidence}


@pytest.mark.parametrize(
    ("assumptions", "expected"),
    ((a > b, b), (a < b, sp.oo), (sp.Eq(a, b), a + b)),
)
def test_sympy_22893_parameter_strata_is_known(assumptions, expected):
    expression = (a * sp.exp(-a * x) + b * sp.exp(-b * x)) * sp.exp(b * x)
    result = limit(expression, x, sp.oo, assumptions=assumptions, return_result=True)
    assert result.status is LimitStatus.PROVED
    if expected is sp.oo:
        assert result.value is sp.oo
    else:
        assert sp.simplify(result.value - expected) == 0
    assert "parameter_exponential_rate" in {item.method for item in result.evidence}
