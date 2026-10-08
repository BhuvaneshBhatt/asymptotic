"""Table-driven public mathematical contract for simultaneous limits."""

from dataclasses import dataclass

import pytest
import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y = sp.symbols("x y", real=True)


@dataclass(frozen=True)
class LimitCase:
    family: str
    name: str
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    status: LimitStatus
    value: sp.Expr | None = None
    domain: sp.Expr = sp.S.true
    assumptions: sp.Expr = sp.S.true
    min_evidence: int = 1


CASES = (
    LimitCase(
        "uniform-bound",
        "uniform_bound",
        x * y / sp.sqrt(x**2 + y**2),
        (x, y),
        (0, 0),
        LimitStatus.PROVED,
        sp.S.Zero,
    ),
    LimitCase(
        "path-conflict",
        "path_dependent",
        x * y / (x**2 + y**2),
        (x, y),
        (0, 0),
        LimitStatus.DOES_NOT_EXIST,
        min_evidence=2,
    ),
    LimitCase(
        "radial-composition",
        "radial_composition",
        (x**2 + y**2) / (1 + x**2 + y**2),
        (x, y),
        (0, 0),
        LimitStatus.PROVED,
        sp.S.Zero,
    ),
    LimitCase(
        "cylindrical-reduction",
        "cylindrical",
        sp.atan(1 / sp.Abs(x)),
        (x, y),
        (0, 0),
        LimitStatus.PROVED,
        sp.pi / 2,
    ),
    LimitCase(
        "relative-domain",
        "restricted_identity",
        x / y,
        (x, y),
        (0, 0),
        LimitStatus.PROVED,
        sp.S.One,
        domain=sp.Eq(y, x),
    ),
    LimitCase(
        "proof-boundary",
        "unsupported_function_is_unknown",
        sp.Function("f")(x, y),
        (x, y),
        (0, 0),
        LimitStatus.UNKNOWN,
        None,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.family}:{case.name}")
def test_public_limit_contract(case):
    result = limit(
        case.expression,
        case.variables,
        case.target,
        domain=case.domain,
        assumptions=case.assumptions,
        return_result=True,
    )
    assert result.status is case.status
    assert len(result.evidence) >= case.min_evidence
    if case.status is LimitStatus.PROVED:
        assert sp.simplify(result.value - case.value) == 0
        assert result.exists is True
    else:
        # None is intentional for both proved nonexistence and proof-bounded UNKNOWN.
        assert result.value is None
        assert result.exists is (
            False if case.status is LimitStatus.DOES_NOT_EXIST else None
        )
