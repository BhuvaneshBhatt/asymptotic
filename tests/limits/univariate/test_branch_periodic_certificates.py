"""Focused checks for attained witnesses and principal-branch recognition."""

import pytest
import sympy as s

from asymptotic.attained_periodic_families import (
    imaginary_root_phase_cancellation_certificate,
    periodic_composition_certificate,
)
from asymptotic.limit_models import LimitStatus
from asymptotic.local_nonexistence_families import local_analytic_pole_certificate
from asymptotic.local_radical_branch_germs import (
    reciprocal_root_argument_certificate,
    signed_radical_chart_certificate,
)
from asymptotic.principal_branch_logs import principal_unit_phase_log_certificate


@pytest.mark.parametrize(
    "make,point",
    [
        (lambda x: s.sin(x) / x**2, 0),
        (lambda x: s.cos(x) / x, 0),
        (lambda x: (s.exp(x) - 1) / x**2, 0),
        (lambda x: (1 - x) / (1 - s.sin(s.pi * x / 2)), 1),
        (lambda x: (x - 3) / (s.sqrt(x) - 3), 9),
    ],
)
def test_analytic_poles_have_opposite_attained_limits(make, point):
    x = s.Symbol("x", real=True)
    result = local_analytic_pole_certificate(
        make(x), x, s.sympify(point), s.true, s.true
    )
    assert result is not None and result[0] is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result[2]} == {s.oo, -s.oo}
    assert all(e.substitutions for e in result[2])


def test_principal_root_charts_do_not_force_denesting():
    x = s.Symbol("x")
    result = signed_radical_chart_certificate(
        x / s.sqrt(x * x), x, s.S.Zero, s.true, s.true
    )
    assert result is not None and result[0] is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result[2]} == {s.S.One, -s.S.One}
    assert (
        signed_radical_chart_certificate(s.sqrt(x), x, s.S.Zero, s.Eq(x, 0), s.true)
        is None
    )


def test_argument_cut_and_nonzero_imaginary_increment():
    x = s.Symbol("x")
    cut = reciprocal_root_argument_certificate(
        s.arg(-s.sqrt(1 / x) - s.I), x, s.S.Zero, s.true, s.true
    )
    assert cut[0] is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in cut[2]} == {-s.pi, -s.pi / 2}
    increment = reciprocal_root_argument_certificate(
        s.arg(s.I * (s.sqrt(x * x + 3) - s.sqrt(3))), x, s.S.Zero, s.true, s.true
    )
    assert increment[0] is LimitStatus.PROVED and increment[1] == s.pi / 2


@pytest.mark.parametrize(
    "make",
    [
        lambda x: s.sinh(s.sin(x + 1)),
        lambda x: s.Min(3 * (x + 1) / (x - 1), s.cos(s.sin(x) ** 2)),
        lambda x: s.Max(2 * (x + 1) / (3 * (x - 1)), s.cos(s.sin(x) ** 2)),
    ],
)
def test_periodic_outer_functions_have_distinct_attained_values(make):
    x = s.Symbol("x", real=True)
    result = periodic_composition_certificate(make(x), x, s.oo, s.true, s.true)
    assert result is not None and result[0] is LimitStatus.DOES_NOT_EXIST
    first, second = result[2]
    assert s.simplify(first.value - second.value).is_zero is False
    for evidence in result[2]:
        sequence = evidence.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        assert s.limit(sequence, n, s.oo) == s.oo
        assert (
            abs(
                complex(
                    (make(x).subs(x, sequence.subs(n, 10000)) - evidence.value).evalf(
                        25
                    )
                )
            )
            < 0.01
        )


def test_periodic_matcher_declines_incompatible_phases_and_domains():
    x = s.Symbol("x", real=True)
    assert (
        periodic_composition_certificate(
            s.sin(x) + s.cos(2 * x), x, s.oo, s.true, s.true
        )
        is None
    )
    assert periodic_composition_certificate(s.sin(x), x, s.oo, x > 0, s.true) is None


def test_imaginary_root_common_phase_cancels():
    x = s.Symbol("x", positive=True)
    expression = s.sqrt(1 - x * x) * s.sinh(s.sqrt(1 - x * x)) / (x + 1) + s.sin(x)
    result = imaginary_root_phase_cancellation_certificate(
        expression, x, s.oo, s.true, s.true
    )
    assert result is not None and result[0] is LimitStatus.PROVED and result[1] == 0
    assert abs(complex(expression.subs(x, 10000).evalf(25))) < 0.001


def test_principal_phase_is_bounded_without_splitting_the_log():
    x = s.Symbol("x", positive=True)
    ratio = s.log((-1) ** x / x) / s.log(x)
    result = principal_unit_phase_log_certificate(ratio, x, s.oo, s.true, s.true)
    assert result is not None and result[0] is LimitStatus.PROVED and result[1] == -1
    growing = principal_unit_phase_log_certificate(
        s.log(x * x * s.exp(5 * s.I * x)), x, s.oo, s.true, s.true
    )
    assert growing is not None and growing[1] == s.oo
    assert (
        principal_unit_phase_log_certificate(
            s.log(-x * s.exp(s.I * x)), x, s.oo, s.true, s.true
        )
        is None
    )
