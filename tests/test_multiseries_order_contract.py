"""Inclusive algebraic-order contract for multiseries (no MultipleIntegrate import)."""

import pytest
import sympy as sp

from asymptotic import multiseries
from asymptotic.multiseries import Multiseries
from asymptotic.order_expansion import OrderedMultiseries

eps = sp.Symbol("eps", positive=True)


def _cutoff(m: int, k: int, *, weighted: bool = False) -> sp.Expr:
    delta = eps / abs(k)
    a = (1 - sp.sqrt(1 - 4 * delta)) / 2
    b = 1 - a
    if weighted:
        return (sp.sqrt(1 - 4 * delta) - (1 - 2 * delta) * sp.log(b / a)) / m
    return (
        sp.log(b) ** 2
        - sp.log(a) ** 2
        - 2 * sp.log(a) * sp.log(b)
        - 4 * sp.polylog(2, a)
        + sp.pi**2 / 3
    ) / (2 * m)


@pytest.mark.parametrize(
    "m,k", [(2, 1), (3, 1), (4, 1), (2, 2), (2, -3), (3, -3), (4, 2)]
)
def test_shared_logarithmic_cutoff_is_complete_through_zero(m, k):
    result = multiseries(_cutoff(m, k), eps, point=0, order=0, return_result=True)
    assert isinstance(result, OrderedMultiseries)
    assert result.is_complete_through(0)
    assert not result.is_complete_through(1)
    assert not result.certified  # formal complete != independently certified
    expected = (
        -(sp.log(eps) ** 2) / (2 * m)
        + sp.log(abs(k)) * sp.log(eps) / m
        - sp.log(abs(k)) ** 2 / (2 * m)
        + sp.pi**2 / (6 * m)
    )
    assert sp.simplify(result.truncate() - expected) == 0
    assert result.remainder is not None
    assert not result.truncate().has(sp.polylog)
    assert (
        sp.simplify(multiseries(_cutoff(m, k), eps, point=0, order=0) - expected) == 0
    )


@pytest.mark.parametrize("m,k", [(2, 1), (3, 1), (2, 2), (3, -3)])
def test_weighted_tangent_regulator_preserved(m, k):
    result = multiseries(
        _cutoff(m, k, weighted=True), eps, point=0, order=0, return_result=True
    )
    assert result.is_complete_through(0)
    expected = (1 + sp.log(eps) - sp.log(abs(k))) / m
    assert sp.simplify(result.truncate() - expected) == 0


@pytest.mark.parametrize("pole", [1, 2, 4])
def test_negative_laurent_orders_and_all_log_coefficients(pole):
    expr = (
        eps ** (-pole)
        + sp.log(eps) / eps
        + 5 * sp.log(eps) ** 2
        + 11
        + eps * sp.log(eps)
    )
    result = multiseries(expr, eps, point=0, order=0, return_result=True)
    assert result.is_complete_through(0)
    assert (
        sp.simplify(
            result.truncate()
            - (eps ** (-pole) + sp.log(eps) / eps + 5 * sp.log(eps) ** 2 + 11)
        )
        == 0
    )
    assert sp.limit(result.remainder, eps, 0, dir="+") == 0


def test_order_is_inclusive_unlike_sympy_series_n():
    f = 1 / eps + 3 + 7 * eps + 5 * eps**2
    assert multiseries(f, eps, point=0, order=-1) == 1 / eps
    assert multiseries(f, eps, point=0, order=0) == 1 / eps + 3
    assert multiseries(f, eps, point=0, order=1) == 1 / eps + 3 + 7 * eps
    assert multiseries(f, eps, point=0, order=2) == f


def test_fractional_powers_above_target_go_to_remainder():
    result = multiseries(
        3 + sp.sqrt(eps) + eps * sp.log(eps), eps, point=0, order=0, return_result=True
    )
    assert result.is_complete_through(0)
    assert result.truncate() == 3
    assert sp.simplify(result.remainder - (sp.sqrt(eps) + eps * sp.log(eps))) == 0


def test_finite_shifted_point_and_infinity():
    x = sp.Symbol("x", real=True)
    assert (
        sp.simplify(
            multiseries(2 + 3 * (x - 2) + (x - 2) ** 2, x, point=2, order=1)
            - (2 + 3 * (x - 2))
        )
        == 0
    )
    assert (
        sp.simplify(
            multiseries(1 + 1 / x + 1 / x**2, x, point=sp.oo, order=1) - (1 + 1 / x)
        )
        == 0
    )


def test_legacy_terms_api_unchanged_and_remains_lazy():
    result = multiseries(sp.exp(eps), eps, point=0, terms=3, return_result=True)
    assert isinstance(result, Multiseries)
    assert sp.simplify(result.truncate(3) - (1 + eps + eps**2 / 2)) == 0


def test_order_rejects_ambiguous_unsupported_modes():
    with pytest.raises(ValueError, match="automatic local scale"):
        multiseries(1 + eps, eps, point=0, order=0, scale=[eps])
    with pytest.raises(ValueError, match="allow_series_fallback"):
        multiseries(1 + eps, eps, point=0, order=0, allow_series_fallback=False)
    with pytest.raises(ValueError, match="term count"):
        multiseries(1 + eps, eps, point=0, order=0, terms=8)
    with pytest.raises(TypeError, match="integer"):
        multiseries(1 + eps, eps, point=0, order=0.5)


def test_unknown_is_not_accepted_as_completed_coefficient():
    from asymptotic.order_expansion import expand_to_order

    result = expand_to_order(sp.sin(1 / eps), eps, point=0, order=0, timeout=1)
    assert not result.complete
    assert result.remainder is None
    with pytest.raises(NotImplementedError):
        result.truncate()


def test_size_budget_fails_closed():
    from asymptotic.order_expansion import expand_to_order

    expr = sp.sin(eps) + sp.cos(eps)
    result = expand_to_order(expr, eps, point=0, order=0, max_ops=1)
    assert result.status == "unknown"
    assert "operation budget" in result.reason
    with pytest.raises(NotImplementedError):
        result.truncate()
