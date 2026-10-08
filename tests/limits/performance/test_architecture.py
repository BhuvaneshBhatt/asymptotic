import sympy as sp

from asymptotic import limit
from asymptotic._multivariate_analytic import analytic_leading_ratio_certificate
from asymptotic.limit_models import LimitStatus
from asymptotic.multivariate_limits_advanced import analytic_jet_radial_vanishing_limit


def test_analytic_ratio_rejects_nonanalytic_radial_denominator():
    x, y = sp.symbols("x y", real=True)
    cert = analytic_leading_ratio_certificate(
        x**2 / sp.sqrt(x**2 + y**2), (x, y), (0, 0)
    )
    assert not cert.certified


def test_cancellation_aware_analytic_jet_over_radial_norm():
    h, k = sp.symbols("h k", real=True)
    expr = (-h + sp.exp(sp.sin(h / (1 + k))) - 1) / sp.sqrt(h**2 + k**2)
    cert = analytic_jet_radial_vanishing_limit(expr, (h, k), (0, 0))
    assert cert.status.name == "CERTIFIED"
    assert cert.value == 0
    result = limit(expr, (h, k), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert result.evidence[0].method == "analytic_jet_radial_order"
