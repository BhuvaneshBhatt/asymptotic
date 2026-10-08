from pathlib import Path
from unittest.mock import patch

import sympy as sp

from asymptotic.limits import LimitStatus
from asymptotic.multivariate_certificates import (
    extremal_limit,
    parameter_stratified_limit,
)
from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    ClusterSetResult,
    multivariate_liminf,
)
from asymptotic.stratification import AsymptoticStratification


def test_scalar_liminf_api_preserved_retains_provenance():
    x, y = sp.symbols("x y", real=True)
    fake = ClusterSetResult(
        x + y,
        (x, y),
        (0, 0),
        sp.FiniteSet(0),
        sp.S.Zero,
        sp.S.Zero,
        AdvancedLimitStatus.UNKNOWN,
        "test_unknown",
        "value present without proof",
    )
    with patch(
        "asymptotic.multivariate_limits_advanced.multivariate_cluster_set",
        lambda *args, **kwargs: fake,
    ):
        assert multivariate_liminf(x + y, (x, y), (0, 0)) == 0
        structured = multivariate_liminf(x + y, (x, y), (0, 0), return_result=True)
        assert structured.value == 0 and not structured.certified
        certificate = extremal_limit(x + y, (x, y), (0, 0), kind="min")
        assert not certificate.certified
        assert certificate.data[0].provider == "test_unknown"


def test_parameter_limit_unknown_leaf():
    x, y, a = sp.symbols("x y a", real=True)
    strat = parameter_stratified_limit(
        (x + a * y) / (x**2 + y**2), (x, y), (0, 0), (a,)
    )
    assert isinstance(strat, AsymptoticStratification)
    assert strat.exhaustive
    assert len(strat.strata) == 1
    assert strat.strata[0].result.status is LimitStatus.UNKNOWN


def test_parity_facade_parameter_contract():
    root = Path(__file__).parents[1] / "src" / "asymptotic"
    facade = (root / "multivariate_certificates.py").read_text()
    parameters = (root / "_multivariate_parameters.py").read_text()
    assert "_old_lojasiewicz" not in facade
    assert "_old_lagrange" not in facade
    assert "return ((" not in parameters
    assert "AsymptoticStratification" in parameters
