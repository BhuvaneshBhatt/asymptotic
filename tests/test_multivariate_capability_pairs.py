import pytest
import sympy as sp

from asymptotic.complex_cluster_geometry import branched_blowup_atlas
from asymptotic.relative_growth import relative_growth_valuation_fan

x, y, a = sp.symbols("x y a", real=True)
PAIRS = [
    ("removable", x**2 * y**2 / (x**2 + y**2), 0, x * y / (x**2 + y**2), None),
    ("anisotropic", x**4 / (x**4 + y**2), None, x**4 / (x**4 + y**4), None),
    (
        "piecewise",
        sp.Piecewise((x, x >= 0), (-x, True)),
        0,
        sp.Piecewise((1, x >= 0), (-1, True)),
        None,
    ),
    ("newton", x**4 + y**6, 0, x**4 / (x**4 + y**6), None),
    ("parameter", x**2 / (x**2 + y**2) ** a, None, x**2 / (x**2 + y**2), None),
    (
        "vector_joint",
        sp.Tuple(x, y),
        sp.Tuple(0, 0),
        sp.Tuple(x / (sp.Abs(x) + sp.Abs(y)), y / (sp.Abs(x) + sp.Abs(y))),
        None,
    ),
]


@pytest.mark.parametrize("family,valid,valid_expected,nearby,nearby_expected", PAIRS)
def test_permanent_pair_registry(
    family, valid, valid_expected, nearby, nearby_expected
):
    assert family and valid != nearby


def test_branch_cut_pair_valid_and_missing_domain_prerequisite():
    z = x + sp.I * y
    charts, cov = branched_blowup_atlas(
        sp.log(z), (x, y), (0, 0), domain=sp.And(x > 0, y > 0)
    )
    assert cov.certified and charts
    # unsupported non-semialgebraic relative side remains explicit rather than guessed
    charts2, _cov2 = branched_blowup_atlas(
        sp.log(z), (x, y), (0, 0), domain=sp.Contains(x, sp.Integers)
    )
    assert (
        charts2
    )  # structural atlas survives; downstream image certification decides applicability


def test_relative_growth_valid_invalid_prerequisite_pair():
    xp, yp = sp.symbols("xp yp", positive=True)
    good, cg = relative_growth_valuation_fan(
        (xp, yp), sp.Eq(sp.log(yp) / sp.log(xp), sp.Rational(2))
    )
    bad, cb = relative_growth_valuation_fan((xp, yp), sp.Gt(yp, xp))
    assert cg.certified and good
    assert not cb.certified and not bad
