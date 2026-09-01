import sympy as sp

from asymptotic import (
    AsymptoticRemainder,
    RemainderKind,
    multivariate_implicit_asymptotics,
    transseries_from_expression,
)
from asymptotic.remainder_theorems import (
    certify_differentiation_remainder,
    certify_inverse_remainder,
)


def test_differentiation_remainder_upgrades_by_exact_replay():
    x = sp.symbols("x", positive=True)
    r = AsymptoticRemainder.big_o(x**-2, x, sp.oo, exact_expression=x**-3)
    cert = certify_differentiation_remainder(r)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.LITTLE_O
    assert sp.simplify(cert.conclusion.exact_expression + 3 / x**4) == 0


def test_nested_exp_log_remainder_theorems_propagate_certified_error():
    x = sp.symbols("x", positive=True)
    source = transseries_from_expression(
        1 / x,
        x,
        point=sp.oo,
        remainder=AsymptoticRemainder.big_o(x**-2, x, sp.oo, exact_expression=x**-3),
    )
    exp_result = source.exp()
    assert exp_result.remainder.kind in {RemainderKind.BIG_O, RemainderKind.LITTLE_O}
    assert exp_result.metadata["remainder_certificates"][-1].certified

    source2 = transseries_from_expression(
        x,
        x,
        point=sp.oo,
        remainder=AsymptoticRemainder.big_o(1, x, sp.oo, exact_expression=1 / x),
    )
    log_result = source2.log()
    assert log_result.remainder.kind is RemainderKind.LITTLE_O
    assert log_result.metadata["remainder_certificates"][-1].certified


def test_inverse_remainder_uses_nondegenerate_mean_value_theorem():
    x, y = sp.symbols("x y", positive=True)
    cert = certify_inverse_remainder(x + 1 / x, x, y, y - 1 / y)
    assert cert.certified
    assert cert.conclusion.kind is RemainderKind.BIG_O
    assert all(h.verdict is True for h in cert.hypotheses)


def test_multivariate_implicit_discovers_joint_dependent_weights():
    u, v = sp.symbols("u v", positive=True)
    y, z = sp.symbols("y z")
    regimes = multivariate_implicit_asymptotics(
        (y**2 - u, z - y - v),
        (y, z),
        (u, v),
        terms=3,
        stratify_parameters=False,
    )
    assert regimes
    # There is a chamber where u and v have equal representative weights and
    # both dependent variables acquire rho = w_u/2.
    chamber = next(r for r in regimes if r.cone.representative[0] == r.cone.representative[1])
    assert sp.simplify(chamber.dependent_weights[0] - chamber.dependent_weights[1]) == 0
    assert chamber.branches
    assert any(branch.complete for branch in chamber.branches)
    for branch in chamber.branches:
        assert len(branch.series) == 2
        assert len(branch.residuals) == 2


def test_multivariate_implicit_parameter_stratifies_support_changes():
    a = sp.symbols("a")
    u, v = sp.symbols("u v", positive=True)
    y, z = sp.symbols("y z")
    result = multivariate_implicit_asymptotics(
        (y**2 - a * u, z - y - v),
        (y, z),
        (u, v),
        terms=2,
    )
    assert hasattr(result, "strata")
    assert any(sp.simplify(s.condition.subs(a, 0)) is sp.S.true for s in result.strata)
    assert any(sp.simplify(s.condition.subs(a, 1)) is sp.S.true for s in result.strata)
