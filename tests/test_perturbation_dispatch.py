import sympy as sp

from asymptotic.perturbation_dispatch import (
    PerturbationCertification,
    PerturbationExpansionResult,
    certify_perturbation,
    perturbation_diagnostics,
    perturbation_expansion,
    resume_perturbation,
)
from asymptotic.remainder import Remainder


def test_auto_dispatch_selects_unique_order_ivp():
    eps = sp.Symbol("eps", positive=True)
    x = sp.Symbol("x", real=True)
    y = sp.Function("y")

    result = perturbation_expansion(
        sp.diff(y(x), x) + y(x) + eps * y(x) ** 2,
        y(x),
        eps,
        order=1,
        conditions=sp.Eq(y(0), 1),
    )

    assert isinstance(result, PerturbationExpansionResult)
    assert result.selected_method == "regular"
    assert result.complete
    assert result.ambiguity is False
    assert result.result.branches[0].verified


def test_auto_dispatch_reports_lindstedt_for_duffing():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")
    equation = sp.diff(y(t), t, 2) + y(t) + eps * y(t) ** 3

    diagnostics = perturbation_diagnostics(equation, y(t), eps)
    assert tuple(item.method for item in diagnostics.preferred) == (
        "lindstedt",
        "multiple-scales",
    )

    result = perturbation_expansion(equation, y(t), eps, order=1)
    assert result.result is None
    assert result.selected_method is None
    assert result.ambiguity is True
    assert "lindstedt" in result.limitation
    assert "multiple-scales" in result.limitation


def test_auto_dispatch_prefers_matched_loses_order():
    eps = sp.Symbol("eps", positive=True)
    x = sp.Symbol("x", real=True)
    y = sp.Function("y")

    result = perturbation_expansion(
        eps * sp.diff(y(x), x, 2) + sp.diff(y(x), x),
        y(x),
        eps,
        order=0,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
    )

    assert result.selected_method == "matched"
    assert result.complete
    assert sp.simplify(result.result.composites[0] - (1 - sp.exp(-x / eps))) == 0


def test_explicit_method_resolves_oscillator_ambiguity():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")
    equation = sp.diff(y(t), t, 2) + y(t) + eps * y(t) ** 3

    result = perturbation_expansion(
        equation,
        y(t),
        eps,
        method="lindstedt",
        order=1,
        conditions=(sp.Eq(y(0), 1), sp.Eq(sp.diff(y(t), t).subs(t, 0), 0)),
    )

    assert len(result.diagnostics.preferred) == 2
    assert result.selected_method == "lindstedt"
    assert result.complete
    assert sp.expand(result.result.frequency).coeff(eps, 1) == sp.Rational(3, 8)


def test_manual_replacement_discards_stale_recomputes_branch():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    solved = perturbation_expansion(
        u**2 - 1 - eps,
        u,
        eps,
        method="regular",
        order=2,
    )
    positive_index = next(
        index
        for index, branch in enumerate(solved.result.branches)
        if branch.approximation[0].subs(eps, 0) == 1
    )
    hierarchy = solved.result.branches[positive_index].hierarchy
    u0 = hierarchy.order(0).unknowns[0]

    resumed = resume_perturbation(
        solved.result,
        0,
        {u0: -1},
        branch=positive_index,
    )

    assert resumed.complete
    assert resumed.approximations == ((-1 - eps / 2 + eps**2 / 8,),)
    assert resumed.branches[0].verified


def test_formal_certificate_is_distinct_from_rigorous_remainder_bound():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    result = perturbation_expansion(
        u - 1 - eps,
        u,
        eps,
        method="regular",
        order=1,
    )

    formal_only = certify_perturbation(result)
    assert isinstance(formal_only, PerturbationCertification)
    assert formal_only.formal.verified is True
    assert formal_only.has_rigorous_bound is False

    remainder = Remainder.big_o(
        eps**2,
        eps,
        0,
        source="external perturbation error theorem",
    )
    certified = certify_perturbation(result, rigorous_remainder=remainder)
    assert certified.formal.verified is True
    assert certified.has_rigorous_bound is True
    assert certified.rigorous_remainder is remainder


def test_unknown_remainder_cannot_be_upgraded_to_rigorous_bound():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    result = perturbation_expansion(u - 1, u, eps, method="regular", order=0)
    unknown = Remainder.unknown(eps, 0)

    try:
        certify_perturbation(result, rigorous_remainder=unknown)
    except ValueError as exc:
        assert "not a rigorous" in str(exc)
    else:
        raise AssertionError("unknown remainder was accepted as rigorous")


def test_lindstedt_replacement_rederives_conditions():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")
    result = perturbation_expansion(
        sp.diff(y(t), t, 2) + y(t) + eps * y(t) ** 3,
        y(t),
        eps,
        method="lindstedt",
        order=2,
        conditions=(sp.Eq(y(0), 1), sp.Eq(sp.diff(y(t), t).subs(t, 0), 0)),
    )
    lindstedt = result.result
    first_solution = lindstedt.hierarchy.order(1).solution_dict()

    resumed = resume_perturbation(lindstedt, 1, first_solution)

    assert resumed.complete
    assert resumed.verified
    assert resumed.hierarchy.order(2).solvability_conditions
    assert sp.expand(resumed.frequency).coeff(eps, 2) == -sp.Rational(21, 256)


def test_multiple_scales_certificate_replays_flag_only():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")
    result = perturbation_expansion(
        sp.diff(y(t), t, 2) + y(t) + eps * y(t) ** 3,
        y(t),
        eps,
        method="multiple-scales",
        order=1,
    )

    certificate = certify_perturbation(result)

    assert certificate.formal.verified is True
    assert certificate.formal.equation_residuals
    assert all(value == 0 for value in certificate.formal.equation_residuals)


def test_formal_certificate_exposes_shared_evidence_model():
    from asymptotic import EvidenceStatus

    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    result = perturbation_expansion(u - 1 - eps, u, eps, method="regular", order=1)
    certificate = certify_perturbation(result)
    assert certificate.evidence.status is EvidenceStatus.FORMAL
    assert certificate.evidence.obligations == ("no certified asymptotic remainder",)
