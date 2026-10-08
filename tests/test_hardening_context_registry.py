import sympy as sp

from asymptotic.context import AsymptoticContext
from asymptotic.parameter_stratification import critical_threshold_partition
from asymptotic.theorem_registry import CertificationStrength, Theorem, TheoremRegistry


def test_context_caches_normalization_entailment_and_applicability():
    x = sp.symbols("x", positive=True)
    ctx = AsymptoticContext(x, assumptions=sp.Gt(x, 0))
    ctx.normalize((x * x) / x)
    ctx.normalize((x * x) / x)
    assert ctx.entails(sp.Gt(x, 0)) is True
    assert ctx.entails(sp.Gt(x, 0)) is True
    assert ctx.cached_applicability(("t", 1), lambda: True) is True
    assert (
        ctx.cached_applicability(
            ("t", 1), lambda: (_ for _ in ()).throw(AssertionError())
        )
        is True
    )
    m = ctx.cache_metrics()
    assert (
        m["hits"]["normalize"] >= 1
        and m["hits"]["entails"] == 1
        and m["hits"]["applicability"] == 1
    )


def test_theorem_registry_selects_strongest_proved_applicable_theorem():
    x = sp.symbols("x", positive=True)
    ctx = AsymptoticContext(x, assumptions=sp.Gt(x, 0))
    r = TheoremRegistry()
    r.register(
        Theorem(
            "formal",
            "sum",
            lambda p: True,
            lambda p: (),
            lambda p: 1,
            CertificationStrength.FORMAL,
        )
    )
    r.register(
        Theorem(
            "cert",
            "sum",
            lambda p: True,
            lambda p: (sp.Gt(x, 0),),
            lambda p: 2,
            CertificationStrength.CERTIFIED,
        )
    )
    assert r.select(x, context=ctx).theorem.name == "cert"


def test_threshold_partition_is_exhaustive_and_budgeted():
    a, b = sp.symbols("a b", real=True)
    p = critical_threshold_partition(a, b, branch_budget=3)
    assert p is not None and p.conditions == (a < b, sp.Eq(a, b), a > b)
    assert critical_threshold_partition(a, b, branch_budget=2) is None
