from __future__ import annotations

import importlib.util
from pathlib import Path

import sympy as sp
from funcprops import (
    Analyticity,
    PropertyDecision,
    analytic,
    domain,
    singularities,
)

from asymptotic._property_support import (
    analytic_at_decision,
    branch_safe_substitution_decision,
)


def test_asymptotic_uses_funcprops_as_the_property_backend():
    package_root = Path(__file__).resolve().parents[1] / "src" / "asymptotic"
    assert not (package_root / "function_properties").exists()
    assert importlib.util.find_spec("asymptotic.function_properties") is None


def test_log_properties_come_from_funcprops():
    x = sp.symbols("x", real=True)
    natural_domain = domain(sp.log(x), x)
    singularity_locus = singularities(sp.log(x), x, sp.S.Complexes)
    assert sp.simplify(sp.Equivalent(natural_domain, x > 0)) is sp.S.true
    assert singularity_locus is not sp.S.false
    assert analytic(sp.log(x), x, domain=sp.S.Complexes) is Analyticity.ANALYTIC


def test_sqrt_singularities_and_airy_analyticity_come_from_funcprops():
    x = sp.symbols("x", real=True)
    sqrt_singularity_locus = singularities(sp.sqrt(x), x, sp.S.Complexes)
    assert sqrt_singularity_locus is not sp.S.false
    assert analytic(sp.airyai(x), x, domain=sp.S.Complexes) is Analyticity.ANALYTIC


def test_asymptotic_local_analyticity_adapter_is_auditable():
    z = sp.symbols("z")
    at_zero = analytic_at_decision(sp.log(z), z, 0)
    at_one = analytic_at_decision(sp.log(z), z, 1)
    assert isinstance(at_zero, PropertyDecision)
    assert at_zero.verdict is False
    assert at_zero.provenance and at_zero.provenance[0].source == "funcprops"
    assert at_one.verdict is True


def test_branch_safety_adapter_delegates_to_funcprops_geometry():
    z = sp.symbols("z")
    assert branch_safe_substitution_decision(sp.log(z), z, 1).verdict is True
    assert branch_safe_substitution_decision(sp.log(z), z, 0).verdict is False
