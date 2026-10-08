"""Certified local semialgebraic domain components for cluster geometry."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.core.relational import Relational

from ._symbolic_policy import bounded_limit
from .coverage import CoverageCertificate


@dataclass(frozen=True)
class LocalDomainComponent:
    condition: sp.Expr
    punctured_condition: sp.Expr
    accumulates: bool
    provider: str = "semialgebraic_local_component"


@dataclass(frozen=True)
class LocalDomainGeometry:
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    domain: sp.Expr
    components: tuple[LocalDomainComponent, ...]
    coverage: CoverageCertificate

    @property
    def certified(self):
        return self.coverage.certified


def _target_equalities(variables, target):
    return sp.And(*(sp.Eq(v, a) for v, a in zip(variables, target, strict=True)))


def _relaxed_closure_formula(expr):
    if isinstance(expr, sp.StrictGreaterThan):
        return sp.Ge(expr.lhs, expr.rhs)
    if isinstance(expr, sp.StrictLessThan):
        return sp.Le(expr.lhs, expr.rhs)
    if isinstance(expr, sp.Unequality):
        return sp.S.true
    if isinstance(expr, sp.And):
        return sp.And(*(_relaxed_closure_formula(a) for a in expr.args))
    if isinstance(expr, sp.Or):
        return sp.Or(*(_relaxed_closure_formula(a) for a in expr.args))
    return expr


def _accumulates_at_target(component, variables, target):
    """Exact first-order accumulation test using semialgebraic QE."""
    distance2 = sp.Add(*((v - a) ** 2 for v, a in zip(variables, target, strict=True)))
    punctured_component = sp.And(component, distance2 > 0)
    try:
        from semialg import is_empty

        empty = is_empty(punctured_component, variables)
        if empty is True:
            return False
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        pass
    eps = sp.Dummy("_eps", positive=True)
    matrix = sp.Implies(
        eps > 0,
        sp.And(component, distance2 > 0, distance2 < eps**2),
    )
    # forall eps > 0, exists x in component within eps of target
    try:
        from semialg import quantifier_eliminate

        prefix = [("forall", eps)] + [("exists", v) for v in variables]
        result = quantifier_eliminate(matrix, prefix)
        if result is sp.S.true:
            return True
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        pass
    relaxed = _relaxed_closure_formula(component)
    at_target = sp.simplify(relaxed.subs(dict(zip(variables, target, strict=True))))
    if at_target is sp.S.true:
        return True
    if at_target is sp.S.false:
        return False
    return None


def local_domain_accumulates(domain, variables, target):
    """Return True/False/None for punctured accumulation at ``target``.

    The test is relative to the supplied domain and ignores membership of the
    target itself.  ``None`` means the semialgebraic accumulation question was
    not certified.
    """
    domain = sp.sympify(domain)
    variables = tuple(variables)
    target = tuple(map(sp.sympify, target))
    return _accumulates_at_target(domain, variables, target)


def local_domain_components(domain, variables, target):
    """Return exact connected components that accumulate at the target.

    Components are computed for the punctured semialgebraic domain.  This
    handles wedges, cusps, algebraic boundaries and deleted varieties without
    encoding their shapes in the limit algorithms.
    """
    domain = sp.sympify(domain)
    variables = tuple(variables)
    target = tuple(map(sp.sympify, target))
    if domain is sp.S.true:
        comp = LocalDomainComponent(
            sp.S.true,
            sp.Not(_target_equalities(variables, target)),
            True,
            "full_neighborhood",
        )
        return LocalDomainGeometry(
            variables,
            target,
            domain,
            (comp,),
            CoverageCertificate.complete(
                "full_domain", "full punctured neighbourhood is connected", ("full",)
            ),
        )
    punctured = sp.And(domain, sp.Not(_target_equalities(variables, target)))
    try:
        from semialg import connected_components

        raw = tuple(connected_components(punctured, variables))
    except (ImportError, AttributeError, NotImplementedError, TypeError, ValueError):
        return LocalDomainGeometry(
            variables,
            target,
            domain,
            (),
            CoverageCertificate.unknown(
                "domain_components_unknown",
                "semialgebraic connected components were not certified",
                ("local_components",),
            ),
        )
    comps = []
    for c in raw:
        accum = _accumulates_at_target(c, variables, target)
        if accum is True:
            comps.append(LocalDomainComponent(c, c, True))
        elif accum is None:
            return LocalDomainGeometry(
                variables,
                target,
                domain,
                tuple(comps),
                CoverageCertificate.unknown(
                    "domain_accumulation_unknown",
                    "component accumulation at target was not certified",
                    ("target_accumulation",),
                ),
            )
    return LocalDomainGeometry(
        variables,
        target,
        domain,
        tuple(comps),
        CoverageCertificate.complete(
            "semialgebraic_local_components",
            "all accumulating connected components of the punctured local domain are represented",
            tuple(c.condition for c in comps),
        ),
    )


@dataclass(frozen=True)
class DomainRelativeClusterSet:
    expression: sp.Expr
    geometry: LocalDomainGeometry
    component_results: tuple[tuple[LocalDomainComponent, object], ...]

    @property
    def certified(self):
        return self.geometry.certified and all(
            getattr(r, "certified", False) for _, r in self.component_results
        )

    @property
    def cluster_set(self):
        if not self.certified:
            return None
        sets = [r.cluster_set for _, r in self.component_results]
        return sp.Union(*sets) if sets else sp.S.EmptySet


def domain_relative_cluster_set(expr, variables, target, *, domain, evaluator=None):
    """Evaluate cluster geometry separately on every accumulating domain component."""
    geometry = local_domain_components(domain, variables, target)
    if not geometry.certified:
        return DomainRelativeClusterSet(sp.sympify(expr), geometry, ())
    if evaluator is None:
        from .multivariate_limits_advanced import newton_fan_cluster_set

        evaluator = newton_fan_cluster_set
    results = []
    for component in geometry.components:
        try:
            result = evaluator(
                expr, tuple(variables), tuple(target), domain=component.condition
            )
        except TypeError:
            # Engines not yet domain-aware cannot certify a relative component.
            return DomainRelativeClusterSet(
                sp.sympify(expr),
                replace_coverage(
                    geometry, False, "cluster evaluator is not domain-aware"
                ),
                tuple(results),
            )
        results.append((component, result))
    return DomainRelativeClusterSet(sp.sympify(expr), geometry, tuple(results))


def replace_coverage(geometry, certified, statement):
    return LocalDomainGeometry(
        geometry.variables,
        geometry.target,
        geometry.domain,
        geometry.components,
        CoverageCertificate.complete(
            "domain_relative_cluster",
            statement,
            tuple(c.condition for c in geometry.components),
        )
        if certified
        else CoverageCertificate.partial(
            "domain_relative_cluster",
            statement,
            (),
            tuple(c.condition for c in geometry.components),
        ),
    )


def transform_domain_to_chart(domain, chart):
    """Pull a semialgebraic domain onto a blow-up chart's exceptional divisor.

    Polynomial relational atoms are divided by their lowest radial power before
    setting r=0. This preserves wedges, cusps and algebraic boundary sectors.
    """
    domain = sp.sympify(domain)
    r = chart.radial_variable

    def atom(rel):
        if not isinstance(rel, Relational):
            return rel.subs(chart.substitution, simultaneous=True)
        lhs = sp.expand((rel.lhs - rel.rhs).subs(chart.substitution, simultaneous=True))
        try:
            poly = sp.Poly(lhs, r)
            powers = [mon[0] for mon, c in poly.terms() if c != 0]
            if not powers:
                lead = sp.S.Zero
            else:
                k = min(powers)
                lead = sp.simplify(
                    bounded_limit(lhs / r**k, r, 0, direction="+", allow_general=True)
                )
        except (sp.PolynomialError, ValueError, TypeError, NotImplementedError):
            lead = sp.simplify(
                bounded_limit(lhs, r, 0, direction="+", allow_general=True)
            )
        op = rel.rel_op
        return {
            "<": sp.Lt,
            "<=": sp.Le,
            ">": sp.Gt,
            ">=": sp.Ge,
            "==": sp.Eq,
            "!=": sp.Ne,
        }[op](lead, 0)

    def walk(e):
        if isinstance(e, (sp.And, sp.Or)):
            return e.func(*(walk(a) for a in e.args))
        if isinstance(e, sp.Not):
            return sp.Not(walk(e.args[0]))
        return atom(e)

    return sp.simplify(walk(domain))


__all__ = [
    "DomainRelativeClusterSet",
    "LocalDomainComponent",
    "LocalDomainGeometry",
    "domain_relative_cluster_set",
    "local_domain_accumulates",
    "local_domain_components",
    "transform_domain_to_chart",
]
