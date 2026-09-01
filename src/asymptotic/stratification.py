"""Parameter stratification for conditional asymptotic algorithms.

The machinery in this module makes parameter case splits explicit.  A stratum
contains both its logical condition and the provenance/certification supporting
that condition; algorithms may therefore return several mathematically distinct
asymptotic answers instead of silently choosing a generic parameter regime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom

from ._symbolic_policy import bounded_assumption_entails
from .canonical import canonical_equal, canonical_expr, canonical_key
from .function_properties.semantics import (
    PropertyDecision,
    PropertyKnowledge,
    PropertyProvenance,
    entails,
)
from .instrumentation import record_symbolic_event

T = TypeVar("T")


def _polynomial_parameters(
    condition: sp.Expr, parameters: Iterable[sp.Symbol] | None = None
) -> tuple[sp.Symbol, ...]:
    if parameters is not None:
        return tuple(sorted(set(parameters), key=sp.default_sort_key))
    return tuple(sorted(condition.free_symbols, key=sp.default_sort_key))


def _normalized_polynomial(expr: sp.Expr, parameters: tuple[sp.Symbol, ...]) -> sp.Expr | None:
    """Canonical square-free representative of a polynomial zero set.

    Multiplication by a nonzero rational scalar and repeated irreducible factors
    do not change ``p = 0`` or ``p != 0``.  Removing those distinctions gives a
    cheap principal-radical normalization without claiming to compute radicals
    of arbitrary multivariate ideals.
    """

    expr = sp.expand(sp.sympify(expr))
    if not parameters:
        return expr
    try:
        poly = sp.Poly(expr, *parameters, domain=sp.QQ)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if poly.is_zero:
        return sp.S.Zero
    try:
        poly = poly.sqf_part()
    except (sp.PolynomialError, TypeError, ValueError):
        pass
    try:
        poly = poly.monic()
    except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError):
        primitive = poly.primitive()[1]
        poly = primitive
        lc = poly.LC()
        if getattr(lc, "could_extract_minus_sign", lambda: False)():
            poly = -poly
    return sp.expand(poly.as_expr())


def _normalize_relational_atom(atom: sp.Expr, parameters: tuple[sp.Symbol, ...]) -> sp.Expr:
    atom = sp.sympify(atom)
    if isinstance(atom, (sp.Equality, sp.Unequality)):
        polynomial = _normalized_polynomial(atom.lhs - atom.rhs, parameters)
        if polynomial is not None:
            if polynomial == 0:
                return sp.S.true if isinstance(atom, sp.Equality) else sp.S.false
            relation = sp.Eq if isinstance(atom, sp.Equality) else sp.Ne
            return relation(polynomial, 0, evaluate=False)
    if atom.func is sp.Not and len(atom.args) == 1:
        inner = atom.args[0]
        if isinstance(inner, sp.Equality):
            return _normalize_relational_atom(
                sp.Ne(inner.lhs, inner.rhs, evaluate=False), parameters
            )
        if isinstance(inner, sp.Unequality):
            return _normalize_relational_atom(
                sp.Eq(inner.lhs, inner.rhs, evaluate=False), parameters
            )
    return canonical_expr(atom)


def _groebner_equalities(
    equalities: list[sp.Expr], parameters: tuple[sp.Symbol, ...]
) -> tuple[list[sp.Expr], sp.GroebnerBasis | None]:
    """Return a deterministic equality basis and its cheap Gröbner certificate."""

    if not equalities or not parameters:
        return equalities, None
    polynomial_exprs = []
    for equality in equalities:
        normalized = _normalized_polynomial(equality.lhs - equality.rhs, parameters)
        if normalized is None:
            return equalities, None
        polynomial_exprs.append(normalized)
    # Keep Gröbner canonicalization deliberately bounded.  Generated parameter
    # strata are normally tiny; larger systems should remain merely structural.
    if len(polynomial_exprs) > 8 or sum(int(sp.count_ops(p)) for p in polynomial_exprs) > 120:
        return equalities, None
    try:
        basis = sp.groebner(polynomial_exprs, *parameters, order="grevlex", domain=sp.QQ)
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        return equalities, None
    normalized_basis = []
    for polynomial in basis.polys:
        expr = _normalized_polynomial(polynomial.as_expr(), parameters)
        if expr not in (None, 0):
            normalized_basis.append(sp.Eq(expr, 0, evaluate=False))
    normalized_basis.sort(key=lambda item: repr(canonical_key(item)))
    return normalized_basis, basis


def _canonicalize_conjunction(
    atoms: Iterable[sp.Expr], parameters: tuple[sp.Symbol, ...]
) -> sp.Expr:
    """Canonicalize a conjunction using bounded polynomial ideal reduction.

    Polynomial equalities are replaced by a deterministic Gröbner basis and
    polynomial inequalities are reduced modulo that basis. Contradictions and
    duplicate atoms are removed before a stable Boolean expression is rebuilt.
    """

    flat = []
    for atom in atoms:
        atom = sp.sympify(atom)
        if atom.func is sp.And:
            flat.extend(atom.args)
        else:
            flat.append(atom)
    normalized = [_normalize_relational_atom(atom, parameters) for atom in flat]
    if any(atom is sp.S.false for atom in normalized):
        return sp.S.false
    normalized = [atom for atom in normalized if atom is not sp.S.true]

    polynomial_equalities = [
        a
        for a in normalized
        if isinstance(a, sp.Equality)
        and _normalized_polynomial(a.lhs - a.rhs, parameters) is not None
    ]
    other = [a for a in normalized if a not in polynomial_equalities]
    equality_basis, groebner = _groebner_equalities(polynomial_equalities, parameters)

    reduced_other = []
    for atom in other:
        if isinstance(atom, sp.Unequality) and groebner is not None:
            polynomial = _normalized_polynomial(atom.lhs - atom.rhs, parameters)
            if polynomial is not None:
                try:
                    _quotients, remainder = groebner.reduce(polynomial)
                except (sp.PolynomialError, TypeError, ValueError):
                    remainder = polynomial
                remainder = _normalized_polynomial(remainder, parameters)
                if remainder == 0:
                    return sp.S.false
                if remainder is not None:
                    atom = sp.Ne(remainder, 0, evaluate=False)
        reduced_other.append(_normalize_relational_atom(atom, parameters))

    combined = equality_basis + reduced_other
    by_key = {}
    for atom in combined:
        key = canonical_key(atom)
        by_key[key] = atom
    atoms_out = sorted(by_key.values(), key=lambda item: repr(canonical_key(item)))

    equality_keys = {canonical_key(a.lhs - a.rhs) for a in atoms_out if isinstance(a, sp.Equality)}
    inequality_keys = {
        canonical_key(a.lhs - a.rhs) for a in atoms_out if isinstance(a, sp.Unequality)
    }
    if equality_keys & inequality_keys:
        return sp.S.false
    if not atoms_out:
        return sp.S.true
    if len(atoms_out) == 1:
        return atoms_out[0]
    return sp.And(*atoms_out, evaluate=False)


def normalize_parameter_condition(
    condition: sp.Expr | bool,
    *,
    parameters: Iterable[sp.Symbol] | None = None,
) -> sp.Expr:
    """Canonicalize a finite parameter condition deterministically.

    Polynomial equality/non-equality atoms use principal square-free radical
    normalization.  Conjunctive equality sets additionally use a bounded
    Gröbner basis, so ideal-equivalent generated strata acquire the same
    representative.  Arbitrary inequalities/non-polynomial predicates are kept
    structural rather than subjected to an unbounded logic search.
    """

    condition = sp.sympify(condition)
    params = _polynomial_parameters(condition, parameters)
    if condition in (sp.S.true, sp.S.false):
        return condition
    if condition.func is sp.And:
        return canonical_expr(_canonicalize_conjunction(condition.args, params))
    if condition.func is sp.Or:
        branches = [
            normalize_parameter_condition(branch, parameters=params) for branch in condition.args
        ]
        if any(branch is sp.S.true for branch in branches):
            return sp.S.true
        branches = [branch for branch in branches if branch is not sp.S.false]
        unique = {canonical_key(branch): branch for branch in branches}
        ordered = sorted(unique.values(), key=lambda item: repr(canonical_key(item)))
        if not ordered:
            return sp.S.false
        if len(ordered) == 1:
            return ordered[0]
        candidate = sp.Or(*ordered, evaluate=False)
        # Boolean minimization is useful for small generated DNF conditions but
        # must not become another unbounded symbolic search.
        if sp.count_ops(candidate) <= 50:
            try:
                simplified = sp.simplify_logic(candidate, form="dnf", force=True)
            except (NotImplementedError, TypeError, ValueError):
                simplified = candidate
            if simplified != candidate:
                return normalize_parameter_condition(simplified, parameters=params)
        return canonical_expr(candidate)
    return canonical_expr(_normalize_relational_atom(condition, params))


def parameter_conditions_equivalent(
    left: sp.Expr | bool, right: sp.Expr | bool, *, parameters: Iterable[sp.Symbol] | None = None
) -> bool:
    """Return whether two generated parameter conditions share a canonical form."""

    symbols = set(sp.sympify(left).free_symbols) | set(sp.sympify(right).free_symbols)
    params = (
        tuple(parameters)
        if parameters is not None
        else tuple(sorted(symbols, key=sp.default_sort_key))
    )
    return canonical_key(normalize_parameter_condition(left, parameters=params)) == canonical_key(
        normalize_parameter_condition(right, parameters=params)
    )


@dataclass(frozen=True)
class ParameterStratum(Generic[T]):
    """One certified parameter regime and the asymptotic result valid there."""

    condition: sp.Expr
    result: T
    knowledge: PropertyKnowledge = PropertyKnowledge.EXACT
    provenance: tuple[PropertyProvenance, ...] = ()
    decisions: tuple[PropertyDecision, ...] = ()
    complete: bool = True
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", normalize_parameter_condition(self.condition))


@dataclass(frozen=True)
class AsymptoticStratification(Generic[T]):
    """Finite conditional family of asymptotic results."""

    parameters: tuple[sp.Symbol, ...]
    strata: tuple[ParameterStratum[T], ...]
    assumptions: sp.Expr = sp.S.true
    exhaustive: bool = False
    provenance: tuple[PropertyProvenance, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        record_symbolic_event("parameter_strata", len(self.strata))
        object.__setattr__(
            self,
            "strata",
            tuple(sorted(self.strata, key=lambda s: repr(canonical_key(s.condition)))),
        )
        object.__setattr__(self, "assumptions", sp.sympify(self.assumptions))

    def select(self, assumptions: sp.Expr | bool = sp.S.true) -> ParameterStratum[T] | None:
        """Return the unique certified applicable stratum, if one exists."""

        combined = sp.And(self.assumptions, sp.sympify(assumptions))
        matches = [s for s in self.strata if entails(s.condition, combined) is True]
        return matches[0] if len(matches) == 1 else None

    @property
    def conditions(self) -> tuple[sp.Expr, ...]:
        return tuple(s.condition for s in self.strata)


def _parameters_in(conditions: Iterable[sp.Expr]) -> tuple[sp.Symbol, ...]:
    symbols = set()
    for condition in conditions:
        symbols.update(sp.sympify(condition).free_symbols)
    return tuple(sorted(symbols, key=sp.default_sort_key))


def _unsatisfiable(condition: sp.Expr) -> bool:
    """Best-effort bounded inconsistency check for generated parameter cases."""

    condition = normalize_parameter_condition(condition)
    if condition is sp.S.false:
        return True
    if condition is sp.S.true:
        return False
    if sp.count_ops(condition) <= 40:
        try:
            rels = list(sp.And.make_args(condition))
            symbols = set().union(*(r.free_symbols for r in rels))
            if symbols and all(isinstance(r, (Relational, BooleanAtom)) for r in rels):
                reduced = sp.reduce_inequalities(rels, list(symbols))
                if reduced is sp.S.false:
                    return True
        except (NotImplementedError, TypeError, ValueError):
            pass
    return bounded_assumption_entails(sp.S.false, condition) is True


def _coalesce_equivalent_strata(
    strata: list[ParameterStratum[T]],
    base: sp.Expr,
) -> list[ParameterStratum[T]]:
    """Merge overlapping strata when their mathematical results are identical.

    This removes the common zero/nonzero split explosion where distinct logical
    paths rediscover the same answer.  Overlaps carrying genuinely different
    results remain an error and are handled by the disjointness check below.
    """
    groups = []
    for stratum in strata:
        condition = normalize_parameter_condition(sp.And(base, stratum.condition))
        merged = False
        for index, current in enumerate(groups):
            if canonical_equal(current.result, stratum.result):
                groups[index] = ParameterStratum(
                    normalize_parameter_condition(sp.Or(current.condition, condition)),
                    current.result,
                    current.knowledge,
                    tuple(dict.fromkeys(current.provenance + stratum.provenance)),
                    current.decisions
                    + tuple(d for d in stratum.decisions if d not in current.decisions),
                    current.complete and stratum.complete,
                    tuple(dict.fromkeys(current.limitations + stratum.limitations)),
                )
                merged = True
                break
        if not merged:
            groups.append(
                ParameterStratum(
                    condition,
                    stratum.result,
                    stratum.knowledge,
                    stratum.provenance,
                    stratum.decisions,
                    stratum.complete,
                    stratum.limitations,
                )
            )
    return groups


def simplify_parameter_strata(
    stratification: AsymptoticStratification[T],
) -> AsymptoticStratification[T]:
    """Return a logically simplified stratification with equal-result overlaps merged."""
    strata = _coalesce_equivalent_strata(list(stratification.strata), stratification.assumptions)
    return AsymptoticStratification(
        stratification.parameters,
        tuple(strata),
        stratification.assumptions,
        stratification.exhaustive,
        stratification.provenance,
    )


def stratify_parameter_cases(
    cases: Iterable[tuple[sp.Expr, T] | ParameterStratum[T]],
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    parameters: Iterable[sp.Symbol] | None = None,
    require_disjoint: bool = True,
    require_exhaustive: bool = False,
    provenance: Iterable[PropertyProvenance] = (),
) -> AsymptoticStratification[T]:
    """Build and validate an explicit finite parameter stratification.

    Cases inconsistent with ``assumptions`` are discarded.  If requested,
    pairwise overlap and exhaustiveness are checked with the same tri-state
    entailment/satisfiability layer used by property enforcement.
    """

    base = normalize_parameter_condition(assumptions)
    strata = []
    for case in cases:
        stratum = case if isinstance(case, ParameterStratum) else ParameterStratum(case[0], case[1])
        if entails(sp.Not(stratum.condition), base) is True:
            continue
        strata.append(stratum)

    strata = _coalesce_equivalent_strata(strata, base)

    if require_disjoint:
        for i, left in enumerate(strata):
            for right in strata[i + 1 :]:
                overlap = sp.And(base, left.condition, right.condition)
                if not _unsatisfiable(overlap):
                    raise ValueError(
                        f"parameter strata are not certified disjoint: {left.condition} and {right.condition}"
                    )

    union = sp.Or(*(s.condition for s in strata)) if strata else sp.S.false
    exhaustive = entails(union, base) is True or _unsatisfiable(sp.And(base, sp.Not(union)))
    if require_exhaustive and not exhaustive:
        raise ValueError(
            "parameter strata are not certified exhaustive under the supplied assumptions"
        )

    params = (
        tuple(parameters) if parameters is not None else _parameters_in(s.condition for s in strata)
    )
    return AsymptoticStratification(params, tuple(strata), base, exhaustive, tuple(provenance))


def evaluate_parameter_strata(
    conditions: Iterable[sp.Expr],
    evaluator: Callable[[sp.Expr], T],
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    parameters: Iterable[sp.Symbol] | None = None,
    require_exhaustive: bool = False,
    provenance: Iterable[PropertyProvenance] = (),
) -> AsymptoticStratification[T]:
    """Evaluate the same algorithm independently on certified parameter cases."""

    base = normalize_parameter_condition(assumptions)
    cases = []
    for condition in conditions:
        condition = normalize_parameter_condition(condition)
        if entails(sp.Not(condition), base) is True:
            continue
        cases.append(ParameterStratum(condition, evaluator(sp.And(base, condition))))
    return stratify_parameter_cases(
        cases,
        assumptions=base,
        parameters=parameters,
        require_exhaustive=require_exhaustive,
        provenance=provenance,
    )


def zero_nonzero_stratification(
    expressions: Iterable[sp.Expr],
    evaluator: Callable[[sp.Expr], T],
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    parameters: Iterable[sp.Symbol] | None = None,
) -> AsymptoticStratification[T]:
    """Split automatically on zero/nonzero status of parameter expressions."""

    conditions = [sp.S.true]
    for expr in expressions:
        e = sp.sympify(expr)
        next_conditions = []
        for condition in conditions:
            next_conditions.extend((sp.And(condition, sp.Eq(e, 0)), sp.And(condition, sp.Ne(e, 0))))
        conditions = next_conditions
    return evaluate_parameter_strata(
        conditions,
        evaluator,
        assumptions=assumptions,
        parameters=parameters,
        require_exhaustive=True,
        provenance=(
            PropertyProvenance("asymptotic.parameter_stratification", note="zero/nonzero split"),
        ),
    )
