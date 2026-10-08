"""Uniform-asymptotic integration with local expansion and limit dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_limit
from .regime_selection import (
    AsymptoticRegime,
    RegimeCertificate,
    select_asymptotic_regime,
)
from .uniform_special_expansions import (
    bessel_j_large_order,
    bessel_j_prime_large_order,
    bessel_y_large_order,
    bessel_y_prime_large_order,
    hankel_large_order,
    hankel_prime_large_order,
    modified_bessel_i_large_order,
    modified_bessel_k_large_order,
)


@dataclass(frozen=True)
class UniformCompositeExpansion:
    expression: sp.Expr
    variable: sp.Symbol
    prefix: sp.Expr
    remainder_scale: sp.Expr
    regimes: tuple[RegimeCertificate, ...]
    certified: bool


def _transition_prefix(
    atom: sp.Expr, cert: RegimeCertificate, *, derivative: bool = False
) -> tuple[sp.Expr, sp.Expr] | None:
    n = cert.large_parameter
    a = cert.transition_parameter
    if atom.func is sp.besselj:
        if derivative:
            prefix = (
                -(2 ** sp.Rational(2, 3))
                * sp.airyaiprime(-(2 ** sp.Rational(1, 3)) * a)
                / n ** sp.Rational(2, 3)
            )
            return prefix, n ** -sp.Rational(4, 3)
        prefix = (
            2 ** sp.Rational(1, 3)
            * sp.airyai(-(2 ** sp.Rational(1, 3)) * a)
            / n ** sp.Rational(1, 3)
        )
        return prefix, n**-1
    if atom.func is sp.bessely:
        if derivative:
            prefix = (
                2 ** sp.Rational(2, 3)
                * sp.airybiprime(-(2 ** sp.Rational(1, 3)) * a)
                / n ** sp.Rational(2, 3)
            )
            return prefix, n ** -sp.Rational(4, 3)
        prefix = (
            -(2 ** sp.Rational(1, 3))
            * sp.airybi(-(2 ** sp.Rational(1, 3)) * a)
            / n ** sp.Rational(1, 3)
        )
        return prefix, n**-1
    if atom.func in (sp.hankel1, sp.hankel2):
        sign = 1 if atom.func is sp.hankel1 else -1
        prefix = (
            2 ** sp.Rational(4, 3)
            * sp.exp(-sign * sp.pi * sp.I / 3)
            * sp.airyai(sp.exp(-sign * sp.pi * sp.I / 3) * 2 ** sp.Rational(1, 3) * a)
            / n ** sp.Rational(1, 3)
        )
        return prefix, n**-1
    return None


def _scaled_prefix(
    atom: sp.Expr, cert: RegimeCertificate, *, derivative: bool = False
) -> tuple[sp.Expr, sp.Expr] | None:
    n, z = cert.large_parameter, cert.scaled_variable
    if atom.func is sp.besselj:
        result = (
            bessel_j_prime_large_order(n, z, terms=2)
            if derivative
            else bessel_j_large_order(n, z, terms=2)
        )
    elif atom.func is sp.bessely:
        result = (
            bessel_y_prime_large_order(n, z, terms=2)
            if derivative
            else bessel_y_large_order(n, z, terms=2)
        )
    elif atom.func is sp.hankel1:
        result = (
            hankel_prime_large_order(n, z, kind=1, terms=2)
            if derivative
            else hankel_large_order(n, z, kind=1, terms=2)
        )
    elif atom.func is sp.hankel2:
        result = (
            hankel_prime_large_order(n, z, kind=2, terms=2)
            if derivative
            else hankel_large_order(n, z, kind=2, terms=2)
        )
    elif atom.func is sp.besseli:
        result = modified_bessel_i_large_order(n, z, terms=4)
    elif atom.func is sp.besselk:
        result = modified_bessel_k_large_order(n, z, terms=4)
    else:
        return None
    if not result.certified:
        return None
    return result.prefix, result.order


def _derivative_signature(atom: sp.Expr) -> tuple[sp.Expr, sp.Expr] | None:
    """Recognize Subs(Derivative(F_n(x), x), x, argument)."""
    if not isinstance(atom, sp.Subs):
        return None
    derivative = atom.expr
    if not isinstance(derivative, sp.Derivative) or len(derivative.variables) != 1:
        return None
    base = derivative.expr
    if base.func not in (sp.besselj, sp.bessely, sp.hankel1, sp.hankel2):
        return None
    differentiation_variable = derivative.variables[0]
    if base.args[1] != differentiation_variable:
        return None
    if len(atom.variables) != 1 or atom.variables[0] != differentiation_variable:
        return None
    return base.func(base.args[0], atom.point[0]), atom.point[0]


def _uniform_atoms(
    expr: sp.Expr, variable: sp.Symbol
) -> tuple[tuple[sp.Expr, sp.Expr, bool], ...]:
    records = []
    seen = set()
    for candidate in sp.preorder_traversal(expr):
        signature = _derivative_signature(candidate)
        if signature is not None and variable in candidate.free_symbols:
            base, _ = signature
            key = (candidate, base, True)
            if candidate not in seen:
                records.append(key)
                seen.add(candidate)
            continue
        if (
            getattr(candidate, "func", None)
            in (sp.besselj, sp.bessely, sp.hankel1, sp.hankel2, sp.besseli, sp.besselk)
            and variable in candidate.free_symbols
            and not any(candidate.has(existing) for existing in seen)
            and candidate not in seen
        ):
            records.append((candidate, candidate, False))
            seen.add(candidate)
    return tuple(records)


def uniform_atom_expansion(
    atom: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = sp.oo,
    *,
    derivative: bool = False,
) -> tuple[sp.Expr, sp.Expr, RegimeCertificate] | None:
    cert = select_asymptotic_regime(atom, variable, point)
    if not cert.verified:
        return None
    if cert.regime is AsymptoticRegime.TURNING_POINT:
        pair = _transition_prefix(atom, cert, derivative=derivative)
    elif cert.regime is AsymptoticRegime.SCALED_ORDER_TAIL:
        pair = _scaled_prefix(atom, cert, derivative=derivative)
    else:
        return None
    if pair is None:
        return None
    return pair[0], pair[1], cert


def uniform_composite_expansion(
    expr: sp.Expr, variable: sp.Symbol, point: sp.Expr = sp.oo
) -> UniformCompositeExpansion | None:
    """Replace all compatible uniform atoms before algebraic cancellation."""
    expr = sp.sympify(expr)
    original_expr = expr
    expr = expr.replace(
        lambda e: (
            isinstance(e, sp.Subs)
            and e.has(sp.besselj, sp.bessely, sp.hankel1, sp.hankel2)
        ),
        lambda e: e.doit(),
    )
    # Put exact connection-equivalent Bessel/Hankel terms on one basis before
    # selecting asymptotic representatives; this exposes exact cancellations
    # without asking the Airy backend to rediscover connection identities.
    expr = expr.replace(
        lambda e: getattr(e, "func", None) is sp.hankel1,
        lambda e: sp.besselj(*e.args) + sp.I * sp.bessely(*e.args),
    )
    expr = expr.replace(
        lambda e: getattr(e, "func", None) is sp.hankel2,
        lambda e: sp.besselj(*e.args) - sp.I * sp.bessely(*e.args),
    )
    atom_records = _uniform_atoms(expr, variable)
    if not atom_records:
        if original_expr.has(sp.hankel1, sp.hankel2) and expr == 0:
            return UniformCompositeExpansion(
                original_expr, variable, sp.S.Zero, sp.S.Zero, (), True
            )
        return None
    replacements = {}
    scales = {}
    regimes = []
    unique_atoms = tuple(record[0] for record in atom_records)
    dummies = {atom: sp.Dummy(f"_uniform_{i}") for i, atom in enumerate(unique_atoms)}
    lifted = expr.xreplace(dummies)
    record_map = {
        original: (base, derivative) for original, base, derivative in atom_records
    }
    for atom in unique_atoms:
        base, derivative = record_map[atom]
        expansion = uniform_atom_expansion(base, variable, point, derivative=derivative)
        if expansion is None:
            return None
        prefix, scale, cert = expansion
        replacements[atom] = prefix
        scales[atom] = sp.sympify(scale)
        regimes.append(cert)
    dummy_prefixes = {dummies[atom]: replacements[atom] for atom in unique_atoms}
    prefix = sp.cancel(sp.expand(lifted.xreplace(dummy_prefixes)))
    # Propagate atom remainders through the exact polynomial composite when
    # possible. Expanding F(prefix + delta)-F(prefix) captures cross-products
    # that a first-order sensitivity estimate misses.
    deltas = {atom: sp.Dummy(f"_delta_{i}") for i, atom in enumerate(unique_atoms)}
    perturbed = {
        dummies[atom]: replacements[atom] + deltas[atom] for atom in unique_atoms
    }
    remainder_expression = sp.expand(lifted.xreplace(perturbed) - prefix)
    try:
        polynomial = sp.Poly(remainder_expression, *deltas.values())
    except sp.PolynomialError:
        remainder_terms = []
        for atom in unique_atoms:
            sensitivity = sp.diff(lifted, dummies[atom]).xreplace(dummy_prefixes)
            remainder_terms.append(sp.Abs(sensitivity) * scales[atom])
        remainder = sp.Add(*remainder_terms)
        algebra_certified = False
    else:
        remainder_terms = []
        delta_list = tuple(deltas.values())
        atom_by_delta = {deltas[atom]: atom for atom in unique_atoms}
        for powers, coefficient in polynomial.terms():
            scale = sp.S.One
            for delta, power in zip(delta_list, powers, strict=True):
                scale *= scales[atom_by_delta[delta]] ** power
            remainder_terms.append(sp.Abs(coefficient) * scale)
        remainder = sp.Add(*remainder_terms)
        algebra_certified = True
    return UniformCompositeExpansion(
        expr, variable, prefix, remainder, tuple(regimes), algebra_certified
    )


def uniform_limit_value(
    expr: sp.Expr, variable: sp.Symbol, point: sp.Expr = sp.oo
) -> tuple[sp.Expr, UniformCompositeExpansion] | None:
    expansion = uniform_composite_expansion(expr, variable, point)
    if expansion is None or not expansion.certified:
        return None
    stabilized_prefix = expansion.prefix
    for regime in expansion.regimes:
        if (
            regime.transition_parameter is not None
            and regime.transition_limit is not None
        ):
            stabilized_prefix = stabilized_prefix.xreplace(
                {regime.transition_parameter: regime.transition_limit}
            )
        if (
            regime.scaled_variable is not None
            and regime.limiting_scaled_variable is not None
        ):
            stabilized_prefix = stabilized_prefix.xreplace(
                {regime.scaled_variable: regime.limiting_scaled_variable}
            )
    continuous = (sp.airyai, sp.airybi, sp.airyaiprime, sp.airybiprime, sp.exp)

    def stabilize_special(function):
        try:
            argument_limit = bounded_limit(function.args[0], variable, point)
        except (
            TypeError,
            ValueError,
            NotImplementedError,
            RecursionError,
            sp.PoleError,
        ):
            return function
        if (
            argument_limit is None
            or isinstance(argument_limit, sp.Limit)
            or argument_limit.has(sp.nan, sp.zoo)
            or argument_limit in (sp.oo, -sp.oo)
        ):
            return function
        return function.func(argument_limit)

    stabilized_prefix = stabilized_prefix.replace(
        lambda e: getattr(e, "func", None) in continuous and variable in e.free_symbols,
        stabilize_special,
    )
    try:
        value = bounded_limit(stabilized_prefix, variable, point)
        remainder = bounded_limit(sp.Abs(expansion.remainder_scale), variable, point)
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        return None
    if (
        value is None
        or remainder != 0
        or isinstance(value, sp.Limit)
        or value.has(variable, sp.nan, sp.zoo)
    ):
        return None
    return value, expansion
