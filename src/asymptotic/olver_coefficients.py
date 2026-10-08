"""Olver coefficient functions for Bessel turning-point expansions."""

from __future__ import annotations

from functools import cache

import sympy as sp

from ._symbolic_policy import bounded_ask


@cache
def debye_u(index: int) -> sp.Expr:
    """Return the Debye polynomial U_index(p) from its differential recurrence."""
    if index < 0:
        raise ValueError("index must be nonnegative")
    p = sp.Symbol("p")
    u = sp.S.One
    t = sp.Symbol("_t")
    for _ in range(index):
        integrand = (1 - 5 * t**2) * u.xreplace({p: t})
        u = sp.expand(
            sp.Rational(1, 2) * p**2 * (1 - p**2) * sp.diff(u, p)
            + sp.Rational(1, 8) * sp.integrate(integrand, (t, 0, p))
        )
    return sp.factor(u)


@cache
def debye_v(index: int) -> sp.Expr:
    """Return the derivative Debye polynomial V_index(p)."""
    if index < 0:
        raise ValueError("index must be nonnegative")
    if index == 0:
        return sp.S.One
    p = sp.Symbol("p")
    previous = debye_u(index - 1)
    return sp.factor(
        debye_u(index)
        - sp.Rational(1, 2) * p * (1 - p**2) * previous
        - p**2 * (1 - p**2) * sp.diff(previous, p)
    )


@cache
def airy_u(index: int) -> sp.Expr:
    """Coefficient u_index in the standard Airy Poincare expansion."""
    if index < 0:
        raise ValueError("index must be nonnegative")
    value = sp.S.One
    for k in range(1, index + 1):
        value *= sp.Rational(
            (6 * k - 5) * (6 * k - 3) * (6 * k - 1),
            (2 * k - 1) * 216 * k,
        )
    return sp.factor(value)


@cache
def airy_v(index: int) -> sp.Expr:
    """Coefficient v_index in the differentiated Airy expansion."""
    if index < 0:
        raise ValueError("index must be nonnegative")
    if index == 0:
        return sp.S.One
    return sp.factor(sp.Rational(6 * index + 1, 1 - 6 * index) * airy_u(index))


def olver_zeta(z: sp.Expr) -> sp.Expr:
    """Olver coordinate continued from the positive axis on C\\(-inf,0].

    The positive real interval z>1 uses the oscillatory continuation explicitly
    so that zeta remains negative real. Elsewhere principal sqrt/log values
    select the analytic continuation from the cut plane.
    """
    z = sp.sympify(z)
    if z == 1:
        return sp.S.Zero
    if z.is_real is True and bounded_ask(sp.Q.positive(z - 1)) is True:
        phase = sp.sqrt(z**2 - 1) - sp.acos(1 / z)
        return -((sp.Rational(3, 2) * phase) ** sp.Rational(2, 3))
    q = sp.sqrt(1 - z**2)
    eta = sp.log((1 + q) / z) - q
    return (sp.Rational(3, 2) * eta) ** sp.Rational(2, 3)


def _olver_powers(
    z: sp.Expr, zeta: sp.Expr, j: int
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return zeta^(1/2), zeta^(-3j/2), and p on the continued branch."""
    if z.is_real is True and bounded_ask(sp.Q.positive(z - 1)) is True:
        root = sp.sqrt(-zeta)
        return (
            -sp.I * root,
            sp.I ** (3 * j) * (-zeta) ** (-sp.Rational(3 * j, 2)),
            sp.I / sp.sqrt(z**2 - 1),
        )
    root = sp.sqrt(zeta)
    return root, zeta ** (-sp.Rational(3 * j, 2)), 1 / sp.sqrt(1 - z**2)


def _coefficient_sum(
    family: str,
    index: int,
    zeta: sp.Expr,
    p: sp.Expr,
    zeta_root: sp.Expr | None = None,
    powers: tuple[sp.Expr, ...] | None = None,
) -> sp.Expr:
    poly = debye_v if family in {"C", "D"} else debye_u
    airy = airy_v if family in {"A", "C"} else airy_u
    if family in {"A", "D"}:
        js = range(2 * index + 1)
        offset = 0
    else:
        js = range(2 * index + 2)
        offset = 1
    if zeta_root is None:
        zeta_root = sp.sqrt(zeta)
    terms = []
    for j in js:
        power = powers[j] if powers is not None else zeta ** (-sp.Rational(3 * j, 2))
        terms.append(
            sp.Rational(3, 2) ** j
            * airy(j)
            * power
            * poly(2 * index - j + offset).subs(sp.Symbol("p"), p)
        )
    total = sp.Add(*terms)
    if family == "B":
        return -total / zeta_root
    if family == "C":
        return -zeta_root * total
    return total


@cache
def _turning_value(family: str, index: int) -> sp.Expr:
    """Evaluate the removable zeta=0 value using the analytic q coordinate."""
    q = sp.Symbol("_q", positive=True)
    # z=sqrt(1-q^2) gives eta=atanh(q)-q and makes zeta/q^2 analytic.
    eta = sp.atanh(q) - q
    zeta = q**2 * (sp.Rational(3, 2) * eta / q**3) ** sp.Rational(2, 3)
    expression = _coefficient_sum(family, index, zeta, 1 / q)
    return sp.simplify(sp.series(expression, q, 0, 2).removeO().subs(q, 0))


def olver_coefficient(family: str, index: int, z: sp.Expr) -> sp.Expr:
    """Return analytically continued A_k, B_k, C_k, or D_k."""
    family = family.upper()
    if family not in {"A", "B", "C", "D"}:
        raise ValueError("family must be A, B, C, or D")
    if index < 0:
        raise ValueError("index must be nonnegative")
    z = sp.sympify(z)
    if z == 1:
        return _turning_value(family, index)
    zeta = olver_zeta(z)
    count = 2 * index + (2 if family in {"B", "C"} else 1)
    branch = [_olver_powers(z, zeta, j) for j in range(count)]
    root = branch[0][0]
    powers = tuple(item[1] for item in branch)
    p = branch[0][2]
    return _coefficient_sum(family, index, zeta, p, root, powers)


def olver_coefficient_block(terms: int, z: sp.Expr) -> dict[str, tuple[sp.Expr, ...]]:
    """Return matched A/B/C/D coefficient blocks through ``terms-1``.

    All four families are generated by the same Debye/Airy recurrences and
    evaluated with the same continuation and removable turning-point rules.
    """
    if terms < 1:
        raise ValueError("terms must be positive")
    z = sp.sympify(z)
    return {
        family: tuple(olver_coefficient(family, k, z) for k in range(terms))
        for family in ("A", "B", "C", "D")
    }


def olver_turning_table(terms: int) -> dict[str, tuple[sp.Expr, ...]]:
    """Return exact removable coefficient values at the Bessel turning point."""
    return olver_coefficient_block(terms, sp.S.One)
