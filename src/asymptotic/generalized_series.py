"""Certified generalized local series with explicit remainder orders."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._polynomial_bounds import bounded_degree


@dataclass(frozen=True)
class GeneralizedSeriesTerm:
    exponent: sp.Expr
    coefficient: sp.Expr
    logarithmic_power: sp.Expr = sp.S.Zero


@dataclass(frozen=True)
class SeriesRemainderCertificate:
    order: sp.Expr
    logarithmic_power_bound: sp.Expr | None
    exact: bool
    statement: str


@dataclass(frozen=True)
class GeneralizedSeries:
    terms: tuple[GeneralizedSeriesTerm, ...]
    order_bound: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    remainder: SeriesRemainderCertificate
    provider: str = "generalized_series"

    @property
    def certified(self):
        return self.remainder is not None

    def as_expr(self):
        z = (
            self.variable - self.point
            if self.point not in (sp.oo, -sp.oo)
            else (1 / self.variable if self.point is sp.oo else -1 / self.variable)
        )
        return sp.Add(
            *(
                t.coefficient * z**t.exponent * sp.log(z) ** t.logarithmic_power
                for t in self.terms
            )
        )


def _parse_generalized_term(term, z):
    pd = term.as_powers_dict()
    exponent = sp.simplify(pd.get(z, sp.S.Zero))
    logz = sp.log(z)
    log_power = sp.simplify(pd.get(logz, sp.S.Zero))
    coeff = sp.simplify(term / (z**exponent * logz**log_power))
    if coeff.has(z):
        return None
    return GeneralizedSeriesTerm(exponent, coeff, log_power)


def _order_from_O(order, z):
    if order is None:
        return None
    try:
        n = order.getn()
        if n is not None:
            return sp.sympify(n)
    except (AttributeError, NotImplementedError, ValueError):
        pass
    e = order.expr
    pd = e.as_powers_dict()
    n = pd.get(z)
    return sp.sympify(n) if n is not None else None


def _rational_jet(local, z, order):
    """Construct a bounded regular rational jet with an analytic remainder.

    Equating coefficients in Q*A=P gives a triangular recurrence. Q(0)!=0
    certifies analyticity, so omitted terms are O(z**order).
    """
    if not isinstance(order, (int, sp.Integer)) or not 1 <= order <= 64:
        return None
    if sp.count_ops(local) > 120 or local.has(sp.Float):
        return None
    numerator, denominator = local.as_numer_denom()
    # Check degree structurally before Poly expands powers and products.
    if any(bounded_degree(a, (z,), 64) is None for a in (numerator, denominator)):
        return None
    try:
        p, q = (sp.Poly(a, z, domain=sp.QQ) for a in (numerator, denominator))
    except (sp.PolynomialError, sp.CoercionFailed):
        return None
    if q.nth(0) == 0 or max(p.degree(), q.degree()) > 64:
        return None
    coefficients = []
    for k in range(int(order)):
        coefficients.append(
            (
                p.nth(k)
                - sum(
                    q.nth(j) * coefficients[k - j]
                    for j in range(1, min(k, q.degree()) + 1)
                )
            )
            / q.nth(0)
        )
    out = tuple(
        GeneralizedSeriesTerm(sp.Integer(k), c)
        for k, c in enumerate(coefficients)
        if c != 0
    )
    exact = q.degree() == 0 and p.degree() < order
    remainder = SeriesRemainderCertificate(
        sp.oo if exact else sp.Integer(order),
        None,
        exact,
        "finite exact rational polynomial"
        if exact
        else "analytic rational jet remainder",
    )
    return out, remainder


def generalized_series(expr, variable, *, point=0, terms=4):
    expr = sp.sympify(expr)
    variable = sp.sympify(variable)
    point = sp.sympify(point)
    z = sp.Dummy("series_z", positive=True)
    local = expr.xreplace(
        {
            variable: (
                1 / z if point is sp.oo else (-1 / z if point is -sp.oo else point + z)
            )
        }
    )
    jet = _rational_jet(local, z, terms)
    if jet is not None:
        out, remainder = jet
        return GeneralizedSeries(out, remainder.order, variable, point, remainder)
    known_remainder = None
    if point == 0 and expr.func is sp.bessely and expr.args == (0, variable):
        local = sp.Rational(2, 1) / sp.pi * (sp.log(z / 2) + sp.EulerGamma)
        known_remainder = SeriesRemainderCertificate(
            sp.Integer(2), sp.Integer(1), False, "Y_0(z) remainder is O(z^2 log z)"
        )
    try:
        raw = sp.series(local, z, 0, terms)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return None
    order_symbol = raw.getO() if hasattr(raw, "getO") else None
    body = raw.removeO().expand() if hasattr(raw, "removeO") else sp.expand(raw)
    out = []
    for term in sp.Add.make_args(body):
        parsed = _parse_generalized_term(term, z)
        if parsed is None:
            return None
        out.append(parsed)
    out.sort(
        key=lambda q: (
            sp.default_sort_key(q.exponent),
            sp.default_sort_key(q.logarithmic_power),
        )
    )
    if known_remainder is not None:
        rem = known_remainder
    elif order_symbol is not None:
        bound = _order_from_O(order_symbol, z)
        if bound is None:
            return None
        rem = SeriesRemainderCertificate(
            bound, None, False, f"exact SymPy Order term {order_symbol}"
        )
    else:
        # No O-term is a certificate only when the returned expression is
        # algebraically identical to the local germ.
        if sp.simplify(local - body) != 0:
            return None
        rem = SeriesRemainderCertificate(
            sp.oo, None, True, "finite exact local expression"
        )
    return GeneralizedSeries(tuple(out), rem.order, variable, point, rem)


def series_add(a, b):
    if not (a and b and a.certified and b.certified) or (a.variable, a.point) != (
        b.variable,
        b.point,
    ):
        return None
    z = a.variable - a.point
    e = sp.expand(a.as_expr() + b.as_expr())
    # Sum remainder is bounded by the smaller certified order.
    bound = sp.Min(a.order_bound, b.order_bound)
    terms = []
    for q in sp.Add.make_args(e):
        t = _parse_generalized_term(q, z)
        if t is None:
            return None
        terms.append(t)
    return GeneralizedSeries(
        tuple(terms),
        bound,
        a.variable,
        a.point,
        SeriesRemainderCertificate(bound, None, False, "sum of certified remainders"),
    )


def series_mul(a, b):
    if not (a and b and a.certified and b.certified) or (a.variable, a.point) != (
        b.variable,
        b.point,
    ):
        return None
    if not a.terms or not b.terms:
        return None
    la = min((t.exponent for t in a.terms), key=sp.default_sort_key)
    lb = min((t.exponent for t in b.terms), key=sp.default_sort_key)
    bound = sp.Min(a.order_bound + lb, b.order_bound + la)
    z = a.variable - a.point
    e = sp.expand(a.as_expr() * b.as_expr())
    terms = []
    for q in sp.Add.make_args(e):
        t = _parse_generalized_term(q, z)
        if t is None:
            return None
        terms.append(t)
    return GeneralizedSeries(
        tuple(terms),
        bound,
        a.variable,
        a.point,
        SeriesRemainderCertificate(
            bound, None, False, "product of certified remainders"
        ),
    )


def singularity_series(outer, inner, variable, *, inner_limit=0, terms=4):
    u = sp.Dummy("singular_u", positive=True)
    try:
        raw = sp.series(outer(u), u, inner_limit, terms)
    except (TypeError, ValueError, NotImplementedError):
        return None
    return sp.expand(raw.removeO().xreplace({u: sp.sympify(inner)}))


__all__ = [
    "GeneralizedSeries",
    "GeneralizedSeriesTerm",
    "SeriesRemainderCertificate",
    "generalized_series",
    "series_add",
    "series_mul",
    "singularity_series",
]
