"""Certified and algebraic helpers for asymptotic discrete summation."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_assumption_sign
from .remainder import AsymptoticRemainder


@dataclass(frozen=True)
class UniformSummationCertificate:
    """Evidence justifying interchange of a finite asymptotic prefix and a sum."""

    kind: str
    lower: sp.Expr
    upper: sp.Expr
    majorant: sp.Expr | None = None

    def replay(self) -> bool:
        if self.kind == "fixed-finite-lattice":
            return bool(
                self.lower.is_integer
                and self.upper.is_integer
                and self.lower.is_finite
                and self.upper.is_finite
            )
        if self.kind == "summable-geometric-majorant":
            return self.majorant is not None and self.majorant.is_finite is True
        return False


@dataclass(frozen=True)
class MellinShiftCertificate:
    """Replayable structural certificate for a left Mellin contour shift."""

    initial_strip: tuple[sp.Expr, sp.Expr]
    initial_line: sp.Expr
    shifted_line: sp.Expr
    poles: tuple[sp.Expr, ...]
    gamma_decay: sp.Expr
    parameter_positive: bool
    remainder: AsymptoticRemainder

    @staticmethod
    def _proved_lt(left: sp.Expr, right: sp.Expr) -> bool:
        if left == -sp.oo or right == sp.oo:
            return True
        return bounded_assumption_sign(sp.simplify(right - left)) == 1

    def replay(self) -> bool:
        lo, hi = self.initial_strip
        strip_ok = self._proved_lt(lo, self.initial_line) and self._proved_lt(self.initial_line, hi)
        crossed = bool(self.poles) and all(
            self._proved_lt(self.shifted_line, pole) and self._proved_lt(pole, self.initial_line)
            for pole in self.poles
        )
        return bool(
            strip_ok
            and crossed
            and self.parameter_positive
            and self.gamma_decay.is_positive is True
            and self.remainder.is_certified
        )


@dataclass(frozen=True)
class CreativeTelescopingCertificate:
    """Exact certificate ``sum p_j(n)F(n+j,k) = Delta_k(R F)``."""

    summand: sp.Expr
    sum_index: sp.Symbol
    parameter: sp.Symbol
    coeffs: tuple[sp.Expr, ...]
    rational_certificate: sp.Expr

    def identity(self) -> sp.Expr:
        k = self.sum_index
        n = self.parameter
        f = self.summand
        lhs = sum(c * f.xreplace({n: n + j}) for j, c in enumerate(self.coeffs))
        g = analytic_powsimp(self.rational_certificate * f)
        rhs = g.xreplace({k: k + 1}) - g
        return sp.cancel(sp.combsimp(lhs - rhs))

    def replay(self) -> bool:
        return self.identity() == 0


def fixed_finite_uniformity(
    lower: sp.Expr, upper: sp.Expr, parameter: sp.Symbol
) -> UniformSummationCertificate | None:
    if parameter in sp.Tuple(lower, upper).free_symbols:
        return None
    if lower.is_integer and upper.is_integer and lower.is_finite and upper.is_finite:
        return UniformSummationCertificate("fixed-finite-lattice", lower, upper)
    return None


def geometric_uniformity(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
) -> UniformSummationCertificate | None:
    """Certify a narrow Weierstrass-majorant pattern on an infinite lattice.

    The recognized form is ``a*k**(-p) * (1+c*x/k**q)**(-m)`` with
    positive ``x,c``, positive integers ``m,q``, and enough p-series decay
    after the requested Taylor truncation.  The binomial-series remainder is
    then bounded uniformly for ``0 <= x <= 1`` by a constant multiple of a
    convergent p-series.
    """
    if (
        point != 0
        or upper is not sp.oo
        or lower.is_integer is not True
        or parameter.is_positive is not True
    ):
        return None
    param_factors = []
    for base, exponent in summand.as_powers_dict().items():
        if parameter in base.free_symbols:
            param_factors.append((base, exponent))
    if len(param_factors) != 1:
        return None
    base, exponent = param_factors[0]
    if exponent.is_integer is not True or exponent.is_negative is not True:
        return None
    if sp.simplify(base.subs(parameter, 0) - 1) != 0:
        return None
    coeff = sp.simplify(sp.diff(base, parameter))
    if parameter in coeff.free_symbols or sp.diff(base, parameter, 2) != 0:
        return None
    powers = coeff.as_powers_dict()
    k_exp = powers.get(variable, sp.S.Zero)
    q = -k_exp
    c = sp.simplify(coeff / variable**k_exp)
    if q.is_integer is not True or q.is_positive is not True or c.is_positive is not True:
        return None
    independent = sp.simplify(summand / base**exponent)
    ipowers = independent.as_powers_dict()
    ik_exp = ipowers.get(variable, sp.S.Zero)
    amplitude = sp.simplify(independent / variable**ik_exp)
    if variable in amplitude.free_symbols or parameter in amplitude.free_symbols:
        return None
    decay = sp.simplify(-(ik_exp - q * terms))
    if decay.is_real is not True or sp.simplify(decay > 1) is not sp.true:
        return None
    majorant = sp.Abs(amplitude) * sp.zeta(decay, lower)
    return UniformSummationCertificate("summable-geometric-majorant", lower, upper, majorant)


def _rational_shift_ratio(expr: sp.Expr, symbol: sp.Symbol) -> sp.Expr | None:
    try:
        ratio = sp.cancel(expr.xreplace({symbol: symbol + 1}) / expr)
    except (TypeError, ValueError, ZeroDivisionError, *SYMBOLIC_ERRORS):
        return None
    if ratio.has(sp.gamma, sp.factorial, sp.binomial):
        ratio = sp.combsimp(ratio)
    ratio = sp.cancel(ratio)
    try:
        sp.Poly(sp.together(ratio).as_numer_denom()[0], symbol)
        sp.Poly(sp.together(ratio).as_numer_denom()[1], symbol)
    except sp.PolynomialError:
        return None
    return ratio


def zeilberger_recurrence(
    summand: sp.Expr,
    sum_index: sp.Symbol,
    parameter: sp.Symbol,
    *,
    max_order: int = 3,
    coeff_degree: int = 2,
    cert_degree: int = 4,
) -> CreativeTelescopingCertificate | None:
    """Find a low-order Zeilberger relation for a bivariate hypergeometric term.

    This is a genuine creative-telescoping search: unknown polynomial recurrence
    coefficients and a rational Gosper certificate are solved simultaneously
    from the cleared polynomial identity.  Search bounds are intentionally
    small and deterministic; failure returns ``None`` rather than invoking a
    general solver.
    """
    f = sp.sympify(summand)
    k, n = sum_index, parameter
    tk = _rational_shift_ratio(f, k)
    if tk is None:
        return None
    q = [sp.S.One]
    for _ in range(max_order):
        j = len(q)
        try:
            qj = sp.cancel(f.xreplace({n: n + j}) / f)
            qj = sp.combsimp(qj)
        except (TypeError, ValueError, *SYMBOLIC_ERRORS):
            break
        if qj.has(sp.gamma, sp.factorial, sp.binomial):
            qj = sp.combsimp(qj)
        q.append(sp.cancel(qj))

    for order in range(1, min(max_order, len(q) - 1) + 1):
        denominators = [sp.factor(sp.denom(sp.together(tk)))]
        denominators.extend(sp.factor(sp.denom(sp.together(x))) for x in q[: order + 1])
        factors: list[sp.Expr] = [sp.S.One]
        for den in denominators:
            for factor, _mult in sp.factor_list(den, k)[1]:
                if factor not in factors:
                    factors.append(factor)
        cert_denoms = factors + [
            sp.factor(a * b) for a in factors[1:] for b in factors[1:] if a != b
        ]
        for cert_den in cert_denoms:
            p_symbols = []
            coeff_polys = []
            for j in range(order + 1):
                cs = sp.symbols(f"ct_p{j}_0:{coeff_degree + 1}")
                p_symbols.extend(cs)
                coeff_polys.append(sum(cs[r] * n**r for r in range(coeff_degree + 1)))
            r_symbols = sp.symbols(f"ct_r0:{cert_degree + 1}")
            r_num = sum(r_symbols[r] * k**r for r in range(cert_degree + 1))
            unknowns = tuple(p_symbols) + tuple(r_symbols)
            r = r_num / cert_den
            lhs = sum(coeff_polys[j] * q[j] for j in range(order + 1))
            identity = sp.together(lhs - (r.xreplace({k: k + 1}) * tk - r))
            try:
                numerator = sp.Poly(sp.expand(identity.as_numer_denom()[0]), k, n)
            except sp.PolynomialError:
                continue
            equations = [c for c in numerator.coeffs()]
            if not equations:
                continue
            matrix, _rhs = sp.linear_eq_to_matrix(equations, unknowns)
            null = matrix.nullspace()
            for vector in null:
                mapping = dict(zip(unknowns, vector))
                coeffs = tuple(sp.factor(p.subs(mapping)) for p in coeff_polys)
                if all(c == 0 for c in coeffs):
                    continue
                cert = sp.cancel(r.subs(mapping))
                first = next((c for c in coeffs if c != 0), sp.S.One)
                coeffs = tuple(sp.cancel(c / first) for c in coeffs)
                cert = sp.cancel(cert / first)
                result = CreativeTelescopingCertificate(f, k, n, coeffs, cert)
                if result.replay():
                    return result
    return None


def endpoint_recurrence(
    summand: sp.Expr,
    sum_index: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
) -> tuple[sp.Expr, sp.Expr] | None:
    """Return ``S(n+1)-S(n)=boundary`` for a partial sum with upper ``n+c``."""
    if parameter in summand.free_symbols:
        return None
    shift = sp.simplify(upper - parameter)
    if shift.is_integer is not True or parameter in shift.free_symbols:
        return None
    rhs = summand.xreplace({sum_index: upper + 1})
    return sp.S.One, analytic_powsimp(rhs)


def gamma_vertical_decay(expr: sp.Expr, s: sp.Symbol) -> sp.Expr:
    """Return the exponential vertical-decay coefficient from Gamma factors."""
    total = sp.S.Zero
    for gamma in expr.atoms(sp.gamma):
        arg = gamma.args[0]
        slope = sp.diff(arg, s)
        if s not in slope.free_symbols and slope.is_real is not False:
            total += sp.Abs(slope)
    return sp.simplify(sp.pi * total / 2)


def poisson_gaussian_sum(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    parameter: sp.Symbol,
    point: sp.Expr,
) -> tuple[sp.Expr, AsymptoticRemainder] | None:
    """Poisson summation for a Gaussian lattice, including linear oscillation."""
    if (lower, upper, point) != (-sp.oo, sp.oo, sp.S.Zero):
        return None
    if parameter.is_positive is not True:
        return None
    k = variable
    # Match exp(-a*p*k**2 + I*b*k) times a parameter-independent constant.
    exponentials = list(summand.atoms(sp.exp))
    if len(exponentials) != 1:
        return None
    e = exponentials[0]
    prefactor = sp.cancel(summand / e)
    if k in prefactor.free_symbols or parameter in prefactor.free_symbols:
        return None
    phase = sp.expand(e.args[0])
    a = -sp.expand(phase).coeff(k, 2) / parameter
    b = sp.expand(phase).coeff(k, 1) / sp.I
    residual = sp.simplify(phase + a * parameter * k**2 - sp.I * b * k)
    if (
        residual != 0
        or parameter in a.free_symbols
        or parameter in b.free_symbols
        or k in a.free_symbols
        or k in b.free_symbols
    ):
        return None
    if a.is_positive is not True or b.is_real is not True:
        return None
    # This implementation keeps the m=0 dual image as the leading term. That
    # is valid only in the principal Fourier cell; otherwise another dual image
    # can dominate and the claimed remainder scale would be wrong.
    principal = bounded_assumption_sign(sp.pi - sp.Abs(b))
    if principal != 1:
        return None
    # Fourier transform convention sum f(k)=sum fhat(2*pi*m).
    leading = prefactor * sp.sqrt(sp.pi / (a * parameter)) * sp.exp(-(b**2) / (4 * a * parameter))
    # Nearest omitted dual lattice image determines an exponentially small bound.
    gap = sp.Min((2 * sp.pi - b) ** 2, (2 * sp.pi + b) ** 2)
    scale = sp.exp(-gap / (4 * a * parameter)) / sp.sqrt(parameter)
    rem = AsymptoticRemainder.big_o(scale, parameter, 0, source="Poisson Gaussian dual-tail bound")
    return analytic_powsimp(leading), rem


def linear_exponential_sum(
    summand: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
) -> sp.Expr | None:
    """Exact finite oscillatory geometric sum for an affine exponential phase."""
    if upper in (sp.oo, -sp.oo) or lower in (sp.oo, -sp.oo):
        return None
    if lower.is_integer is not True or upper.is_integer is not True:
        return None
    ratio = _rational_shift_ratio(summand, variable)
    if ratio is None or variable in ratio.free_symbols:
        return None
    first = summand.xreplace({variable: lower})
    count = sp.simplify(upper - lower + 1)
    if ratio == 1:
        return analytic_powsimp(first * count)
    return analytic_powsimp(first * (1 - ratio**count) / (1 - ratio))


def separable_multisum(
    summand: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> tuple[sp.Expr, ...] | None:
    """Factor a product into one factor for each independent summation variable."""
    factors = list(sp.Mul.make_args(sp.factor_terms(summand)))
    groups = [sp.S.One for _ in variables]
    constant = sp.S.One
    for factor in factors:
        hits = [i for i, v in enumerate(variables) if v in factor.free_symbols]
        if not hits:
            constant *= factor
        elif len(hits) == 1:
            groups[hits[0]] *= factor
        else:
            return None
    groups[0] *= constant
    return tuple(map(analytic_powsimp, groups))
