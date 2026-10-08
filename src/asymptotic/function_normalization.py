"""Exact real-root, discrete and periodic function definitions.

Unknown names, held expressions, missing definitions and unresolved real-root
branches remain unchanged. Undefined functions retain their unspecified semantics.
"""

import sympy as sp

from .limit_models import LimitEvidence
from .step_factorials import StepFactorialPower


class NearestInteger(sp.Function):
    """Nearest-integer function with exact integral values."""

    nargs = 1

    @classmethod
    def eval(cls, z):
        if z.is_integer is True:
            return z

    def _eval_is_real(self):
        return self.args[0].is_real


class TriangleWave(sp.Function):
    """Real unit-period triangle: 2/pi*asin(sin(2*pi*z))."""

    nargs = 1

    @classmethod
    def eval(cls, z):
        if z.is_Rational:
            t = z - sp.floor(z)
            if t <= sp.Rational(1, 4):
                return 4 * t
            if t <= sp.Rational(3, 4):
                return 2 - 4 * t
            return 4 * t - 4

    def _eval_rewrite_as_asin(self, z, **kwargs):
        if z.is_real is True:
            return 2 / sp.pi * sp.asin(sp.sin(2 * sp.pi * z))


class RealRoot(sp.Function):
    """Real nth root with the even-order domain retained explicitly."""

    nargs = 2

    @classmethod
    def eval(cls, argument, order):
        if argument.is_real is False:
            return None
        if order.is_Integer is True and order > 0:
            if (
                order.is_odd is True and argument.is_real is True
            ) or argument.is_nonnegative is True:
                return sp.real_root(argument, order)


def normalize_functions(expr, variables, target, assumptions=sp.S.true):
    """Reduce real-root and finite-product definitions on an admissible chart.

    Even roots need an eventual positive real argument. Fixed negative step
    factorials need a nonzero base before their rational product is exposed.
    Unknown functions keep their unspecified semantics.
    """
    expr = sp.sympify(expr)
    if not isinstance(expr, sp.Expr) or sp.count_ops(expr) > 160:
        return expr, ()
    from ._limit_composition import _regular_composition
    from .discontinuous_functions import SignedFractionalPart

    replacements = {}
    for atom in expr.atoms(sp.acot):
        argument = atom.args[0]
        if sp.count_ops(argument) > 30 or any(v.is_real is not True for v in variables):
            continue
        from ._polynomial_bounds import bounded_degree, bounded_expansion_width

        if bounded_expansion_width(argument, 64) > 64:
            continue
        numerator, denominator = argument.as_numer_denom()
        if any(
            bounded_degree(part, variables, 16) is None
            for part in (numerator, denominator)
        ):
            continue
        try:
            sp.Poly(numerator, *variables, domain=sp.QQ)
            polynomial = sp.Poly(denominator, *variables, domain=sp.QQ)
        except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
            continue
        # Inverting the rational argument removes its denominator holes.
        # This chart is valid on the whole punctured neighborhood only when
        # that denominator has a coercive signed lower bound there.
        from .uniform_radial_bounds import _positive_order

        point = tuple(map(sp.sympify, target))
        if any(
            p.is_real is not True or p.is_finite is not True or p.free_symbols
            for p in point
        ):
            continue
        shifted = denominator.subs(
            dict(
                zip(
                    variables,
                    (v + p for v, p in zip(variables, point, strict=True)),
                    strict=True,
                )
            ),
            simultaneous=True,
        )
        lower = _positive_order(shifted, variables)
        if lower is None:
            lower = _positive_order(-shifted, variables)
        if lower is None or lower.is_positive is not True:
            continue
        center = _regular_composition(numerator, variables, target)
        reciprocal = sp.cancel(1 / argument)
        if (
            not polynomial.is_zero
            and center is not None
            and center.is_zero is False
            and _regular_composition(reciprocal, variables, target) == 0
        ):
            replacements[atom] = sp.atan(reciprocal)
    for atom in expr.atoms(SignedFractionalPart):
        argument = atom.args[0]
        if (
            argument.is_real is True
            and _regular_composition(argument, variables, target) == 0
        ):
            replacements[atom] = argument
    if len(variables) > 1 and len(variables) == len(target):
        point_condition = sp.And(
            *(sp.Eq(v, p) for v, p in zip(variables, target, strict=True))
        )
        for piece in sorted(expr.atoms(sp.Piecewise), key=sp.count_ops):
            branches = [
                (value, condition)
                for value, condition in piece.args
                if condition != point_condition
            ]
            # The ambient punctured germ excludes this single point. A
            # selector fixing only some coordinates is a seam and must stay.
            if branches and len(branches) != len(piece.args):
                replacements[piece] = sp.Piecewise(*branches)
    if len(variables) == len(target) == 1 and variables[0].is_integer is not True:
        from .local_tail_germs import fixed_finite

        variable, point = variables[0], target[0]
        for piece in sorted(expr.atoms(sp.Piecewise), key=sp.count_ops):
            branches = []
            changed = False
            for branch, condition in piece.args:
                truth = None
                from sympy.core.relational import Relational

                if isinstance(condition, Relational):
                    delta = condition.lhs - condition.rhs
                    try:
                        polynomial = sp.Poly(delta, variable)
                    except sp.PolynomialError:
                        polynomial = None
                    if (
                        polynomial is not None
                        and 0 < polynomial.degree() <= 8
                        and polynomial.LC().is_zero is False
                        and all(fixed_finite(c) for c in polynomial.all_coeffs())
                    ):
                        if isinstance(condition, sp.Equality):
                            truth = sp.S.false
                        elif isinstance(condition, sp.Unequality):
                            truth = sp.S.true
                        elif point in (sp.oo, -sp.oo) and all(
                            c.is_real is True for c in polynomial.all_coeffs()
                        ):
                            leading = polynomial.LC() * (
                                -1 if point == -sp.oo and polynomial.degree() % 2 else 1
                            )
                            if (
                                leading.is_positive is True
                                or leading.is_negative is True
                            ):
                                truth = condition.func(
                                    1 if leading.is_positive is True else -1, 0
                                )
                if truth is sp.S.false:
                    changed = True
                    continue
                branches.append(
                    (
                        branch.xreplace(replacements),
                        truth if truth is not None else condition,
                    )
                )
                if truth is sp.S.true:
                    changed = True
                    break
            if changed:
                replacements[piece] = sp.Piecewise(*branches)
    expr_with_branches = expr.xreplace(replacements)
    for atom in sorted(
        expr_with_branches.atoms(RealRoot, StepFactorialPower, sp.Product, sp.catalan),
        key=sp.count_ops,
    ):
        args = tuple(a.xreplace(replacements) for a in atom.args)
        if atom.func is sp.catalan:
            value = atom.rewrite(sp.gamma)
        elif isinstance(atom, sp.Product) and len(atom.limits) == 1:
            index, low, high = atom.limits[0]
            if (
                not low.is_Integer
                or not high.is_Integer
                or not 0 <= high - low + 1 <= 32
            ):
                continue
            value = sp.Mul(
                *(atom.function.subs(index, j) for j in range(int(low), int(high) + 1))
            )
        elif atom.func is StepFactorialPower:
            base, order, step = args
            if order.is_Integer is not True or not -128 <= order < 0:
                continue
            if base.is_zero is not False and sp.Ne(base, 0) not in sp.And.make_args(
                assumptions
            ):
                continue
            value = 1 / sp.Mul(*(base + j * step for j in range(1, int(-order) + 1)))
        elif isinstance(atom, RealRoot):
            argument, order = args
            if order.is_Integer is not True or not 0 < order <= 16:
                continue
            negative = argument.is_negative is True or sp.Lt(
                argument, 0
            ) in sp.And.make_args(assumptions)
            positive = argument.is_positive is True or sp.Gt(
                argument, 0
            ) in sp.And.make_args(assumptions)
            if (
                not positive
                and len(variables) == len(target) == 1
                and target[0] is sp.oo
                and variables[0].is_real is not False
                and sp.count_ops(argument) <= 30
            ):
                variable = variables[0]
                tail = sp.Dummy("positive_root_tail", positive=True)
                chart = argument.subs(variable, tail)
                bessel_atoms = chart.atoms(sp.besseli)
                if bessel_atoms:
                    from .elementary_limit_germs import rational_value

                    certified = {}
                    for bessel in bessel_atoms:
                        degree, arg = bessel.args
                        if (
                            degree.is_Integer is True
                            and degree.is_nonnegative is True
                            and arg.is_rational_function(tail)
                            and rational_value(arg, tail, sp.oo) is sp.oo
                        ):
                            certified[bessel] = sp.Dummy(
                                "positive_bessel_value", positive=True
                            )
                    positive = chart.xreplace(certified).is_positive is True
                    if positive and len(certified) == len(bessel_atoms):
                        value = argument ** (sp.S.One / order)
                        replacements[atom] = value
                        continue
            if (
                not positive
                and len(variables) == len(target) == 1
                and sp.count_ops(argument) < 30
            ):
                variable, point = variables[0], target[0]
                numerator, denominator = sp.fraction(sp.cancel(argument))
                try:
                    num_poly, den_poly = (
                        sp.Poly(numerator, variable),
                        sp.Poly(denominator, variable),
                    )
                except sp.PolynomialError:
                    continue
                if max(num_poly.degree(), den_poly.degree()) > 2:
                    continue
                leading = num_poly.LC() / den_poly.LC()
                if point is sp.oo:
                    positive = leading.is_positive is True
                    negative = leading.is_negative is True
                elif point is -sp.oo:
                    signed_leading = leading * (-1) ** (
                        num_poly.degree() - den_poly.degree()
                    )
                    positive = signed_leading.is_positive is True
                    negative = signed_leading.is_negative is True
                else:
                    positive = argument.subs(variable, point).is_positive is True
                    negative = argument.subs(variable, point).is_negative is True
            real_chart = (
                argument.xreplace(
                    {
                        v: sp.Dummy(
                            "real_root_chart",
                            real=True,
                            nonzero=True if p in (sp.oo, -sp.oo) else None,
                        )
                        for v, p in zip(variables, target, strict=True)
                        if p.is_real is True or p in (sp.oo, -sp.oo)
                    }
                ).is_real
                is True
            )
            if not real_chart and all(v.is_real is True for v in variables):
                from .multivariate_pole_bounds import _locally_real

                real_chart = _locally_real(argument, variables, target)
            if positive and real_chart:
                value = argument ** (sp.S.One / order)
            elif negative and real_chart and order.is_odd is True:
                value = -((-argument) ** (sp.S.One / order))
            elif (
                order.is_odd is True
                and real_chart
                and all(p.is_real is True or p in (sp.oo, -sp.oo) for p in target)
            ):
                value = sp.sign(argument) * sp.Abs(argument) ** (sp.S.One / order)
            elif order.is_odd is True and argument.is_real is True:
                value = sp.real_root(argument, order)
            else:
                continue
        else:
            continue
        if value != atom:
            replacements[atom] = value
    reduced = expr_with_branches.xreplace(replacements)
    if reduced == expr:
        return expr, ()
    return reduced, (
        LimitEvidence(
            "local_function_normalization",
            "Finite products, real roots and locally small signed fractional parts use checked definitions. A real nonzero cotangent argument approaching infinity uses its reciprocal arctangent chart. Polynomial Piecewise selectors are fixed on the punctured local or infinite tail; isolated branch values cannot affect the limit.",
            tuple(replacements.items()),
            reduced,
        ),
    )


class InverseRegularizedGamma(sp.Function):
    """Principal inverse of the upper regularized gamma function in its argument."""

    nargs = 2

    @classmethod
    def eval(cls, order, value):
        if order == 1:
            return -sp.log(value)


def dawson(argument):
    """Dawson integral with represented floating-point constants made exact."""
    argument = sp.sympify(argument)
    argument = argument.xreplace(
        {value: sp.Rational(value) for value in argument.atoms(sp.Float)}
    )
    return sp.sqrt(sp.pi) * sp.exp(-(argument**2)) * sp.erfi(argument) / 2


def inverse_error_increment(center, increment):
    """Principal inverse error value after adding an increment to erf(center)."""
    return sp.erfinv(sp.erf(center) + increment)


def hypergeometric_1f1(a, b, argument):
    """Confluent hypergeometric function with one numerator and denominator parameter."""
    return sp.hyper((a,), (b,), argument)
