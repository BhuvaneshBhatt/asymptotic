"""Analytic function contracts and periodic discontinuity limits."""

import sympy as sp
from sympy.core.function import AppliedUndef

from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult


def normalize_direction(direction):
    """Normalize named real directions to outward real approach signs."""
    aliases = {"above": "+", "below": "-", "both": None}
    return aliases.get(direction, direction)


def analytic_limit(
    expr,
    x,
    point,
    *,
    direction=None,
    analytic_functions=(),
    assumptions=True,
    return_result=False,
):
    """Evaluate a limit with a call-scoped analytic contract for named functions.

    Finite image points and the variable approach domain must be admissible.
    Unspecified functions receive no continuity assumption. Value mode is the
    default; return_result=True exposes status, hypotheses and evidence."""
    from .limits import limit
    from .path_limits import _value_or_result, one_sided_limit

    expr, x, point = map(sp.sympify, (expr, x, point))
    direction = normalize_direction(direction)
    if direction not in (None, "+", "-"):
        raise ValueError("unsupported direction: " + str(direction))
    if not isinstance(x, sp.Symbol):
        raise TypeError("limit variable must be a symbol")
    if point.has(x):
        raise ValueError("limit target must be independent of the approach variable")
    clauses = sp.And.make_args(sp.sympify(assumptions))
    if (
        expr == sp.csc(sp.pi * x)
        and (point.is_integer is True or sp.Q.integer(point) in clauses)
        and x.is_integer is not True
        and x.is_real is not False
        and sp.sympify(assumptions) is not sp.S.false
    ):
        from .fixed_ray_branch_germs import DirectionalInfinity

        n = sp.Dummy("integer_pole_index", positive=True, integer=True)
        sides = (1,) if direction == "+" else (-1,) if direction == "-" else (1, -1)
        items = tuple(
            LimitEvidence(
                "integer_lattice_cosecant_pole",
                "The exact identity sin(pi*(k+h))=(-1)^k*sin(pi*h) gives a simple pole at an integer k. At h=+/-1/n, the denominator is nonzero for n>1 and each normalized direction is attained.",
                ((x, point + side / n),),
                DirectionalInfinity(side * (-1) ** point),
            )
            for side in sides
        )
        status = LimitStatus.PROVED if len(sides) == 1 else LimitStatus.DOES_NOT_EXIST
        result = SimultaneousLimitResult(
            expr,
            (x,),
            (point,),
            status,
            items[0].value if len(sides) == 1 else None,
            items,
            (x,),
            sp.S.true,
        )
        return _value_or_result(result, return_result)
    if (
        isinstance(point, sp.Symbol)
        and point.is_integer is not True
        and sp.Q.integer(point) in clauses
    ):
        from dataclasses import replace

        integer = sp.Dummy("integer_center", integer=True)
        inner = analytic_limit(
            expr.xreplace({point: integer}),
            x,
            integer,
            direction=direction,
            analytic_functions=analytic_functions,
            assumptions=sp.sympify(assumptions).xreplace({point: integer}),
            return_result=True,
        )
        back = {integer: point}
        evidence = tuple(
            replace(
                e,
                statement=e.statement.replace(str(integer), str(point)),
                value=e.value.xreplace(back)
                if isinstance(e.value, sp.Basic)
                else e.value,
                substitutions=tuple((v, q.xreplace(back)) for v, q in e.substitutions),
            )
            for e in inner.evidence
        )
        result = replace(
            inner,
            expression=expr,
            target=(point,),
            value=inner.value.xreplace(back)
            if isinstance(inner.value, sp.Basic)
            else inner.value,
            domain=inner.domain.xreplace(back),
            evidence=evidence,
        )
        return _value_or_result(result, return_result)
    chart_ok = x.is_zero is not True and x.is_finite is not False
    chart_ok &= not (x.is_real is True and point.is_real is False)
    if x.is_nonnegative is True:
        chart_ok &= point.is_nonnegative is True and not (
            point.is_zero is True and direction == "-"
        )
    if x.is_nonpositive is True:
        chart_ok &= point.is_nonpositive is True and not (
            point.is_zero is True and direction == "+"
        )
    if (
        chart_ok
        and isinstance(expr, AppliedUndef)
        and expr.func.__name__ in analytic_functions
        and len(expr.args) == 1
    ):
        arg = expr.args[0]
        value = None
        clauses = sp.And.make_args(sp.sympify(assumptions))
        if (
            assumptions is False
            or sp.sympify(assumptions) is sp.S.false
            or x.is_integer is True
            or x.is_real is False
            or point.has(x)
            or not (point.is_finite is True or sp.Q.finite(point) in clauses)
        ):
            return limit(
                expr, x, point, assumptions=assumptions, return_result=return_result
            )
        if arg == x:
            value = expr.func(point)
        elif arg == sp.real_root(x, 3) and sp.Q.real(point) in sp.And.make_args(
            sp.sympify(assumptions)
        ):
            value = sp.Piecewise(
                (expr.func(-((-point) ** sp.Rational(1, 3))), point < 0),
                (expr.func(point ** sp.Rational(1, 3)), True),
            )
        if value is not None:
            result = SimultaneousLimitResult(
                expr,
                (x,),
                (point,),
                LimitStatus.PROVED,
                value,
                (
                    LimitEvidence(
                        "analytic_function_contract",
                        "The caller declares this function analytic at the finite image point. The identity or real cube-root inner map is continuous there; the declaration is scoped to this call, and does not give any other undefined function continuity.",
                        value=value,
                    ),
                ),
                (x,),
                sp.S.true,
            )
            return _value_or_result(result, return_result)
    if point in (sp.oo, -sp.oo):
        return limit(
            expr, x, point, assumptions=assumptions, return_result=return_result
        )
    if direction in ("+", "-"):
        return one_sided_limit(
            expr,
            x,
            point,
            direction=direction,
            assumptions=assumptions,
            return_result=return_result,
        )
    if direction is not None:
        raise ValueError("unsupported direction: " + str(direction))
    return limit(expr, x, point, assumptions=assumptions, return_result=return_result)


class SquareWave(sp.Function):
    """Odd period-one wave, with zero values at its jumps."""

    nargs = (1, 2)

    @classmethod
    def eval(cls, *args):
        if len(args) == 2:
            heights, z = args
            if not isinstance(heights, sp.Tuple) or len(heights) != 2:
                return None
            lo, hi = heights
            return (lo + hi) / 2 + (hi - lo) * cls(z) / 2
        z = args[0]
        if z.is_Rational:
            if (2 * z).is_Integer:
                return sp.S.Zero
            return sp.S.One if z % 1 < sp.S.Half else -sp.S.One


def square_wave_certificate(expr, x, point, domain, assumptions):
    """Evaluate attained one-sided wave values on an affine real phase chart."""
    from .local_nonexistence_families import _sides
    from .special_function_limit_germs import linear_in

    if sp.count_ops(expr) > 40:
        return None
    atoms = expr.atoms(SquareWave)
    if len(atoms) != 1:
        return None
    sides = _sides(x, sp.sympify(point), domain, assumptions)
    if not sides:
        return None
    atom = next(iter(atoms))
    arg = atom.args[0]
    slope = sp.diff(arg, x)
    center = arg.subs(x, point)
    if (
        slope.has(x)
        or slope.is_real is not True
        or slope.is_finite is not True
        or slope.is_zero is not False
        or not center.is_Rational
    ):
        return None
    data = linear_in(expr, atom)
    if data is None or any(c.has(x) or c.is_finite is not True for c in data):
        return None
    a, b = data
    n = sp.Dummy("attained_wave_n", integer=True, positive=True)
    items = []
    for side in sides:
        phase = center % 1
        if phase in (0, sp.S.Half):
            above = (side * slope).is_positive
            wave = (1 if above else -1) if phase == 0 else (-1 if above else 1)
        else:
            wave = 1 if phase < sp.S.Half else -1
        value = a * wave + b
        items.append(
            LimitEvidence(
                "attained_square_wave_sides",
                "The exact period-one definition has jumps at integers and half-integers. A nonzero real affine slope determines the side of the jump. x=point+/-1/n eventually stays in the relevant open half-period and avoids every discontinuity.",
                ((x, point + side / n),),
                value,
            )
        )
    if all(i.value == items[0].value for i in items):
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
