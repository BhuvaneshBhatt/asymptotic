"""Regular parameter strata for step factorials and integral tails."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult
from .local_tail_germs import fixed_finite
from .reference_normalization import Normal, Series
from .step_factorials import StepFactorialPower
from .stratification import AsymptoticStratification, ParameterStratum


def special_parameter_limit(expr, x, point, domain, assumptions):
    """Return checked regular regimes and retain every unresolved complement."""
    if (
        domain is not sp.S.true
        or assumptions is sp.S.false
        or (point.is_finite is True and x.is_integer is True)
        or sp.count_ops(expr) > 90
    ):
        return None

    def result(value, statement, status=LimitStatus.PROVED, evidence=None):
        return SimultaneousLimitResult(
            expr,
            (x,),
            (point,),
            status,
            value,
            evidence
            or (
                LimitEvidence("step_factorial_parameter_germ", statement, value=value),
            ),
            (x,),
            domain,
        )

    def partition(condition, value, statement, singular=None):
        from ._symbolic_policy import bounded_ask

        selected = condition is sp.S.true or condition in sp.And.make_args(assumptions)
        if not selected:
            selected = bounded_ask(condition, assumptions) is True
        regular = result(value, statement)
        if selected:
            return regular
        complement = sp.Not(condition)
        other = singular or result(
            None,
            "This parameter complement needs a separate pole, branch or zero-step theorem.",
            LimitStatus.UNKNOWN,
        )
        if condition is sp.S.false:
            return other
        return AsymptoticStratification(
            tuple(sorted(expr.free_symbols - {x}, key=sp.default_sort_key)),
            (ParameterStratum(condition, regular), ParameterStratum(complement, other)),
            assumptions=assumptions,
            exhaustive=True,
        )

    if point == sp.oo and x.is_real is not False and expr.has(sp.Ei):
        atoms = expr.atoms(sp.Ei)
        if len(atoms) == 2 and sp.Ei(x) in atoms:
            other = next(a for a in atoms if a != sp.Ei(x))
            slope = sp.diff(other.args[0], x)
            s = 1 - slope
            if (
                isinstance(s, sp.Symbol)
                and fixed_finite(s)
                and s.is_real is not False
                and other.args[0].expand() == ((1 - s) * x).expand()
            ):
                expected = (sp.exp(s * x) * other - sp.Ei(x)) * sp.exp(-s * x) / s
                if expr == expected and (s > 1) in sp.And.make_args(assumptions):
                    return result(
                        sp.S.Zero,
                        "For fixed real s>1, Ei((1-s)*x) decays on the negative real axis; exp(-s*x)*Ei(x)=O(exp(-(s-1)*x)/x). The nonzero constant s preserves zero.",
                    )

    if expr.func is StepFactorialPower and expr.args[1].is_Integer:
        base, order, step = expr.args
        if (
            step == x
            and point == 0
            and not base.has(x)
            and order.is_Integer
            and -128 <= order < 0
            and fixed_finite(base)
        ):
            k = int(-order)
            index = sp.Dummy("step_index", positive=True, integer=True)
            if k % 2 and x.is_positive is not True and x.is_negative is not True:
                witnesses = tuple(
                    LimitEvidence(
                        "attained_zero_base_step_pole",
                        "At base zero the product is k!*step**k. The two nonzero sequences avoid every denominator zero and attain opposite real infinities.",
                        ((x, side / index),),
                        side * sp.oo,
                    )
                    for side in (1, -1)
                )
                singular = result(
                    None,
                    "Opposite attained pole directions.",
                    LimitStatus.DOES_NOT_EXIST,
                    witnesses,
                )
            else:
                singular = result(
                    -sp.oo if k % 2 and x.is_negative is True else sp.oo,
                    "At base zero the reciprocal product is 1/(k!*step**k); even k gives positive divergence on both real sides; an odd power keeps the sign of the declared one-sided coordinate.",
                )
            if x.is_real is False:
                return None
            return partition(
                sp.Ne(base, 0),
                base**order,
                "For a fixed nonzero base, every factor stays nonzero near step zero and the finite reciprocal product tends to base**order.",
                singular,
            )
    if expr.func is StepFactorialPower and expr.args[1].is_Integer is not True:
        base, order, step = expr.args
        if (
            order == x
            and point == 0
            and not (base.has(x) or step.has(x))
            and fixed_finite(base)
            and fixed_finite(step)
        ):
            if base.is_real is False or step.is_real is False:
                return None
            condition = sp.And(sp.Q.real(base), base >= 0, step > 0)
            if base.is_nonnegative is True and step.is_positive is True:
                condition = sp.S.true
            return partition(
                condition,
                sp.S.One,
                "On base>=0 and step>0, the principal gamma quotient (1/step)**(-order)*gamma(1+base/step)/gamma(1+base/step-order) is jointly analytic near order zero and equals one there.",
            )
        if (
            base == x
            and point == 0
            and not (order.has(x) or step.has(x))
            and fixed_finite(order)
            and fixed_finite(step)
        ):
            return partition(
                sp.Ne(step, 0),
                1 / ((1 / step) ** order * sp.gamma(1 - order)),
                "For nonzero fixed step, the numerator gamma has regular center one. Reciprocal gamma is entire, including its zeros at nonpositive integers, so the principal constant power times the gamma quotient has the stated continuous limit.",
            )
    if expr.func is Normal:
        series = expr.args[0]
        if series.func is Series:
            function, specification = series.args
            if isinstance(specification, sp.Tuple) and len(specification) == 3:
                base, center, cutoff = specification
                if (
                    function.func is StepFactorialPower
                    and function.args == (base, x, sp.S.One)
                    and isinstance(base, sp.Symbol)
                    and center == 0
                    and cutoff.is_Integer
                    and 1 <= cutoff <= 9
                    and point.is_Integer
                    and 0 <= point <= 8
                    and base != x
                    and fixed_finite(base)
                ):
                    polynomial = sp.Poly(
                        sp.prod(base - j for j in range(int(point))), base
                    )
                    value = sp.Add(
                        *(
                            c * base ** m[0]
                            for m, c in polynomial.terms()
                            if m[0] < cutoff
                        )
                    )
                    return result(
                        value,
                        "Near base zero and integer order, gamma(1+base)/gamma(1+base-order) is jointly analytic: the numerator is regular and reciprocal gamma is entire. Each finite Taylor coefficient is continuous in order, so truncation commutes with specialization to the exact falling-factorial polynomial.",
                    )
    return None
