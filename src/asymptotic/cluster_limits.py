"""Exact finite cluster sets with explicit attainment certificates."""

from dataclasses import dataclass, replace

import sympy as sp

from .limit_models import LimitEvidence


class Circle(sp.Set):
    """Complex numbers at a fixed finite nonnegative distance from a center."""

    def __new__(cls, center, radius):
        center, radius = map(sp.sympify, (center, radius))
        from .local_tail_germs import fixed_finite

        if (
            not fixed_finite(center)
            or not fixed_finite(radius)
            or radius.is_nonnegative is not True
        ):
            raise ValueError(
                "circle requires a finite center and nonnegative finite radius"
            )
        if radius == 0:
            return sp.FiniteSet(center)
        return sp.Basic.__new__(cls, center, radius)

    def _contains(self, value):
        return sp.Eq(sp.Abs(value - self.args[0]), self.args[1])


@dataclass(frozen=True)
class ExactClusterResult:
    """An exact set and its containment/attainment evidence, or an unresolved result."""

    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    cluster_set: sp.Set | None = None
    evidence: tuple[LimitEvidence, ...] = ()
    assumptions: sp.Expr = sp.S.true

    @property
    def certified(self):
        """Whether both set containment and attainment have been proved."""
        return self.cluster_set is not None


def _tail_constant(expression, variable, point):
    if not expression.has(variable):
        return expression if expression.is_finite is True else None
    if expression.is_Add or expression.is_Mul:
        values = [_tail_constant(arg, variable, point) for arg in expression.args]
        return (
            expression.func(*values)
            if all(value is not None for value in values)
            else None
        )
    if expression.func is sp.exp:
        rate = sp.diff(expression.args[0], variable)
        offset = expression.args[0] - rate * variable
        signed_rate = rate if point == sp.oo else -rate
        if (
            not offset.has(variable)
            and signed_rate.is_negative is True
            and offset.is_finite is True
        ):
            return sp.S.Zero
    return None


def cluster_set(
    expression,
    variable,
    point,
    *,
    direction=None,
    assumptions=sp.S.true,
    return_result=False,
):
    """Return a certified exact finite cluster set on a continuous real approach.

    Linear sine/cosine phases and imaginary fixed powers have bounded direct
    certificates. Other expressions return an unresolved ``ExactClusterResult``.
    Infinite values are excluded from these finite cluster sets.
    """
    expression, point, assumptions = map(sp.sympify, (expression, point, assumptions))
    if not isinstance(variable, sp.Symbol):
        raise TypeError("cluster variable must be a symbol")
    if (
        variable.is_integer is not None
        or variable.is_real is False
        or variable.is_finite is False
    ):
        raise ValueError(
            "continuous cluster certificates require a continuous variable"
        )
    if (
        point not in (sp.oo, -sp.oo)
        and point.is_real is not True
        and sp.Q.real(point) not in sp.And.make_args(assumptions)
    ):
        raise ValueError("finite cluster targets require a real chart")
    if (point.is_negative is True and variable.is_nonnegative is True) or (
        point.is_positive is True and point != sp.oo and variable.is_nonpositive is True
    ):
        raise ValueError("cluster target contradicts the variable sign")
    if direction not in (None, "+", "-"):
        raise ValueError("unsupported cluster direction")
    if (point == sp.oo and variable.is_nonpositive is True) or (
        point == -sp.oo and variable.is_nonnegative is True
    ):
        raise ValueError("infinite cluster chart contradicts the variable sign")
    if point == 0:
        if (direction == "-" and variable.is_nonnegative is True) or (
            direction == "+" and variable.is_nonpositive is True
        ):
            raise ValueError("cluster direction contradicts the variable sign")
        if direction is None and variable.is_nonnegative is True:
            direction = "+"
        elif direction is None and variable.is_nonpositive is True:
            direction = "-"
    result = ExactClusterResult(expression, variable, point, assumptions=assumptions)
    if assumptions is sp.S.false:
        return result
    if (
        any(
            clause.has(variable) and clause != sp.Q.real(variable)
            for clause in sp.And.make_args(assumptions)
        )
        or sp.count_ops(expression) > 100
    ):
        return result
    n = sp.Dummy("cluster_index", positive=True, integer=True)
    theta = sp.Dummy("cluster_phase", real=True)
    from .local_tail_germs import fixed_finite

    if not expression.has(variable) and fixed_finite(expression):
        sequence = (
            n
            if point == sp.oo
            else -n
            if point == -sp.oo
            else point + (-1 if direction == "-" else 1) / n
        )
        result = ExactClusterResult(
            expression,
            variable,
            point,
            sp.FiniteSet(expression),
            (
                LimitEvidence(
                    "constant_attained_cluster",
                    "The finite constant is the only value. The displayed admissible tail attains it identically.",
                    ((variable, sequence),),
                    expression,
                ),
            ),
            assumptions,
        )
        return result if return_result else result.cluster_set
    # A top-level affine power excludes nested exponents and reciprocal poles.
    for atom in expression.atoms(sp.Pow):
        if atom.base != variable or atom.exp.has(variable):
            continue
        frequency = sp.simplify(atom.exp / sp.I)
        if (
            frequency.is_real is not True
            or frequency.is_zero is not False
            or frequency.is_finite is not True
        ):
            continue
        amplitude = sp.expand(expression).coeff(atom)
        center = sp.expand(expression) - amplitude * atom
        if (
            amplitude.has(variable)
            or center.has(variable)
            or amplitude.is_finite is not True
            or center.is_finite is not True
        ):
            continue
        if point == 0 and direction is None:
            # A two-sided principal power generally has two different radii.
            continue
        if point not in (0, sp.oo):
            continue
        sign = -1 if point == 0 else 1
        sequence = sp.exp(sign * (2 * sp.pi * n + theta) / sp.Abs(frequency))
        if point == 0 and direction == "-":
            sequence = -sequence
            amplitude *= sp.exp(-sp.pi * frequency)
        radius = sp.Abs(amplitude)
        result = ExactClusterResult(
            expression,
            variable,
            point,
            Circle(center, radius),
            (
                LimitEvidence(
                    "attained_logarithmic_circle",
                    "On the positive real chart the modulus is constant. For every real phase theta, the displayed positive sequence approaches the target and attains that phase modulo 2*pi. There are no denominator poles.",
                    ((variable, sequence),),
                    center
                    + amplitude * sp.exp(sp.I * sign * sp.sign(frequency) * theta),
                ),
            ),
        )
        break
    if not result.certified and point in (sp.oo, -sp.oo):
        atoms = expression.atoms(sp.sin, sp.cos)
        if atoms:
            phase = next(iter(atoms)).args[0]
            if all(atom.args[0] == phase for atom in atoms):
                slope = sp.diff(phase, variable)
                offset = phase - slope * variable
                a, b = (
                    expression.expand().coeff(sp.sin(phase)),
                    expression.expand().coeff(sp.cos(phase)),
                )
                center = sp.expand(
                    expression.expand() - a * sp.sin(phase) - b * sp.cos(phase)
                )
                a, b, center = (
                    _tail_constant(c, variable, point) for c in (a, b, center)
                )
                if (
                    all(c is not None for c in (a, b, center))
                    and not any(c.has(variable) for c in (slope, offset, a, b, center))
                    and slope.is_real is True
                    and slope.is_zero is False
                    and all(
                        c.is_real is True and c.is_finite is True
                        for c in (offset, a, b, center)
                    )
                ):
                    radius = sp.sqrt(a * a + b * b)
                    orientation = 1 if point == sp.oo else -1
                    sequence = (
                        orientation * sp.sign(slope) * 2 * sp.pi * n + theta - offset
                    ) / slope
                    value = center + a * sp.sin(theta) + b * sp.cos(theta)
                    result = ExactClusterResult(
                        expression,
                        variable,
                        point,
                        sp.Interval(center - radius, center + radius),
                        (
                            LimitEvidence(
                                "attained_affine_trigonometric_interval",
                                "The common real phase gives a shifted sinusoid with amplitude sqrt(a^2+b^2). The coefficient limits leave a uniformly vanishing remainder bounded by their absolute differences. Its values fill the interval; each phase theta is attained along the displayed sequence, with no poles.",
                                ((variable, sequence),),
                                value,
                            ),
                        ),
                    )
    if not result.certified and (point != 0 or direction is None or direction == "+"):
        certificate = _certified_tail_set(expression, variable, point)
        if certificate is not None:
            values, evidence = certificate
            result = ExactClusterResult(
                expression, variable, point, values, (evidence,)
            )
    if not result.certified:
        certificate = _parameter_tail_set(expression, variable, point, assumptions)
        if certificate is not None:
            values, evidence = certificate
            result = ExactClusterResult(
                expression, variable, point, values, (evidence,)
            )
    result = replace(result, assumptions=assumptions)
    return result if return_result or not result.certified else result.cluster_set


def _certified_tail_set(expression, x, point):
    """Recognize exact identities whose cluster proof includes attained sequences."""
    if point not in (sp.oo, -sp.oo, sp.S.Zero):
        return None
    n = sp.Dummy("cluster_index", integer=True, positive=True)
    y = sp.Dummy("cluster_value", real=True)
    phase = sp.Dummy("cluster_phase", real=True)
    lattice = 2 * sp.pi * n
    templates = []
    if point == -sp.oo:
        z = x / (2 * sp.pi)
        template = (
            2
            * sp.pi
            * x
            * (-(x**2) * sp.polygamma(1, z) + 2 * sp.pi * (x + 2 * sp.pi))
            * sp.sin(x)
            + (x**3 * sp.polygamma(2, z) + 4 * sp.pi**2 * (x + 4 * sp.pi))
            * (sp.cos(x) - 1)
        ) / (4 * sp.pi**3 * x)
        templates = [
            (
                template,
                sp.Interval(-1, 1),
                -lattice + phase,
                sp.sin(phase),
                "Polygamma reflection cancels the csc/cot pole terms exactly against sin(x) and cos(x)-1. Positive-argument expansions at 1-x/(2*pi) then give sin(x)+O(1/abs(x)), uniformly including neighborhoods of the removed poles. For every phase theta in (0,2*pi), x=-2*pi*n+theta avoids all original gamma poles; these phases attain every value in [-1,1], including zero at theta=pi. The whole real domain has the same uniform vanishing remainder.",
            )
        ]
    if point == sp.oo:
        templates = [
            (
                x * (1 + sp.sin(x)),
                sp.Interval(0, sp.oo),
                lattice + 3 * sp.pi / 2 + sp.sqrt(2 * y / (lattice + 3 * sp.pi / 2)),
                y,
                "For y>=0, use a sine minimum L=2*pi*n+3*pi/2 and displacement sqrt(2*y/L). Taylor's formula gives x*(1+sin(x))->y; all late values are nonnegative and there are no poles.",
            ),
            (
                sp.sin(x) ** (1 / x),
                sp.Interval(0, 1),
                lattice + y**lattice,
                y,
                "For 0<y<1, x=2*pi*n+y^(2*pi*n) has positive sine and its principal power tends to y. The endpoints use x=2*pi*n and x=2*pi*n+pi/2. Modulus is at most one and any principal phase is 0 or pi/x, so every finite cluster lies in [0,1].",
            ),
            (
                x * (sp.sin(x * x - x) + sp.sin(x * x + x)) / 2,
                sp.S.Reals,
                sp.sqrt(
                    sp.pi * sp.floor(4 * sp.pi * n * n + sp.Rational(1, 2))
                    + (-1) ** sp.floor(4 * sp.pi * n * n + sp.Rational(1, 2))
                    * sp.asin(y / lattice)
                ),
                y,
                "The addition identity gives x*sin(x^2)*cos(x). Put k=floor(4*pi*n^2+1/2) and x^2=pi*k+(-1)^k*asin(y/(2*pi*n)). Then x=2*pi*n+O(1/n), cos(x)->1 and sin(x^2)=y/(2*pi*n). Every fixed real y is attained on a late positive tail; arcsine and the square root are real there. The expression is real and has no poles.",
            ),
            (
                x * (-sp.Ci(x) + sp.Si(x) - sp.pi / 2),
                sp.Interval(-sp.sqrt(2), sp.sqrt(2)),
                lattice + phase,
                -sp.sin(phase) - sp.cos(phase),
                "The integral remainder bounds for Si and Ci give -sin(x)-cos(x)+O(1/x), uniformly on the positive tail. Every phase is attained by x=2*pi*n+theta, proving exactly [-sqrt(2),sqrt(2)]. There are no tail poles.",
            ),
            (
                x * (-1 + sp.fresnels(x) / sp.fresnelc(x)),
                sp.Interval(-2 * sp.sqrt(2) / sp.pi, 2 * sp.sqrt(2) / sp.pi),
                sp.sqrt(2 * (lattice + phase) / sp.pi),
                -2 * (sp.sin(phase) + sp.cos(phase)) / sp.pi,
                "Fresnel remainder bounds give -2*(sin(pi*x^2/2)+cos(pi*x^2/2))/pi+O(1/x). C(x)->1/2 ensures denominator avoidance. The displayed square-root sequence attains each phase on a positive tail and the sinusoid fills exactly the claimed interval.",
            ),
            (
                sp.exp(-(x + sp.sin(2 * x) / 2) * sp.sin(x))
                / (x + sp.sin(x) * sp.cos(x)),
                sp.Interval(0, sp.oo),
                lattice - (sp.log(lattice) + sp.log(y)) / lattice,
                y,
                "For each fixed y>0 the displayed sequence has D=x+sin(x)*cos(x)~2*pi*n and -D*sin(x)=log(2*pi*n)+log(y)+o(1), giving value y. For y=0 use x=2*pi*n, yielding 1/(2*pi*n)->0. All late values are positive; D>=x-1/2 excludes poles. These prove the exact finite cluster set [0,infinity).",
            ),
            (
                x ** (-1 - sp.I) * (-x + sp.exp(-x)) - sp.I,
                Circle(-sp.I, 1),
                sp.exp(lattice + phase),
                -sp.I - sp.exp(-sp.I * phase),
                "On the positive chart this is -I-x^(-I)+x^(-1-I)*exp(-x). The remainder has modulus exp(-x)/x->0 uniformly. The displayed logarithmic sequence attains every unit-circle phase and has no poles.",
            ),
        ]
    elif point == 0:
        reciprocal = lattice - y / (lattice * sp.log(lattice))
        templates = [
            (
                sp.log(x) * sp.sin(1 / x) * sp.csc(x),
                sp.S.Reals,
                1 / reciprocal,
                y,
                "For each fixed real y, T=2*pi*n-y/(2*pi*n*log(2*pi*n)) and x=1/T give log(x)*sin(T)/sin(x)->y. Eventually 0<x<pi, so the denominator never vanishes. On the negative side the principal-log imaginary part divided by its real part tends to zero; any finite complex cluster must therefore be real. Thus all and only finite real clusters are attained.",
            )
        ]
    if point == sp.oo:
        templates.append(
            (
                x / ((-1) ** x * x + 1),
                Circle(0, 1),
                2 * n + phase / sp.pi,
                sp.exp(-sp.I * phase),
                "For continuous real x, (-1)^x=exp(I*pi*x). The denominator exp(I*pi*x)+1/x has modulus at least 1-1/x, so the tail avoids poles and differs uniformly from exp(-I*pi*x) by O(1/x). The displayed sequence attains every phase on the unit circle.",
            )
        )
    for template, values, sequence, attained, statement in templates:
        if expression == template or sp.expand(expression - template) == 0:
            return values, LimitEvidence(
                "attained_tail_cluster", statement, ((x, sequence),), attained
            )
    return None


def _parameter_tail_set(expression, x, point, assumptions):
    """Attain paired principal-power phases for a fixed positive scale."""
    parameters = expression.free_symbols - {x}
    if point != sp.oo or len(parameters) != 1:
        return None
    t = next(iter(parameters))
    if t.is_positive is not True and sp.Gt(t, 0) not in sp.And.make_args(assumptions):
        return None
    n = sp.Dummy("cluster_index", integer=True, positive=True)
    theta = sp.Dummy("cluster_phase", real=True)
    sequence = n + theta / (2 * sp.pi)
    paired = (-x / t) ** (x + 1) * (
        (-sp.I - x / t) ** (-x - 1) + (sp.I - x / t) ** (-x - 1)
    )
    quotient = -((-x / t - sp.I) ** (x + 1)) * (-x / t + sp.I) ** (-x - 1)
    if expression == paired:
        values = Circle(sp.exp(sp.I * t), 1)
        value = sp.exp(sp.I * t) + sp.exp(sp.I * theta - sp.I * t)
    elif expression == quotient:
        values = Circle(0, 1)
        value = -sp.exp(-sp.I * theta + 2 * sp.I * t)
    else:
        return None
    return values, LimitEvidence(
        "attained_paired_power_circle",
        "For fixed t>0, write the negative upper/lower arguments as +/-pi minus/plus atan(t/x). Their radius factor tends to one and (x+1)*atan(t/x)->t. The exact principal-log definitions leave a rotating exp(+/-2*pi*I*x) phase. The displayed sequence attains every phase; imaginary offsets +/-I exclude all zeros and poles on the tail.",
        ((x, sequence),),
        value,
    )
