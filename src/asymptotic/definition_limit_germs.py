"""Local limits from exact definitions and controlled Taylor remainders."""

import sympy as sp

from .elementary_limit_germs import answer, rational_value
from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .local_tail_germs import budget


def axis_argument_certificate(expr, x, point, domain, assumptions):
    """Resolve a floor of a constant argument on a punctured real axis germ."""
    if expr.func is not sp.floor or not budget(expr, 35):
        return None
    phase = expr.args[0]
    if phase.func is not sp.arg:
        return None
    sides = _sides(x, point, domain, assumptions)
    if not sides:
        return None
    t = sp.Dummy("axis_distance", positive=True)
    values = []
    for side in sides:
        chart = phase.args[0].subs(x, point + side * t)
        real, imag = sp.expand_complex(chart).as_real_imag()
        if real != 0 or imag.is_real is not True or not imag.is_algebraic_expr(t):
            return None
        lead = imag.as_leading_term(t)
        coefficient, _ = lead.as_coeff_exponent(t)
        if coefficient.is_positive is True:
            values.append(sp.floor(sp.pi / 2))
        elif coefficient.is_negative is True:
            values.append(sp.floor(-sp.pi / 2))
        else:
            return None
    if len(set(values)) != 1:
        return None
    return answer(
        values[0],
        "punctured_axis_argument",
        "The real part vanishes identically. The leading imaginary coefficient has a fixed nonzero sign on every requested real side, so the argument is exactly +/-pi/2 throughout the punctured germ. Its floor is constant; the zero value at the endpoint is excluded.",
    )


def logarithmic_ei_certificate(expr, x, point, domain, assumptions):
    """Separate the two real sides of a logarithmic Ei cancellation."""
    if point != 0:
        return None
    oscillatory = expr == -(x**sp.I) / sp.log(x) + sp.I * sp.Ei(sp.I * sp.log(x))
    if not oscillatory and expr != sp.log(x) - sp.Ei(sp.log(1 - x)):
        return None
    sides = _sides(x, point, domain, assumptions)
    if not sides:
        return None
    if oscillatory:
        return answer(
            sp.pi,
            "logarithmic_ei_vertical_tail",
            "For each real side write L=log(abs(x))->-infinity. The principal logarithm is L or L+i*pi, so i*Log(x) has bounded real part and imaginary part tending to negative infinity. The signed Ei continuation tends to -i*pi with error O(1/abs(L)); x**i is bounded on each side and its division by Log(x) vanishes. These two attained sides both give pi, avoiding the endpoint and the Ei cut.",
        )
    n = sp.Dummy("branch_index", positive=True, integer=True)
    items = tuple(
        LimitEvidence(
            "logarithmic_ei_real_sides",
            "The convergent Ei germ is EulerGamma+log(abs(z))+O(z) on the real axis. log(1-x)=-x+O(x**2). On x>0 the real Ei boundary cancels log(x); on x<0 the principal log(x) adds i*pi. The sequences +/-1/n avoid both logarithmic endpoints and attain these values.",
            ((x, side / n),),
            -sp.EulerGamma + (sp.I * sp.pi if side < 0 else 0),
        )
        for side in sides
    )
    if len({item.value for item in items}) > 1:
        return LimitStatus.DOES_NOT_EXIST, None, items
    return LimitStatus.PROVED, items[0].value, items


def local_integral_certificate(expr, x, point, domain, assumptions):
    """Plan the cancellation order before expanding a local definite integral."""
    if (
        not expr.has(sp.Integral)
        or not budget(expr, 65)
        or not _sides(x, point, domain, assumptions)
    ):
        return None
    atoms = expr.atoms(sp.Integral)
    if len(atoms) != 1 or expr.free_symbols - {x}:
        return None
    atom = next(iter(atoms))
    if len(atom.limits) != 1 or len(atom.limits[0]) != 3:
        return None
    variable, lower, upper = atom.limits[0]
    if lower != x or upper != point or variable == x:
        return None
    integrand = atom.function
    if any(
        f.func not in (sp.sin, sp.cos, sp.exp, sp.Integral)
        for f in expr.atoms(sp.Function)
    ):
        return None
    if any(p.base.has(x) and p.exp.is_Integer is not True for p in expr.atoms(sp.Pow)):
        return None
    if integrand.subs(variable, point).is_finite is not True:
        return None
    # These entire heads guarantee a uniform Taylor remainder on a small disc.
    if any(
        f.func not in (sp.sin, sp.cos, sp.exp) for f in integrand.atoms(sp.Function)
    ):
        return None
    if any(
        p.exp.is_Integer is not True
        or (
            p.exp.is_negative is True
            and p.base.subs(variable, point).is_zero is not False
        )
        for p in integrand.atoms(sp.Pow)
        if p.base.has(variable)
    ):
        return None
    marker = sp.Dummy("integral_value")
    polynomial = expr.xreplace({atom: marker})
    if sp.diff(polynomial, marker, 2) != 0:
        return None
    coefficient = sp.diff(polynomial, marker)
    t = sp.Dummy("local_distance", positive=True)
    coefficient = coefficient.subs(x, point + t)
    leading = coefficient.as_leading_term(t)
    factor, order = leading.as_coeff_exponent(t)
    if not order.is_Integer or not -6 <= order <= 6 or factor.is_finite is not True:
        return None
    terms = max(1, int(-order))
    model = -sum(
        sp.diff(integrand, variable, k).subs(variable, point)
        * (x - point) ** (k + 1)
        / sp.factorial(k + 1)
        for k in range(terms)
    )
    chart = expr.xreplace({atom: model}).subs(x, point + t)
    series = chart.series(t, 0, 1).removeO()
    value = rational_value(series, t, 0)
    if value is None or value.is_finite is not True:
        return None
    return answer(
        value,
        "local_integral_taylor_cancellation",
        f"The entire integrand gives an integral Taylor polynomial through degree {terms} with error O((x-point)**{terms + 1}). The multiplying coefficient has order {order}, so the error vanishes after cancellation. Analytic coefficient denominators have an isolated zero of the checked finite order and are nonzero on a sufficiently small punctured disc. The same constant applies to both requested real sides.",
    )


def stirling_remainder_certificate(expr, x, point):
    """Retain the first surviving term in two standard Stirling remainders."""
    if (
        point is not sp.oo
        or not budget(expr, 180)
        or not expr.has(sp.gamma, sp.factorial)
        or not expr.has(sp.exp)
        or not any(p.base == x and p.exp.has(x) for p in expr.atoms(sp.Pow))
    ):
        return None
    scale = sp.sqrt(2 * sp.pi) * x ** (x + sp.S.Half) * sp.exp(-x)
    linear = sp.gamma(x + 1) / sp.sqrt(2 * sp.pi) - (
        x ** (x + sp.S.Half) + x ** (x - sp.S.Half) / 12
    ) * sp.exp(-x)
    if sp.expand_mul(expr) == sp.expand_mul(linear):
        return answer(
            sp.oo,
            "stirling_surviving_remainder",
            "Gamma(x+1)=S(x)*(1+1/(12*x)+1/(288*x**2)+O(x**-3)) with S=sqrt(2*pi)*x**(x+1/2)*exp(-x)>0. Exact subtraction leaves S/(288*sqrt(2*pi)*x**2)*(1+O(1/x)), which tends to positive infinity.",
        )
    difference = sp.exp(x) * scale * sp.exp(1 / (12 * x)) - sp.factorial(x) * sp.exp(x)
    derivative = sp.diff(difference**2, x) / sp.diff(sp.exp(2 * x), x)
    if expr != derivative:
        return None
    return answer(
        sp.oo,
        "differentiated_stirling_remainder",
        "Write the difference as exp(x)*S(x)*h(x). The log-Gamma expansion gives h=1/(360*x**3)*(1+O(1/x)); the compatible digamma expansion gives h'=-1/(120*x**4)*(1+O(1/x)). The displayed derivative quotient is S**2*h*(h'+(1+log(x)+1/(2*x))*h), asymptotic to S**2*log(x)/(360**2*x**6)>0 and divergent. Differentiated errors are justified by the digamma expansion, not by differentiating an unspecified big-O term.",
    )


def divergent_integer_certificate(expr, x, point, domain, assumptions):
    """Transfer signed rational divergence through floor or ceiling."""
    if expr.func not in (sp.floor, sp.ceiling) or not budget(expr, 40):
        return None
    sides = _sides(x, point, domain, assumptions)
    if not sides:
        return None
    t = sp.Dummy("integer_distance", positive=True)
    n = sp.Dummy("integer_index", integer=True, positive=True)
    items = []
    for side in sides:
        argument = expr.args[0].subs(x, point + side * t)
        if argument.is_real is not True:
            return None
        value = rational_value(argument, t, 0)
        if value not in (sp.oo, -sp.oo):
            return None
        items.append(
            LimitEvidence(
                "rational_integer_divergence",
                "On each real side the rational argument has the recorded signed divergence. Floor and ceiling differ from a real argument by at most one, preserving that divergence. x=point+/-1/n is attained and eventually avoids every rational pole.",
                ((x, point + side / n),),
                value,
            )
        )
    if len({item.value for item in items}) > 1:
        return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
    return LimitStatus.PROVED, items[0].value, tuple(items)


def unresolved_gamma_parameter_certificate(expr, x, point, domain, assumptions):
    """Stop a parameter-dependent accumulating-pole family before general series."""
    if point is not sp.oo or assumptions is not sp.S.true or domain is not sp.S.true:
        return None
    atoms = expr.atoms(sp.gamma)
    if len(atoms) != 1 or sp.count_ops(expr) > 20:
        return None
    atom = next(iter(atoms))
    parameter = (2 - atom.args[0]) / x
    parameter = sp.cancel(parameter)
    if (
        not parameter.is_Symbol
        or parameter.is_real is True
        or parameter.is_zero is True
    ):
        return None
    if expr != sp.sin(x) * sp.gamma(2 - parameter * x) / x:
        return None
    return (
        LimitStatus.UNKNOWN,
        None,
        LimitEvidence(
            "gamma_tail_parameter_strata_required",
            "The unconstrained fixed parameter changes the gamma argument sector and may create accumulating real poles. The zero, real-sign and complex-sector parameter strata require separate attained-domain certificates. General series cannot settle this unstratified family.",
        ),
    )
