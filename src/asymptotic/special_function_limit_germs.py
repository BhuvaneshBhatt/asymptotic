"""Signed special-function germs with controlled errors and attained witnesses.

Sources: DLMF 6.4/6.12, 7.12, 9.7, 10.17, 11.6, 12.9 and 4.13.
"""

import sympy as sp

from .elementary_limit_germs import answer, rational_value
from .limit_models import LimitEvidence, LimitStatus
from .local_tail_germs import fixed_finite
from .special_functions import ParabolicCylinderD, StruveH, StruveL
from .tail_cancellation_germs import (
    clean,
    monomial_phase_subsequences,
    perturbation_budget,
)


def linear_in(expr, atom):
    w = sp.Dummy("special_function_value")
    try:
        p = sp.Poly(sp.expand_mul(expr.xreplace({atom: w})), w)
    except sp.PolynomialError:
        return None
    if p.degree() != 1:
        return None
    return p.coeff_monomial(w), p.coeff_monomial(1)


def affine_trig_subsequences(expr, x):
    if x.is_integer is True:
        return None
    atoms = expr.atoms(sp.sin, sp.cos)
    if not atoms or len(atoms) > 2 or not perturbation_budget(expr, 80):
        return None
    phase = next(iter(atoms)).args[0]
    if any(a.args[0] != phase for a in atoms):
        return None
    try:
        p = sp.Poly(phase, x)
    except sp.PolynomialError:
        return None
    if (
        p.degree() != 1
        or p.LC().is_positive is not True
        or not all(c.is_finite is True and c.is_real is True for c in p.all_coeffs())
    ):
        return None
    n = sp.Dummy("subsequence_n", integer=True, positive=True)
    witnesses = []
    for theta in (0, sp.pi, sp.pi / 2, 3 * sp.pi / 2):
        sequence = (2 * sp.pi * n + theta - p.nth(0)) / p.LC()
        replacements = {a: a.func(theta) for a in atoms}
        collapsed = sp.cancel(expr.xreplace(replacements))
        value = rational_value(collapsed, x, sp.oo)
        if value is None:
            continue
        ev = LimitEvidence(
            "attained_affine_special_tail_subsequence",
            "The affine real phase equals 2*pi*n+theta exactly. The sequence is positive on a sufficiently late tail and tends to infinity. The specialized rational denominator has finitely many zeros, eventually avoided.",
            ((x, sequence),),
            value,
        )
        for previous, evidence in witnesses:
            if sp.simplify(previous - value).is_zero is False:
                return LimitStatus.DOES_NOT_EXIST, None, (evidence, ev)
        witnesses.append((value, ev))
    return None


def transfer_tail(leading, x, method, statement):
    if sp.count_ops(leading) > 140:
        return None
    value = clean(leading, x, sp.oo)
    if value is not None:
        return answer(value, method, statement)
    from .gamma_exponential_germs import exponential_envelope_certificate

    certificate = exponential_envelope_certificate(leading, x, sp.oo)
    if certificate is None:
        certificate = affine_trig_subsequences(leading, x)
    if certificate is None:
        certificate = monomial_phase_subsequences(leading, x, sp.oo, sp.S.true)
    if certificate is None:
        return None
    status, value, evidence = certificate
    items = evidence if isinstance(evidence, tuple) else (evidence,)
    return status, value, (LimitEvidence(method, statement, value=value),) + items


def li_signed_boundary_certificate(expr, x, point, domain):
    """Select the logarithmic-integral branch from the signed local approach."""
    atoms = expr.atoms(sp.li)
    if (
        point != 0
        or domain is not sp.S.true
        or len(atoms) != 1
        or not perturbation_budget(expr, 70)
        or x.is_real is False
        or x.is_integer is True
        or x.is_finite is False
    ):
        return None
    atom = next(iter(atoms))
    argument = atom.args[0]
    try:
        p = sp.Poly(argument, x)
    except sp.PolynomialError:
        return None
    if p.degree() != 1 or p.nth(0).is_negative is not True or not fixed_finite(p.LC()):
        return None
    slope = sp.simplify(sp.im(p.LC()))
    if slope.is_positive is not True and slope.is_negative is not True:
        return None
    r = -p.nth(0)
    if r.is_finite is not True:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = (rational_value(v, x, 0) for v in data)
    if a is None or b is None or not fixed_finite(a) or not fixed_finite(b):
        return None
    if a == 0:
        return answer(
            b,
            "li_bounded_cut_boundary",
            "Both signed li boundary values at a fixed negative argument are finite, and their multiplying coefficient vanishes.",
        )
    if a.is_zero is not False:
        return None
    orientation = 1 if slope.is_positive is True else -1
    n = sp.Dummy("subsequence_n", integer=True, positive=True)
    evidence = []
    signs = (
        (1,)
        if x.is_nonnegative is True
        else (-1,)
        if x.is_nonpositive is True
        else (1, -1)
    )
    for sign in signs:
        value = a * sp.Ei(sp.log(r) + sign * orientation * sp.I * sp.pi) + b
        evidence.append(
            LimitEvidence(
                "attained_li_signed_cut_subsequence",
                "x=+/-1/n approaches the negative argument strictly from the specified half-plane, avoiding 0, 1 and the cuts. li(z)=Ei(Log(z)) has boundary Ei(log(r)+/-i*pi). Its upper imaginary part equals pi+integral_0^r pi/(log(t)^2+pi^2) dt>pi; the lower value is its conjugate, so a nonzero coefficient preserves their difference.",
                ((x, sign / n),),
                value,
            )
        )
    if len(evidence) == 1:
        return LimitStatus.PROVED, evidence[0].value, tuple(evidence)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence)


def erfi_sector_certificate(expr, x, point):
    """Apply an imaginary-error-function germ only in a proved supported sector."""
    atoms = expr.atoms(sp.erfi)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 80)
    ):
        return None
    atom = next(iter(atoms))
    c = sp.simplify(atom.args[0] / x)
    if c.has(x) or c.is_number is not True or c.is_finite is not True:
        return None
    c = sp.expand_complex(c)
    im = sp.simplify(sp.im(c))
    growth = sp.simplify(sp.re(c * c))
    if (
        im.is_positive is not True and im.is_negative is not True
    ) or growth.is_nonpositive is not True:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    if clean(sp.Abs(a) * sp.exp(growth * x * x) / x, x, point) != 0:
        return None
    value = clean(a * (sp.I if im.is_positive is True else -sp.I) + b, x, point)
    if value is None:
        return None
    return answer(
        value,
        "erfi_closed_decay_sector",
        "For fixed c with nonzero imaginary part and Re(c^2)<=0, erfi(c*x)=sign(Im(c))*i+O(exp(Re(c^2)*x^2)/x). Rotation to erf has argument in the closed +/-pi/4 sector; the amplified DLMF 7.12 tail bound vanishes, including its oscillatory boundary.",
    )


def oscillatory_integral_tail_certificate(expr, x, point):
    """Transfer a controlled oscillatory integral tail through a checked scale."""
    atoms = expr.atoms(sp.Si, sp.fresnels, sp.fresnelc)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 90)
    ):
        return None
    atom = next(iter(atoms))
    u = atom.args[0]
    if (
        not u.is_rational_function(x)
        or u.is_positive is not True
        or clean(u, x, point) is not sp.oo
    ):
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    if atom.func is sp.Si:
        approximation = sp.pi / 2 - sp.cos(u) / u - sp.sin(u) / u**2
        error = 1 / u**3
    else:
        phase = sp.pi * u * u / 2
        error = 1 / u**5
        approximation = (
            sp.Rational(1, 2)
            - sp.cos(phase) / (sp.pi * u)
            - sp.sin(phase) / (sp.pi**2 * u**3)
            if atom.func is sp.fresnels
            else sp.Rational(1, 2)
            + sp.sin(phase) / (sp.pi * u)
            - sp.cos(phase) / (sp.pi**2 * u**3)
        )
    if clean(sp.Abs(a) * error, x, point) != 0:
        return None
    leading = sp.expand(a * approximation + b)
    return transfer_tail(
        leading,
        x,
        "oscillatory_integral_tail_remainder",
        "Real positive Si/Fresnel integration-by-parts tails retain two oscillatory terms. The amplified remainder is bounded by a constant times u^-3 (Si) or u^-5 (Fresnel) and tends to zero; attained subsequences of the retained expression therefore transfer to the original function.",
    )


def fresnel_signed_pole_certificate(expr, x, point, domain):
    """Resolve a signed Fresnel argument pole without discarding an amplified remainder."""
    if (
        expr.func not in (sp.fresnels, sp.fresnelc)
        or point.is_real is not True
        or point.is_finite is not True
        or domain is not sp.S.true
        or x.is_real is False
        or x.is_integer is True
        or x.is_finite is False
    ):
        return None
    if (x.is_nonnegative is True and point.is_negative is True) or (
        x.is_nonpositive is True and point.is_positive is True
    ):
        return None
    a = sp.cancel(expr.args[0] * (x - point))
    if (
        a.has(x)
        or a.is_real is not True
        or a.is_finite is not True
        or (a.is_positive is not True and a.is_negative is not True)
    ):
        return None
    signs = (
        (1,)
        if point == 0 and x.is_nonnegative is True
        else (-1,)
        if point == 0 and x.is_nonpositive is True
        else (1, -1)
    )
    n = sp.Dummy("subsequence_n", integer=True, positive=True)
    evidence = []
    for sign in signs:
        value = sp.Rational(sign, 2) * (1 if a.is_positive is True else -1)
        evidence.append(
            LimitEvidence(
                "attained_fresnel_signed_infinity",
                "x=point+/-1/n has nonzero denominator and sends the argument to the stated signed real infinity. Fresnel C and S are odd entire functions with limits +/-1/2, giving attained distinct boundary limits.",
                ((x, point + sign / n),),
                value,
            )
        )
    if len(evidence) == 1:
        return LimitStatus.PROVED, evidence[0].value, tuple(evidence)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence)


def fixed_bessel_tail_certificate(expr, x, point):
    """Use fixed-order Bessel tail bounds on a supported real approach."""
    atoms = expr.atoms(sp.besselj, sp.bessely, StruveH)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 90)
    ):
        return None
    atom = next(iter(atoms))
    nu, u = atom.args
    scale = sp.simplify(u / x)
    if (
        scale.has(x)
        or scale.is_number is not True
        or scale.is_positive is not True
        or scale.is_finite is not True
        or nu.is_number is not True
        or nu.is_real is not True
        or nu.is_finite is not True
        or abs(nu) > 16
    ):
        return None
    if atom.func is StruveH and nu != 0:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    if clean(sp.Abs(a) / u ** sp.Rational(3, 2), x, point) != 0:
        return None
    phase = u - nu * sp.pi / 2 - sp.pi / 4
    approximation = sp.sqrt(2 / (sp.pi * u)) * (
        sp.cos(phase) if atom.func is sp.besselj else sp.sin(phase)
    )
    if atom.func is StruveH:
        approximation += 2 / (sp.pi * u)
    return transfer_tail(
        sp.expand(a * approximation + b),
        x,
        "fixed_order_real_bessel_tail",
        "DLMF 10.17 gives the fixed real-order J/Y leading phase with error O(x^-3/2) on the positive ray. H_0-Y_0=2/(pi*x)+O(x^-3) by DLMF 11.6. Every multiplied error vanishes; signed phase subsequences or equal outward bounds establish the recorded result.",
    )


def spherical_tail_bound_certificate(expr, x, point):
    """Bound a fixed-order spherical Bessel tail on its checked argument chart."""
    atoms = expr.atoms(sp.jn, sp.yn)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not atoms
        or len(atoms) > 2
        or not perturbation_budget(expr, 70)
    ):
        return None
    replacements = {}
    symbols = []
    for atom in atoms:
        order, u = atom.args
        scale = sp.simplify(u / x)
        if (
            scale.has(x)
            or scale.is_number is not True
            or scale.is_positive is not True
            or scale.is_finite is not True
            or order.has(x)
            or not fixed_finite(order)
        ):
            return None
        w = sp.Dummy("bounded_spherical_tail")
        symbols.append(w)
        replacements[atom] = w / u
    try:
        p = sp.Poly(sp.expand_mul(expr.xreplace(replacements)), *symbols)
    except sp.PolynomialError:
        return None
    if p.total_degree() > 2:
        return None
    for monomial, coefficient in p.terms():
        if any(monomial) and clean(coefficient, x, point) != 0:
            if len(atoms) == 1 and next(iter(atoms)).args[0].is_number:
                atom = next(iter(atoms))
                order, u = atom.args
                replacement = sp.sqrt(sp.pi / (2 * u)) * (
                    sp.besselj(order + sp.Rational(1, 2), u)
                    if atom.func is sp.jn
                    else sp.bessely(order + sp.Rational(1, 2), u)
                )
                return fixed_bessel_tail_certificate(
                    expr.xreplace({atom: replacement}), x, point
                )
            return None
    value = clean(p.coeff_monomial(1), x, point)
    if value is None:
        return None
    return answer(
        value,
        "fixed_order_spherical_tail_bound",
        "For fixed finite complex order, spherical J/Y have bounded oscillatory coefficients and O(1/x) positive-real tails (DLMF 10.17/10.47). Every nonconstant remainder coefficient tends to zero; variable or nonfinite orders are declined.",
    )


def airy_positive_remainder_certificate(expr, x, point):
    """Retain a controlled positive-axis Airy remainder for cancellation-sensitive limits."""
    atoms = expr.atoms(sp.airyai, sp.airyaiprime)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not atoms
        or len(atoms) > 2
        or not perturbation_budget(expr, 90)
    ):
        return None
    z = next(iter(atoms)).args[0]
    if z not in (x, x * x) or any(a.args[0] != z for a in atoms):
        return None
    leading = sp.exp(-2 * z ** sp.Rational(3, 2) / 3) / (2 * sp.sqrt(sp.pi))
    replacements = {}
    errors = []
    for atom in atoms:
        error = sp.Dummy("bounded_Airy_remainder")
        errors.append(error)
        replacements[atom] = (
            leading
            * z ** sp.Rational(-1, 4)
            * (1 - 5 / (48 * z ** sp.Rational(3, 2)) + error / z**3)
            if atom.func is sp.airyai
            else -leading
            * z ** sp.Rational(1, 4)
            * (1 + 7 / (48 * z ** sp.Rational(3, 2)) + error / z**3)
        )
    try:
        p = sp.Poly(sp.expand(expr.xreplace(replacements)), *errors)
    except sp.PolynomialError:
        return None
    if p.total_degree() > 2:
        return None
    for monomial, coefficient in p.terms():
        if any(monomial) and clean(sp.Abs(sp.simplify(coefficient)), x, point) != 0:
            return None
    value = clean(sp.simplify(p.coeff_monomial(1)), x, point)
    if value is None:
        return None
    return answer(
        value,
        "airy_positive_cancellation_remainder",
        "DLMF 9.7 positive-ray Ai and Ai-prime expansions retain their first relative corrections with bounded O(z^-3) remainders. Exact polynomial combination of these remainders keeps cancellations; every amplified error coefficient tends to zero independently.",
    )


def fixed_order_subtracted_tail_certificate(expr, x, point):
    """Plan the tail order needed after subtracting fixed-order leading terms."""
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not perturbation_budget(expr, 100)
    ):
        return None
    atoms = expr.atoms(ParabolicCylinderD)
    if len(atoms) == 1:
        atom = next(iter(atoms))
        nu, u = atom.args
        if u == x and nu.is_Rational and abs(nu) <= 16:
            data = linear_in(expr, atom)
            if data:
                a, b = data
                scale = x**nu * sp.exp(-x * x / 4)
                if clean(sp.Abs(a) * scale / x**6, x, point) == 0:
                    approximation = scale * (
                        1
                        - nu * (nu - 1) / (2 * x * x)
                        + nu * (nu - 1) * (nu - 2) * (nu - 3) / (8 * x**4)
                    )
                    value = clean(sp.simplify(a * approximation + b), x, point)
                    if value is not None:
                        return answer(
                            value,
                            "parabolic_cylinder_fixed_order_remainder",
                            "DLMF 12.9 retains two relative correction terms for fixed rational order on the positive real ray; the amplified relative O(x^-6) remainder vanishes.",
                        )
    modified = expr.atoms(StruveL, sp.besseli)
    if modified == {StruveL(0, x), sp.besseli(0, x)}:
        w, v = sp.Dummy("I0"), sp.Dummy("L0")
        try:
            p = sp.Poly(
                sp.expand_mul(expr.xreplace({sp.besseli(0, x): w, StruveL(0, x): v})),
                w,
                v,
            )
        except sp.PolynomialError:
            return None
        if p.total_degree() != 1:
            return None
        a = p.coeff_monomial(w)
        if (
            sp.simplify(a + p.coeff_monomial(v)) != 0
            or clean(sp.Abs(a) / x**7, x, point) != 0
        ):
            return None
        leading = a * (
            2 / (sp.pi * x) + 2 / (sp.pi * x**3) + 18 / (sp.pi * x**5)
        ) + p.coeff_monomial(1)
        value = clean(sp.simplify(leading), x, point)
        if value is not None:
            return answer(
                value,
                "struve_bessel_exact_subtraction_remainder",
                "DLMF 11.6 expands the exact I_0-L_0 difference through x^-5 with O(x^-7) remainder. The exponentially large functions are never approximated separately, and the amplified remainder vanishes.",
            )
    return None


def lambert_positive_chart_certificate(expr, x, point):
    """Evaluate a Lambert-W germ on a proved positive real chart."""
    atoms = expr.atoms(sp.LambertW)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 75)
    ):
        return None
    atom = next(iter(atoms))
    if atom.args[0] != x or (len(atom.args) > 1 and atom.args[1] != 0):
        return None
    if expr.func is sp.LambertW:
        return None
    w = sp.Dummy("positive_W", positive=True)
    transformed = sp.simplify(expr.xreplace({atom: w}).subs(x, w * sp.exp(w)))
    if transformed.has(x, sp.LambertW) or sp.count_ops(transformed) > 100:
        return None
    if any(
        a.func not in (sp.exp, sp.log, sp.sin, sp.cos, sp.sinh, sp.cosh)
        for a in transformed.atoms(sp.Function)
    ):
        return None
    value = clean(transformed, w, sp.oo)
    if value is None:
        return None
    return answer(
        value,
        "principal_lambert_exact_positive_chart",
        "On the positive real tail, w=W_0(x)>0 is an increasing coordinate with x=w*exp(w) exactly and w->infinity. This identity removes the coupled x/W scales before bounded asymptotic analysis; other branches remain outside this chart.",
    )


def lambert_minus_one_branchpoint_certificate(expr, x, point):
    """Use the lower real Lambert-W branch near its branch point."""
    atoms = expr.atoms(sp.LambertW)
    if (
        point != 0
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 50)
    ):
        return None
    atom = next(iter(atoms))
    if (
        len(atom.args) != 2
        or atom.args[1] != -1
        or sp.simplify(atom.args[0] + 1 / sp.E - x) != 0
    ):
        return None
    if sp.simplify(expr - 1 / (atom + 1)) != 0:
        return None
    return answer(
        -sp.oo,
        "lambert_minus_one_signed_branchpoint",
        "DLMF 4.13: on -1/e<z<0, W_-1(z)<-1 and W_-1(-1/e+u)+1=-sqrt(2*e*u)+O(u), u>0. The reciprocal therefore tends to negative infinity without crossing the branch point.",
    )


def finite_complex_entire_certificate(expr, x, point, domain, assumptions=sp.S.true):
    """Evaluate a finite entire composition while preserving approach-domain conditions."""
    if (
        not isinstance(expr, sp.Expr)
        or assumptions is sp.S.false
        or assumptions.has(x)
        or domain is not sp.S.true
        or point.is_finite is not True
        or point.is_real is not False
        or x.is_extended_real is True
        or x.is_finite is False
        or sp.count_ops(expr) > 70
    ):
        return None
    if x.is_imaginary is True and point.is_imaginary is not True:
        return None
    allowed = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.erf, sp.erfc, sp.erfi)

    def regular(node):
        if node == x:
            return point
        if not node.has(x):
            return node if fixed_finite(node) else None
        if node.is_Add or node.is_Mul:
            values = [regular(a) for a in node.args]
            if any(v is None for v in values):
                return None
            return node.func(*values)
        if node.is_Pow and node.exp.is_Integer:
            base = regular(node.base)
            if base is None or (node.exp < 0 and base.is_zero is not False):
                return None
            return base**node.exp
        if node.func in allowed:
            value = regular(node.args[0])
            return node.func(value) if value is not None else None
        return None

    value = regular(expr)
    if value is None:
        return None
    return answer(
        sp.simplify(value),
        "finite_complex_entire_composition",
        "Every varying function is entire, every inner rational denominator is nonzero at the finite complex target, and fixed coefficients are finite. Holomorphic continuity gives the value for every complex approach direction; branch-cut functions are excluded.",
    )
