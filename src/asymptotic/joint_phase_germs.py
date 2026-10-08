"""Attained joint phase limits with exact modular residues."""

import sympy as sp

from .discontinuous_tail_germs import _finite_tail
from .limit_models import LimitEvidence, LimitStatus
from .local_tail_germs import budget


class PhaseHittingIndex(sp.Function):
    """Least integer m>=j with two phase errors below 1/j.

    For j>=1, r in {0,1} and c in {-1,1}, the defining inequalities are
    abs(sin(2*m+r))<1/j and abs(cos(sqrt(2)*(2*m+r))-c)<1/j.
    Kronecker density proves existence. This symbolic definition avoids an
    unbounded numeric search while constructing the limit certificate.
    """

    nargs = 3

    def _eval_is_integer(self):
        j, r, c = self.args
        if (
            j.is_integer is True
            and j.is_positive is True
            and r in (0, 1)
            and c in (-1, 1)
        ):
            return True

    def _eval_is_positive(self):
        return True if self.is_integer is True else None


def phase_hitting_index(j, residue, cosine, max_checks=100000):
    """Evaluate a finite witness search with certified real-ball comparisons.

    The search returns the first qualifying integer. If an interval comparison
    stays undecided or the work limit is reached, it raises instead of treating
    an approximate numerical hit as an exact witness.
    """
    from flint import arb, ctx

    if (
        not isinstance(j, int)
        or j < 1
        or residue not in (0, 1)
        or cosine not in (-1, 1)
    ):
        raise ValueError("positive integer index, residue 0/1 and cosine +/-1 required")
    if not isinstance(max_checks, int) or max_checks < 1:
        raise ValueError("max_checks must be a positive integer")
    for m in range(j, j + max_checks):
        decided = False
        for precision in (80, 160, 320):
            with ctx.workprec(precision):
                phase = arb(2 * m + residue)
                sine_error = abs(phase.sin())
                cosine_error = abs((arb(2).sqrt() * phase).cos() - cosine)
                tolerance = arb(1) / j
                if sine_error < tolerance and cosine_error < tolerance:
                    return m
                if sine_error >= tolerance or cosine_error >= tolerance:
                    decided = True
                    break
        if not decided:
            raise ArithmeticError("phase inequality remains undecided")
    raise ValueError("phase witness search exceeded max_checks")


def _phase_inverse(phase, x):
    y = sp.Dummy("attained_phase", positive=True)
    if phase == x**3 - x:
        # On y>=2, q is positive and q^3+1/(27*q^3)=y.
        # Consequently (q+1/(3*q))^3-(q+1/(3*q))=y, on x>1.
        q = (y / 2 + sp.sqrt(y * y / 4 - sp.Rational(1, 27))) ** sp.Rational(1, 3)
        return sp.Lambda(y, q + 1 / (3 * q))
    logarithmic = phase.func is sp.log
    argument = phase.args[0] if logarithmic else phase
    try:
        polynomial = sp.Poly(argument, x)
    except sp.PolynomialError:
        return None
    if not 1 <= polynomial.degree() <= 2 or any(
        c.is_real is not True or c.is_finite is not True
        for c in polynomial.all_coeffs()
    ):
        return None
    a = polynomial.LC()
    if a.is_positive is not True:
        return None
    if polynomial.degree() == 1:
        value = ((sp.exp(y) if logarithmic else y) - polynomial.nth(0)) / a
    elif logarithmic:
        return None
    else:
        b, c = polynomial.nth(1), polynomial.nth(0)
        value = (-b + sp.sqrt(b * b + 4 * a * (y - c))) / (2 * a)
    return sp.Lambda(y, value)


def _joint_tail(expr, x):
    value = _finite_tail(expr, x)
    if value is not None:
        return value
    if expr.has(sp.log(x)):
        from .elementary_limit_germs import rational_value

        log_tail = sp.Dummy("positive_log_tail", positive=True)
        chart = expr.xreplace({sp.log(x): log_tail})
        if not chart.has(x):
            value = rational_value(chart, log_tail, sp.oo)
            if value is not None and value.is_finite is True:
                return value
    if expr.func in (sp.Min, sp.Max) or expr.is_Add or expr.is_Mul:
        values = [_joint_tail(arg, x) for arg in expr.args]
        if any(v is None for v in values):
            return None
        value = expr.func(*values)
        return value if value.is_real is True and value.is_finite is True else None
    return _finite_tail(expr, x)


def _joint_value(expr, x, replacements):
    if expr in replacements:
        return replacements[expr]
    if not expr.has(sp.Mod, sp.sin, sp.cos):
        return _joint_tail(expr, x)
    if expr.is_Add or expr.is_Mul:
        # Group finite-tail factors before evaluating phase limits. Substituting
        # a zero phase limit first could erase an unbounded multiplier.
        fixed = [a for a in expr.args if not a.has(sp.Mod, sp.sin, sp.cos)]
        varying = [a for a in expr.args if a.has(sp.Mod, sp.sin, sp.cos)]
        values = [_joint_tail(expr.func(*fixed), x)] if fixed else []
        values.extend(_joint_value(a, x, replacements) for a in varying)
        if any(v is None for v in values):
            return None
        value = expr.func(*values)
    elif expr.func in (sp.Min, sp.Max):
        values = [_joint_value(a, x, replacements) for a in expr.args]
        if any(v is None for v in values):
            return None
        value = expr.func(*values)
    elif expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        base = _joint_value(expr.base, x, replacements)
        value = None if base is None else base**expr.exp
    else:
        return None
    return (
        value
        if value is not None and value.is_real is True and value.is_finite is True
        else None
    )


def joint_phase_certificate(expr, x, point, domain, assumptions):
    """Construct two attained limits for the checked sqrt(2)/modular family."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or x.is_real is False
        or x.is_integer is True
        or not expr.has(sp.Mod)
        or not budget(expr, 100)
        or expr.free_symbols - {x}
    ):
        return None
    modular = expr.atoms(sp.Mod)
    if len(modular) != 1:
        return None
    atom = next(iter(modular))
    phase, period = atom.args
    if period != 2:
        return None
    inverse = _phase_inverse(phase, x)
    if inverse is None:
        return None
    trig = expr.atoms(sp.sin, sp.cos)
    kinds = {}
    for node in trig:
        argument = node.args[0]
        if node.func is sp.sin and sp.expand_mul(argument - phase) == 0:
            kinds[node] = "sine"
        elif node.func is sp.cos and sp.expand_mul(argument - sp.sqrt(2) * phase) == 0:
            kinds[node] = "cosine"
        elif (
            node.func is sp.cos
            and argument.is_Pow
            and argument.exp == 2
            and argument.base.func is sp.sin
            and sp.expand_mul(argument.base.args[0] - phase) == 0
        ):
            kinds[node] = "folded"
        else:
            return None
    # Oscillatory denominators need their own pole avoidance proof. The
    # accepted rational/logarithmic denominators are defined on a late tail.
    if any(
        p.exp.is_negative is True and p.base.has(sp.Mod, sp.sin, sp.cos, sp.Min, sp.Max)
        for p in expr.atoms(sp.Pow)
    ):
        return None
    j = sp.Dummy("joint_phase_index", integer=True, positive=True)
    items = []
    for residue, cosine in ((0, -1), (1, 1)):
        replacements = {atom: sp.Integer(residue)}
        replacements.update(
            {
                node: sp.Integer(cosine)
                if kind == "cosine"
                else sp.S.Zero
                if kind == "sine"
                else sp.S.One
                for node, kind in kinds.items()
            }
        )
        value = _joint_value(expr, x, replacements)
        if value is None or value.is_real is not True or value.is_finite is not True:
            return None
        index = (
            PhaseHittingIndex(j, sp.Integer(residue), sp.Integer(cosine)) if trig else j
        )
        sequence = inverse(2 * index + residue)
        statement = "For j>=1, m is the least integer >=j with |sin(2*m+r)|<1/j and |cos(sqrt(2)*(2*m+r))-c|<1/j. The numbers 1,1/pi,sqrt(2)/pi are rationally independent: an integer relation would make pi algebraic, or force an integer relation between 1 and sqrt(2). Kronecker density therefore guarantees such an m on every tail. The inverse positive phase map gives phase(x_j)=2*m+r exactly, so Mod equals r, sin tends to zero, cos(sqrt(2)*phase) tends to c and cos(sin(phase)**2) tends to one. Finite-tail algebra and continuity of Min/Max give the recorded limit. These are attained real sequences with x_j->infinity. Rational denominator poles are finite in number; logarithmic denominators and inverse radicands are valid on the eventual tail. Oscillatory denominators and integer sampling are excluded."
        if not trig:
            statement = "The explicit sequence is the positive inverse phase at 2*j+r, so the modular residue is exactly r. It tends to real infinity; the checked finite-tail algebra retains denominator avoidance."
        items.append(
            LimitEvidence(
                "attained_joint_modular_phase", statement, ((x, sequence),), value
            )
        )
    if (items[0].value - items[1].value).is_zero is not False:
        return None
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
