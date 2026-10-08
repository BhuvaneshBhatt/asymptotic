"""Fixed-parameter tail limits with explicit domain and pole prerequisites."""

import sympy as sp

from ._symbolic_policy import bounded_ask
from .compact_limit_germs import answer
from .fixed_ray_branch_germs import DirectionalInfinity
from .limit_models import LimitEvidence, LimitStatus


def binomial_joint_pole_certificate(expr, variables, target, domain, assumptions):
    """Attain two gamma-residue limits at a negative-integer binomial corner."""
    if (
        expr.func is not sp.binomial
        or len(variables) != 2
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or set(expr.args) != set(variables)
    ):
        return None
    points = dict(zip(variables, target, strict=True))
    x, y = expr.args
    m, n = points[x], points[y]
    if not (m.is_Integer and n.is_Integer and n < m < 0):
        return None
    j = sp.Dummy("binomial_corner_index", positive=True, integer=True)
    diagonal = (
        (-1) ** (m - n)
        * sp.factorial(-n - 1)
        / (sp.factorial(-m - 1) * sp.factorial(m - n))
    )
    evidence = (
        LimitEvidence(
            "binomial_fixed_lower_subsequence",
            "At x=m+1/j, y=n<0, the numerator gamma is finite for j>1 "
            "and reciprocal Gamma(n+1)=0. The binomial is exactly zero; "
            "the full pair approaches the corner through distinct defined points.",
            ((x, m + 1 / j), (y, n)),
            sp.S.Zero,
        ),
        LimitEvidence(
            "binomial_diagonal_residue_subsequence",
            "At x=m+1/j, y=n+1/j, Gamma(x-y+1)=(m-n)! is finite. "
            "The simple-pole residues of Gamma(m+1+t) and Gamma(n+1+t) "
            "give the stated nonzero ratio. For j>1 neither moving gamma "
            "argument is an integer pole, so the subsequence is attained.",
            ((x, m + 1 / j), (y, n + 1 / j)),
            diagonal,
        ),
    )
    return LimitStatus.DOES_NOT_EXIST, None, evidence


def binomial_ratio_certificate(expr, x, point, domain, assumptions):
    """Use fixed-shift gamma ratios for adjacent binomial parameter gaps."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or sp.count_ops(expr) > 100
    ):
        return None
    atoms = expr.atoms(sp.binomial)
    if len(atoms) != 2:
        return None
    powers = expr.as_powers_dict()
    upper = next((a for a in atoms if powers.get(a) == 1), None)
    lower = next((a for a in atoms if powers.get(a) == -1), None)
    if upper is None or lower is None or powers.get(x) != -1:
        return None
    coefficient = sp.Mul(
        *(b**e for b, e in powers.items() if b not in (upper, lower, x))
    )
    if coefficient.has(x) or coefficient.is_finite is not True:
        return None
    for atom in atoms:
        if any(sp.expand(a).coeff(x) != 1 or (a - x).has(x) for a in atom.args):
            return None
    p = sp.cancel(lower.args[0] - lower.args[1] + 1)
    if sp.cancel(upper.args[0] - upper.args[1] - p) != 0:
        return None
    parameters = expr.free_symbols - {x}
    if any(bounded_ask(sp.Q.finite(a), assumptions) is not True for a in parameters):
        return None
    clauses = sp.And.make_args(assumptions)
    pole_free = sp.Ne(1 / sp.gamma(p), 0) in clauses
    pole_free |= bounded_ask(sp.Q.positive(p), assumptions) is True
    if not pole_free:
        return None
    return answer(
        coefficient / p,
        "fixed_shift_binomial_ratio",
        "For fixed finite complex a,b, Gamma(x+a)/Gamma(x+b)="
        "x**(a-b)*(1+O(1/x)) on the positive tail. The two binomial "
        "gaps differ by one, so their x powers cancel the explicit 1/x. "
        "Gamma(p)/Gamma(p+1)=1/p gives the remaining constant. "
        "The nonzero reciprocal-Gamma prerequisite excludes nonpositive "
        "integer p; all moving gamma arguments and denominator binomials "
        "are finite and nonzero eventually. No cancellation at an "
        "identically undefined parameter is asserted.",
    )


def hyperbolic_parameter_tail_certificate(expr, x, point, domain, assumptions):
    """Separate the exponential terms when a trigonometric coefficient vanishes."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or sp.count_ops(expr) > 100
    ):
        return None
    parameters = expr.free_symbols - {x}
    if len(parameters) != 1:
        return None
    n = next(iter(parameters))
    if bounded_ask(sp.Q.finite(n), assumptions) is not True:
        return None
    angle = sp.pi * (n / 2 + sp.Rational(1, 4))
    template = (
        2 ** (1 - n) * x ** (n - 1) / sp.gamma(n + sp.S.Half)
        - sp.cos(angle) * sp.sinh(x)
        + sp.sin(angle) * sp.cosh(x) / (sp.I**n * sp.sqrt(x))
    )
    if expr != template:
        return None
    a, b = -sp.cos(angle), sp.sin(angle) / sp.I**n
    value = sp.Piecewise(
        (DirectionalInfinity(a), sp.Ne(a, 0)), (DirectionalInfinity(b), True)
    )
    return answer(
        value,
        "hyperbolic_parameter_dominance",
        "For each fixed finite n the algebraic power is dominated by "
        "exp(x)/sqrt(x). If cos(angle) is nonzero, the sinh term has "
        "that fixed leading direction; if it vanishes, sin(angle)**2=1 "
        "and the cosh/sqrt(x) term is nonzero and dominates. The two "
        "strata cover every finite n. Reciprocal Gamma is entire and "
        "I**n is nonzero, so there are no parameter poles in these "
        "coefficients or on the positive real approach tail.",
    )
