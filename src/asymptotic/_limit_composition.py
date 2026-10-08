"""Proof-aware simultaneous real multivariate limits.

A simultaneous Euclidean limit ranges over the full admissible approach domain,
not an iterated sequence of one-variable limits. The solver combines exact
reductions, uniform bounds, path certificates, and local geometric methods;
unsupported cases return an explicit unknown status.
"""

from __future__ import annotations

import sympy as sp
from sympy.core.function import AppliedUndef

from ._symbolic_policy import bounded_ask, bounded_limit
from .limit_models import (
    LimitEvidence,
    LimitStatus,
)


def limit(*args, **kwargs):
    """Internal lazy recursion into the limit engine without an import cycle."""
    from ._limit_engine import limit as _engine_limit

    return _engine_limit(*args, **kwargs)


def _exact_equal(a: sp.Expr, b: sp.Expr) -> bool | None:
    if a == b:
        return True
    # Preserve coefficient-zero cells before cancellation can erase the
    # parameter from an unevaluated generic infinite value.
    if any(
        getattr(value, "free_symbols", ()) and value.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
        for value in (a, b)
    ):
        return None
    try:
        d = sp.cancel(a - b)
    except (TypeError, ValueError, ZeroDivisionError):
        d = a - b
    if d == 0 or d.is_zero is True:
        return True
    # A generic infinite path value can lose its leading coefficient on a
    # parameter cell. For example, a*oo does not describe the a=0 germ.
    if d.free_symbols and d.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    if d.is_zero is False:
        return False
    # Symbolic nonidentity is not pointwise inequality: a parameterized
    # difference can vanish on a stratum even when equals(0) returns False.
    if d.free_symbols:
        return None
    try:
        eq = d.equals(0)
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        return None
    return eq if eq in (True, False) else None


def _two_sided_limit(
    expr: sp.Expr, variable: sp.Symbol, point: sp.Expr
) -> sp.Expr | None:
    right = bounded_limit(expr, variable, point, direction="+", allow_general=True)
    if variable.is_positive is True and point == 0:
        return right
    left = bounded_limit(expr, variable, point, direction="-", allow_general=True)
    if right is None or left is None:
        return None
    same = _exact_equal(right, left)
    return right if same is True else None


def _local_germ_composition_certificate(expr, variables, target, domain, assumptions):
    """Lift a registered first-order local germ through a certified inner germ.

    Restricted to f(u)/v with a registered order-one germ at u=0.  Both
    u->0 and u/v must be independently certified by the ordinary limit engine.
    """
    num, den = sp.fraction(expr)
    if den == 1:
        return None
    atoms = [
        a
        for a in sp.preorder_traversal(num)
        if getattr(a, "func", None) in (sp.erf, sp.erfi)
    ]
    # Cancellation of equal first-order registered germs: erfi(u)-erf(u)=o(u).
    if (
        len(atoms) == 2
        and sp.simplify(
            sp.expand(num) - (sp.erfi(atoms[0].args[0]) - sp.erf(atoms[0].args[0]))
        )
        == 0
        and atoms[0].args[0] == atoms[1].args[0]
        and den == atoms[0].args[0]
    ):
        inner_result = limit(
            atoms[0].args[0],
            variables,
            target,
            domain=domain,
            assumptions=assumptions,
            return_result=True,
        )
        if inner_result.status is LimitStatus.PROVED and inner_result.value == 0:
            from .local_germ_algebra import function_germ

            radial = sp.Dummy("_germ_r", positive=True)
            g0 = function_germ(atoms[0].func, radial_variable=radial)
            g1 = function_germ(atoms[1].func, radial_variable=radial)
            if (
                g0 is not None
                and g1 is not None
                and g0.order == g1.order
                and sp.simplify(g0.coefficient - g1.coefficient) == 0
            ):
                return sp.S.Zero, LimitEvidence(
                    "local_germ_cancellation",
                    "equal leading local germs cancel, leaving o(inner)",
                    value=sp.S.Zero,
                )
    if len(atoms) != 1 or num != atoms[0]:
        return None
    atom = atoms[0]
    inner = atom.args[0]
    inner_result = limit(
        inner,
        variables,
        target,
        domain=domain,
        assumptions=assumptions,
        return_result=True,
    )
    if inner_result.status is not LimitStatus.PROVED or inner_result.value != 0:
        return None
    if inner == den or (inner.func is sp.sin and inner.args[0] == den):
        ratio_value = sp.S.One
    else:
        ratio_result = limit(
            sp.cancel(inner / den),
            variables,
            target,
            domain=domain,
            assumptions=assumptions,
            return_result=True,
        )
        if ratio_result.status is not LimitStatus.PROVED:
            return None
        ratio_value = ratio_result.value
    from .local_germ_algebra import function_germ

    radial = sp.Dummy("_germ_r", positive=True)
    germ = function_germ(atom.func, radial_variable=radial)
    if germ is None or germ.order.radial != 1:
        return None
    value = sp.simplify(germ.coefficient * ratio_value)
    return value, LimitEvidence(
        "local_germ_composition",
        "registered first-order local germ composed with certified inner quotient",
        value=value,
    )


def _resolved_limit_value(value, variables=()):
    """Return True only for an actual scalar/vector extended-real/complex value.

    Reject unevaluated function calls and parameter-dependent indeterminate
    powers/products that substitution or SymPy limit may leave behind.
    """
    if value is None:
        return False
    value = sp.sympify(value)
    if value.has(*variables) or value.has(sp.nan, sp.zoo):
        return False
    if isinstance(value, sp.Set) or value.has(
        sp.AccumBounds, sp.Limit, sp.Subs, sp.Derivative, AppliedUndef
    ):
        return False
    from .fixed_ray_branch_germs import DirectionalInfinity

    if value.func is not DirectionalInfinity and value.has(DirectionalInfinity):
        return False
    # Unevaluated special-function values at infinity are not scalar answers.
    if any(
        argument.has(sp.oo, -sp.oo, sp.zoo, sp.nan)
        for atom in value.atoms(sp.Function)
        for argument in atom.args
    ):
        return False
    for atom in value.atoms(sp.uppergamma, sp.li):
        argument = atom.args[-1]
        if sp.simplify(argument).is_negative is True:
            return False
    # Unevaluated special-function calls at singular/parameter-dependent data
    # are not certified values merely because they contain no limit variable.
    unsafe_functions = (sp.bessely, sp.besselj, sp.hankel1, sp.hankel2)
    for atom in sp.preorder_traversal(value):
        if getattr(atom, "func", None) in unsafe_functions and atom.free_symbols:
            return False
    # Symbolic zero/infinity powers and their products need assumptions.
    for power in value.atoms(sp.Pow):
        if power.exp.free_symbols and power.base in (sp.S.Zero, sp.oo, -sp.oo):
            return False
    return not (
        value.has(sp.oo, -sp.oo) and value.free_symbols and value not in (sp.oo, -sp.oo)
    )


def _domain_ratio_squeeze_certificate(expr, variables, target, domain):
    """Cheap exact squeeze for cusp-like domains ``|p| <= q**m``, ``q > 0``.

    If ``m>1`` and ``q -> 0``, then ``|p/q| <= q**(m-1) -> 0``.
    This avoids sending a common semialgebraic cusp to full epsilon-delta QE.
    """
    if domain is sp.S.true or any(value != 0 for value in target):
        return None
    numerator, denominator = sp.fraction(sp.cancel(sp.together(expr)))
    if denominator == 1:
        return None
    clauses = sp.And.make_args(domain)
    positive = any(
        isinstance(clause, sp.StrictGreaterThan)
        and clause.lhs == denominator
        and clause.rhs == 0
        for clause in clauses
    )
    if not positive:
        return None
    for clause in clauses:
        if not isinstance(clause, (sp.LessThan, sp.StrictLessThan)):
            continue
        if clause.lhs != sp.Abs(numerator):
            continue
        rhs = clause.rhs
        if (
            isinstance(rhs, sp.Pow)
            and rhs.base == denominator
            and rhs.exp.is_integer is True
            and rhs.exp > 1
        ):
            return sp.S.Zero, LimitEvidence(
                "domain_ratio_squeeze",
                f"domain gives |numerator| <= denominator**{rhs.exp} with positive denominator tending to zero",
                value=sp.S.Zero,
            )
    return None


def _branch_center_regular(atom, variables, substitutions):
    """Certify that inverse-function centers avoid their principal branch cuts.

    A finite value on a cut is insufficient for complex approaches. The
    principal Lambert-W branch is continuous at its square-root endpoint;
    inverse sine and cosine are also continuous at their finite endpoints.
    """
    if not atom.has(*variables):
        return True
    argument = atom.args[0]
    center = argument.subs(substitutions, simultaneous=True)
    branch = atom.args[1] if len(atom.args) > 1 else sp.S.Zero
    if branch.has(*variables):
        return False
    # Ordinary coordinate limits use real displacements. Unrestricted symbols
    # therefore need a centered real chart for this continuity check; free
    # parameters retain their declared assumptions. Whole-plane limits supply
    # separate real coordinates, so their imaginary displacement is preserved.
    real_argument = argument
    if argument.is_real is not True:
        chart = {
            v: substitutions[v] + sp.Dummy("real_branch_displacement", real=True)
            for v in variables
            if v.is_real is None and v.is_imaginary is None
        }
        if chart:
            real_argument = argument.xreplace(chart)
    # Real arguments stay on one assigned boundary value of a complex cut.
    # Their restrictions are continuous away from logarithmic singularities.
    if real_argument.is_real is True and center.is_real is True:
        if atom.func is sp.atanh:
            return (center - 1).is_zero is False and (center + 1).is_zero is False
        if atom.func is sp.LambertW and branch != 0:
            return center.is_zero is False
        return True
    if atom.func is sp.atan:
        return (
            sp.re(center).is_zero is False
            or (1 - sp.Abs(sp.im(center))).is_positive is True
        )
    if atom.func in (sp.asin, sp.acos, sp.atanh):
        if sp.im(center).is_zero is False:
            return True
        distance = 1 - sp.Abs(center)
        return (
            distance.is_positive is True
            if atom.func is sp.atanh
            else distance.is_nonnegative is True
        )
    if sp.im(center).is_zero is False:
        return True
    if branch == 0:
        return (center + sp.exp(-1)).is_nonnegative is True
    return center.is_positive is True


def _regular_composition(expr, variables, target) -> sp.Expr | None:
    """Prove a limit by exact substitution through continuous expression heads."""
    try:
        value = expr.subs(dict(zip(variables, target, strict=True)), simultaneous=True)
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        return None
    if not _resolved_limit_value(value, variables):
        return None
    if value in (sp.oo, -sp.oo):
        return None
    # An undefined symbolic function carries no continuity theorem.  Merely
    # substituting its arguments cannot certify a limit.
    if expr.has(AppliedUndef):
        return None
    for atom in expr.atoms(sp.binomial):
        top = atom.args[0]
        top_at = top.subs(dict(zip(variables, target, strict=True)), simultaneous=True)
        if (
            top.has(*variables)
            and top_at.is_integer is True
            and top_at.is_negative is True
        ):
            return None
    # Direct substitution is a proof only when no denominator vanishes and the
    # expression tree is composed of heads continuous at the substituted data.
    _, den = sp.fraction(expr)
    try:
        den_at = den.subs(dict(zip(variables, target, strict=True)), simultaneous=True)
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        return None
    if den_at == 0 or den_at.is_zero is True:
        return None
    subs_map = dict(zip(variables, target, strict=True))
    unsafe = (
        sp.Piecewise,
        sp.sign,
        sp.Heaviside,
        sp.floor,
        sp.ceiling,
        sp.Mod,
        sp.atan2,
    )
    if expr.has(*unsafe):
        return None
    # Every function argument must be regular, even when substitution into the
    # whole expression cancels equal singular calls (gamma(oo)/gamma(oo)).
    for atom in expr.atoms(sp.Function):
        for argument in atom.args:
            at = argument.subs(subs_map, simultaneous=True)
            if at.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                return None
    # SymPy leaves several singular special-function values
    # unevaluated (for example hankel1(0, 0)).  An unevaluated value is not a
    # continuity certificate.  Reject substitution across their singular
    # origin so branch-aware germ machinery can decide the limit.
    for atom in expr.atoms(sp.hankel1, sp.hankel2, sp.bessely):
        try:
            arg_at = sp.simplify(atom.args[1].subs(subs_map, simultaneous=True))
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            return None
        if arg_at == 0 or arg_at.is_zero is True:
            return None
    # At zero argument, a varying Bessel order introduces the coupled factor
    # (argument/2)**order. Separate substitution can erase its growth.
    for atom in expr.atoms(sp.besselj, sp.besseli):
        if atom.args[0].has(*variables):
            arg_at = atom.args[1].subs(subs_map, simultaneous=True)
            if arg_at == 0 or arg_at.is_zero is True:
                return None
    # Principal complex branches are discontinuous across the negative-real
    # cut.  Direct substitution at a cut point is therefore never a continuity
    # certificate: nearby upper/lower germs may have different values even
    # though SymPy returns the principal value at the target itself.
    for atom in sp.preorder_traversal(expr):
        branch_arg = None
        if getattr(atom, "func", None) in (sp.arg, sp.log):
            branch_arg = atom.args[0]
        elif getattr(atom, "is_Pow", False) and atom.exp.is_integer is not True:
            branch_arg = atom.base
        elif getattr(atom, "func", None) in (sp.Ei, sp.Ci, sp.Chi):
            branch_arg = atom.args[0]
        elif getattr(atom, "func", None) in (sp.uppergamma, sp.expint):
            branch_arg = atom.args[1]
        if branch_arg is None:
            continue
        try:
            branch_at = sp.simplify(
                sp.expand_complex(branch_arg.subs(subs_map, simultaneous=True))
            )
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            return None
        if branch_at.is_real is True and bounded_ask(sp.Q.negative(branch_at)) is True:
            return None

    for atom in expr.atoms(sp.atan, sp.atanh, sp.asin, sp.acos, sp.LambertW):
        if not _branch_center_regular(atom, variables, subs_map):
            return None

    # A symbolic exponent over a base tending to zero is not a continuous
    # composition uniformly in that parameter (0**p depends on Re(p)).
    limit_vars = set(variables)
    for power in expr.atoms(sp.Pow):
        if power.exp.is_negative is True:
            base_at = power.base.subs(subs_map, simultaneous=True)
            if not base_at.free_symbols and base_at.is_algebraic is True:
                base_at = sp.cancel(base_at)
            if base_at.is_zero is not False:
                return None
        if power.exp.has(*variables):
            base_at = sp.simplify(power.base.subs(subs_map, simultaneous=True))
            exponent_at = power.exp.subs(subs_map, simultaneous=True)
            if base_at == 0 or exponent_at.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                return None
        if power.exp.free_symbols - limit_vars:
            try:
                base_at = sp.simplify(power.base.subs(subs_map, simultaneous=True))
            except (TypeError, ValueError, NotImplementedError, RecursionError):
                return None
            if base_at == 0 or base_at.is_zero is True:
                return None
    return value


def _special_univariate_limit_certificate(expr, variable, target, domain, assumptions):
    """Certify a few structural scalar limits before the SymPy fallback.

    These rules exist partly as an upstream-soundness firewall: they either
    prove the result from simpler limits or explicitly decline parameter
    regimes for which a single unconditional value would be unsound.
    """
    side_assumptions = sp.And(sp.sympify(assumptions), sp.sympify(domain))

    # exp(LambertW(g)) -> +oo when g -> +oo through the positive real germ.
    if expr.func is sp.exp and expr.args[0].func is sp.LambertW:
        warg = expr.args[0].args[0]
        inner = limit(
            warg,
            variable,
            target,
            domain=domain,
            assumptions=assumptions,
            return_result=True,
        )
        if inner.status is LimitStatus.PROVED and inner.value is sp.oo:
            return sp.oo, LimitEvidence(
                "lambertw_positive_infinity",
                "LambertW is increasing and unbounded on the positive real axis, and exp preserves +infinity",
                value=sp.oo,
            )

    # Positive variable powers are safely normalized as exp(exponent*log(base)).
    # This handles indeterminate 0**0 forms without asking the generic SymPy
    # limit engine to recurse on the original power.
    if expr.is_Pow and expr.exp.has(variable):
        base = expr.base

        def _eventually_positive(e):
            if bounded_ask(sp.Q.positive(e), side_assumptions) is True:
                return True
            if e == variable and domain == (variable > target):
                return True
            if (
                e.func is sp.log
                and e.args[0] == 1 / variable
                and domain == (variable > 0)
            ):
                return True
            if e.func is sp.log and e.args[0].func is sp.log:
                # Iterated logs of 1/t are eventually positive as t -> 0+.
                z = e
                while z.func is sp.log:
                    z = z.args[0]
                return z == 1 / variable and target == 0 and domain == (variable > 0)
            if e.is_Mul:
                return all(_eventually_positive(f) for f in e.args)
            if e.is_Pow and e.exp.is_integer is True:
                return e.exp.is_even is True or _eventually_positive(e.base)
            return False

        if _eventually_positive(base):
            log_expr = sp.expand_log(expr.exp * sp.log(base), force=True)
            log_result = limit(
                log_expr,
                variable,
                target,
                domain=domain,
                assumptions=assumptions,
                return_result=True,
            )
            if log_result.status is LimitStatus.PROVED:
                lv = log_result.value
                value = (
                    sp.S.Zero
                    if lv is -sp.oo
                    else (sp.oo if lv is sp.oo else sp.exp(lv))
                )
                if _resolved_limit_value(value, (variable,)):
                    return value, LimitEvidence(
                        "positive_power_log_normalization",
                        "eventually positive power normalized through exponent*log(base)",
                        value=value,
                    )

    # Finite sums with one symbolic exponential rate c/t.  Normalize products
    # first so the decisive rate is visible, then use only assumption-proved
    # signs.  This is the structured core of SymPy issue #22893.
    normalized = sp.powsimp(sp.expand_mul(expr))
    if target == 0 and domain == (variable > 0) and normalized.is_Add:
        exponential_terms = []
        constant_terms = []
        for term in normalized.args:
            exp_atoms = tuple(term.atoms(sp.exp))
            if len(exp_atoms) == 1:
                atom = exp_atoms[0]
                coefficient = sp.simplify(term / atom)
                rate = sp.simplify(atom.args[0] * variable)
                if not rate.has(variable) and coefficient.is_positive is True:
                    exponential_terms.append((coefficient, rate))
                    continue
            if not term.has(variable):
                constant_terms.append(term)
                continue
            exponential_terms = []
            break
        if len(exponential_terms) == 1:
            coefficient, rate = exponential_terms[0]
            constant = sp.simplify(sum(constant_terms, sp.S.Zero))
            if bounded_ask(sp.Q.positive(rate), side_assumptions) is True:
                return sp.oo, LimitEvidence(
                    "parameter_exponential_rate",
                    "assumptions prove the symbolic exponential rate is positive",
                    value=sp.oo,
                )
            if bounded_ask(sp.Q.negative(rate), side_assumptions) is True:
                return constant, LimitEvidence(
                    "parameter_exponential_rate",
                    "assumptions prove the symbolic exponential rate is negative",
                    value=constant,
                )
            if bounded_ask(sp.Q.zero(rate), side_assumptions) is True:
                value = sp.simplify(constant + coefficient)
                return value, LimitEvidence(
                    "parameter_exponential_rate",
                    "assumptions prove the symbolic exponential rate vanishes",
                    value=value,
                )

    # Min/Max of exponentials: eventual ordering follows the exponent
    # difference. This proves the branch before taking its limit.
    if expr.func in (sp.Min, sp.Max) and len(expr.args) == 2:
        left, right = expr.args
        if left.func is sp.exp and right.func is sp.exp:
            delta = limit(
                left.args[0] - right.args[0],
                variable,
                target,
                domain=domain,
                assumptions=assumptions,
                return_result=True,
            )
            chosen = None
            if delta.status is LimitStatus.PROVED:
                if delta.value is sp.oo:
                    chosen = right if expr.func is sp.Min else left
                elif delta.value is -sp.oo:
                    chosen = left if expr.func is sp.Min else right
                elif delta.value == 0:
                    # Equal limiting exponents do not alone prove eventual order.
                    chosen = None
            if chosen is not None:
                branch = limit(
                    chosen,
                    variable,
                    target,
                    domain=domain,
                    assumptions=assumptions,
                    return_result=True,
                )
                if branch.status is LimitStatus.PROVED:
                    return branch.value, LimitEvidence(
                        "eventual_exp_minmax_dominance",
                        "exponent difference proves the eventual Min/Max branch",
                        value=branch.value,
                    )

    return None


def _parameter_exponential_rate_is_unresolved(
    expr, variable, target, domain, assumptions
):
    """Detect symbolic exponential rates whose sign controls an infinite limit."""
    if target != 0 or domain != (variable > 0):
        return False
    params = expr.free_symbols - {variable}
    if not params:
        return False
    # Combine exponential products first so cancellation exposes the actual
    # symbolic rate (e.g. exp(-a/t)*exp(b/t) -> exp((b-a)/t)).
    expr = sp.powsimp(sp.expand_mul(expr))
    side_assumptions = sp.And(sp.sympify(assumptions), sp.sympify(domain))
    for atom in expr.atoms(sp.exp):
        rate = sp.simplify(atom.args[0] * variable)
        if rate.has(variable) or not (rate.free_symbols & params):
            continue
        if (
            bounded_ask(sp.Q.positive(rate), side_assumptions) is not True
            and bounded_ask(sp.Q.negative(rate), side_assumptions) is not True
            and bounded_ask(sp.Q.zero(rate), side_assumptions) is not True
        ):
            return True
    return False


def _polygamma_composition_certificate(expr, variables, target, domain, assumptions):
    """Certify ``polygamma(n(x), z(x))`` by regular inner composition.

    For a nonnegative-integer limiting order, polygamma is meromorphic in its
    second argument with poles only at the nonpositive integers.  Certifying
    both inner limits and excluding that pole set therefore makes substitution
    a sound local continuity argument, including projective/infinite charts.
    """
    if getattr(expr, "func", None) is not sp.polygamma or len(expr.args) != 2:
        return None
    order_expr, argument_expr = expr.args
    order_result = limit(
        order_expr,
        variables,
        target,
        domain=domain,
        assumptions=assumptions,
        return_result=True,
    )
    argument_result = limit(
        argument_expr,
        variables,
        target,
        domain=domain,
        assumptions=assumptions,
        return_result=True,
    )
    if (
        order_result.status is not LimitStatus.PROVED
        or argument_result.status is not LimitStatus.PROVED
    ):
        return None
    order_value = sp.sympify(order_result.value)
    argument_value = sp.sympify(argument_result.value)
    if not (order_value.is_integer is True and order_value.is_nonnegative is True):
        return None
    # For integer order >= 0 the z-poles are exactly 0, -1, -2, ... .
    if argument_value.is_integer is True and argument_value.is_nonpositive is True:
        return None
    if argument_value.is_finite is not True:
        return None
    # A nonreal finite point is automatically off the real pole set.  For a
    # real symbolic point require a proof that it is not a nonpositive integer.
    if argument_value.is_real is True and argument_value.is_integer is not False:
        positive = bounded_ask(sp.Q.positive(argument_value))
        if positive is not True:
            return None
    value = sp.polygamma(order_value, argument_value)
    return value, LimitEvidence(
        "polygamma_regular_composition",
        "both arguments have certified limits and the limiting polygamma point is regular",
        value=value,
    )


def _complex_log_origin_conflict(expr, variables, target):
    """Prove direction-dependent principal phase for a rank-two complex inner germ."""
    if getattr(expr, "func", None) is not sp.log or len(variables) < 2:
        return None
    inner = expr.args[0]
    subs = dict(zip(variables, target, strict=True))
    try:
        if sp.simplify(inner.subs(subs, simultaneous=True)) != 0:
            return None
        real_part = sp.re(inner).expand(complex=True)
        imag_part = sp.im(inner).expand(complex=True)
        jacobian = sp.Matrix(
            [
                [sp.diff(real_part, v).subs(subs) for v in variables],
                [sp.diff(imag_part, v).subs(subs) for v in variables],
            ]
        )
        if jacobian.rank() < 2:
            return None
    except (TypeError, ValueError, NotImplementedError):
        return None
    return LimitEvidence(
        "complex_log_phase_conflict",
        "the complex logarithm argument vanishes with two independent real phase directions",
    )


def _principal_cut_conflict(expr, variables, target):
    """Prove a direct principal-branch jump at a negative-real target.

    This handles only a top-level ``arg``, ``log`` or
    nonintegral power.  Nested branch expressions remain UNKNOWN unless a
    separate branch-germ rule proves their composition soundly.
    """
    # Limit variables range over real coordinates even if the public Symbols
    # have no assumptions. Real dummies let imaginary transverse derivatives
    # be decided without inventing assumptions about free parameters.
    real_variables = tuple(sp.Dummy("cut_coordinate", real=True) for _ in variables)
    node = sp.sympify(expr).xreplace(dict(zip(variables, real_variables)))
    variables = real_variables
    kind = None
    if node.func in (sp.acos, sp.asin):
        base = node.args[0]
        subs = dict(zip(variables, target, strict=True))
        at = sp.simplify(base.subs(subs, simultaneous=True))
        outside = at.is_real is True and (
            (node.func is sp.acos and bounded_ask(sp.Q.negative(at + 1)) is True)
            or (
                node.func is sp.asin
                and (
                    bounded_ask(sp.Q.positive(at - 1)) is True
                    or bounded_ask(sp.Q.negative(at + 1)) is True
                )
            )
        )
        if not outside:
            return None
        imag = sp.simplify(sp.im(sp.expand_complex(base)))
        transverse_var = None
        for variable in variables:
            slope = sp.simplify(sp.diff(imag, variable).subs(subs, simultaneous=True))
            if slope.is_zero is False or (slope.is_number and slope != 0):
                transverse_var = variable
                break
        if transverse_var is None:
            return None
        t = sp.Dummy("_branch_side", positive=True)
        plus = dict(subs)
        minus = dict(subs)
        plus[transverse_var] = target[variables.index(transverse_var)] + t
        minus[transverse_var] = target[variables.index(transverse_var)] - t
        try:
            upper = bounded_limit(
                node.subs(plus, simultaneous=True),
                t,
                0,
                direction="+",
                allow_general=True,
            )
            lower = bounded_limit(
                node.subs(minus, simultaneous=True),
                t,
                0,
                direction="+",
                allow_general=True,
            )
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            return None
        if _exact_equal(upper, lower) is not False:
            return None
        return (
            LimitEvidence(
                "principal_cut_upper",
                "upper-side inverse-trigonometric branch germ",
                value=upper,
            ),
            LimitEvidence(
                "principal_cut_lower",
                "lower-side inverse-trigonometric branch germ",
                value=lower,
            ),
        )
    if node.func in (sp.arg, sp.log):
        base = node.args[0]
        kind = node.func
        exponent = None
    elif node.is_Pow and node.exp.is_integer is not True:
        base = node.base
        exponent = node.exp
        kind = sp.Pow
    else:
        return None
    subs = dict(zip(variables, target, strict=True))
    try:
        at = sp.simplify(base.subs(subs, simultaneous=True))
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        return None
    if not (at.is_real is True and bounded_ask(sp.Q.negative(at)) is True):
        return None
    # Certify that the argument crosses the cut transversely: the imaginary
    # part has a nonzero first derivative in at least one real limit variable.
    imag = sp.simplify(sp.im(sp.expand_complex(base)))
    transverse = False
    for variable in variables:
        try:
            slope = sp.simplify(sp.diff(imag, variable).subs(subs, simultaneous=True))
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            continue
        if slope.is_zero is False or (slope.is_number and slope != 0):
            transverse = True
            break
    if not transverse:
        return None
    radius = sp.Abs(at)
    if kind is sp.arg:
        upper, lower = sp.pi, -sp.pi
    elif kind is sp.log:
        upper, lower = sp.log(radius) + sp.I * sp.pi, sp.log(radius) - sp.I * sp.pi
    else:
        if exponent.has(*variables):
            return None
        upper = sp.simplify(radius**exponent * sp.exp(sp.I * sp.pi * exponent))
        lower = sp.simplify(radius**exponent * sp.exp(-sp.I * sp.pi * exponent))
    if _exact_equal(upper, lower) is not False:
        return None
    return (
        LimitEvidence(
            "principal_cut_upper", "upper-side principal branch germ", value=upper
        ),
        LimitEvidence(
            "principal_cut_lower", "lower-side principal branch germ", value=lower
        ),
    )


def _proper_composition_candidates(expr, variables):
    """Yield scalar inner expressions whose replacement makes a univariate outer skeleton.

    This is structural rather than heuristic algebraic rewriting: the
    outer expression must become independent of the original approach variables
    after one exact ``xreplace``.  Scalar composition can therefore reuse the existing univariate
    asymptotic/limit machinery without confusing pathwise agreement with proof.
    """
    z = sp.Dummy("_composition_z", real=True)
    seen = set()
    for inner in sp.preorder_traversal(expr):
        if inner == expr or inner in seen or not isinstance(inner, sp.Expr):
            continue
        if not inner.has(*variables):
            continue
        seen.add(inner)
        outer = expr.xreplace({inner: z})
        if outer.has(*variables) or not outer.has(z):
            continue
        yield inner, z, outer


def _monotone_infinite_outer_certificate(expr, variables, target, domain):
    """Certify selected real outer compositions from a positive divergent inner germ.

    This route is narrow.  It proves ``atan(g) -> pi/2`` when
    ``g`` is nonnegative on the punctured real germ and a simple exact norm
    inequality proves ``g -> +oo``.  It avoids asking path limits to reason
    about an expression whose one-sided positivity is already structural.
    """
    if domain is not sp.S.true or getattr(expr, "func", None) is not sp.atan:
        return None
    g = expr.args[0]
    if len(variables) < 2 or any(a != 0 for a in target):
        return None
    num, den = sp.fraction(g)
    if den != sum(v**2 for v in variables):
        return None
    # Two exact radial forms are enough for the current theorem.
    # 1/r^2 -> +oo directly; sum |x_i|/r^2 >= 1/r -> +oo.
    if num == 1:
        statement = (
            "1/sum(x_i**2) -> +oo on the punctured real germ, then atan is monotone"
        )
    elif sp.expand(num - sum(sp.Abs(v) for v in variables)) == 0:
        statement = (
            "sum(abs(x_i))/sum(x_i**2) >= 1/sqrt(sum(x_i**2)) "
            "-> +oo, then atan is monotone"
        )
    else:
        return None
    return sp.pi / 2, LimitEvidence(
        "positive_radial_divergence", statement, value=sp.pi / 2
    )


def _transcendental_composition_certificate(expr, variables, target, domain):
    """Prove a multivariate limit by certified scalar composition.

    The composition certificate first proves the multivariate limit of an exact inner expression
    using the ordinary proof machinery, including selective semialgebraic CAD.
    It then computes the *two-sided* univariate limit of the outer skeleton with the
    package's bounded limit policy.  This admits removable transcendental outer
    singularities such as ``sin(g)/g`` as well as ordinary analytic composition.
    """
    # Scalar outer reduction is sound only for a single-valued continuous/removable
    # outer germ. Piecewise and jump/discrete heads require their dedicated
    # accumulation-aware geometry; replacing an inner expression can otherwise
    # collapse a punctured branch to its exceptional target value.
    if expr.has(sp.Piecewise, sp.sign, sp.Heaviside, sp.floor, sp.ceiling):
        return None

    for inner, z, outer in _proper_composition_candidates(expr, variables):
        inner_result = limit(
            inner, variables, target, domain=domain, return_result=True
        )
        if inner_result.status is not LimitStatus.PROVED:
            continue
        center = inner_result.value
        if center is None or center in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            continue
        value = _two_sided_limit(outer, z, center)
        if (
            value is None
            or value in (sp.oo, -sp.oo, sp.zoo, sp.nan)
            or isinstance(value, (sp.AccumBounds, sp.Set))
        ):
            continue
        if value.has(*variables):
            continue
        return value, (
            LimitEvidence(
                "transcendental_inner_limit",
                "certified multivariate limit of the scalar composition argument",
                substitutions=((z, inner),),
                value=center,
            ),
            LimitEvidence(
                "transcendental_outer_asymptotic",
                "two-sided outer limit certified by existing univariate asymptotic machinery",
                substitutions=((z, inner),),
                value=value,
            ),
        )
    return None


def _piecewise_branch_limit_certificate(
    expr, variables, target, domain, assumptions=sp.S.true
):
    """Certify a Piecewise germ from exactly the branches that accumulate.

    Branch selection respects Piecewise priority.  For branch i the effective
    local domain is ``ambient & condition_i & ~condition_0 & ... & ~condition_{i-1}``.
    A branch whose punctured effective domain does not accumulate at the target
    is irrelevant to the limit.  Unknown accumulation or an unknown branch
    limit remains UNKNOWN.  Conflicting certified accumulating branch limits
    prove nonexistence.
    """
    if not isinstance(expr, sp.Piecewise):
        return None

    from .domain_cluster_geometry import local_domain_accumulates

    if (
        len(variables) == 1
        and target[0].is_finite is True
        and all(branch.is_rational_function(variables[0]) for branch, _ in expr.args)
    ):
        branch_values = [
            limit(
                branch,
                variables,
                target,
                domain=domain,
                assumptions=assumptions,
                return_result=True,
            )
            for branch, _ in expr.args
        ]
        if branch_values and all(
            item.status is LimitStatus.PROVED and item.value == branch_values[0].value
            for item in branch_values
        ):
            return (
                "proved",
                branch_values[0].value,
                (
                    LimitEvidence(
                        "piecewise_common_germ",
                        "Every branch has the same limit on the ambient approach, so its branch selector cannot change that value.",
                        value=branch_values[0].value,
                    ),
                ),
            )

    for _, condition in expr.args:
        if condition.free_symbols and not (condition.free_symbols & set(variables)):
            if bounded_ask(condition, assumptions) is None:
                return None

    previous = sp.S.false
    accumulating = []
    evidence = []
    for index, (branch_expr, condition) in enumerate(expr.args):
        effective = sp.simplify(sp.And(domain, condition, sp.Not(previous)))
        previous = sp.simplify(sp.Or(previous, condition))
        accumulates = local_domain_accumulates(effective, variables, target)
        if accumulates is None:
            return None
        if not accumulates:
            evidence.append(
                LimitEvidence(
                    "piecewise_nonaccumulating_branch",
                    f"Piecewise branch {index} has no punctured local accumulation",
                )
            )
            continue

        branch = limit(
            branch_expr,
            variables,
            target,
            domain=effective,
            assumptions=assumptions,
            return_result=True,
        )
        if branch.status is LimitStatus.UNKNOWN:
            return None
        if branch.status is LimitStatus.DOES_NOT_EXIST:
            return (
                "dne",
                None,
                tuple(evidence)
                + (
                    LimitEvidence(
                        "piecewise_branch_dne",
                        f"accumulating Piecewise branch {index} has no limit",
                    ),
                ),
            )
        accumulating.append(branch.value)
        evidence.append(
            LimitEvidence(
                "piecewise_accumulating_branch",
                f"Piecewise branch {index} accumulates and has certified local limit",
                value=branch.value,
            )
        )

    if not accumulating:
        return None

    first = accumulating[0]
    for value in accumulating[1:]:
        same = _exact_equal(first, value)
        if same is False:
            return (
                "dne",
                None,
                tuple(evidence)
                + (
                    LimitEvidence(
                        "piecewise_conflicting_germs",
                        "accumulating Piecewise branches have different certified limits",
                    ),
                ),
            )
        if same is not True:
            return None
    return ("proved", first, tuple(evidence))


def _growth_comparison_limit_certificate(
    expr, variable, target, assumptions, domain=sp.S.true
):
    if sp.sympify(expr).has(sp.Abs, sp.sign):
        return None
    if sp.sympify(expr).has(
        sp.Ei, sp.Ci, sp.expint, sp.polylog, sp.hankel1, sp.hankel2
    ):
        return None
    """Lift a certified scale comparison of a quotient into a scalar limit.

    This is restricted to positive real scale quotients at 0+.
    It never infers a signed infinity from magnitude alone unless the quotient
    is provably positive under the supplied assumptions.
    """
    if target != 0:
        return None
    if not (variable.is_positive is True or domain == (variable > target)):
        return None
    side_assumptions = sp.And(assumptions, domain)
    num, den = sp.fraction(sp.cancel(expr))
    from .relative_growth import GrowthScaleComparison, prove_growth_comparison

    proof = prove_growth_comparison(num, den, variable, target, side_assumptions)
    if not proof.certified:
        return None
    if proof.relation is GrowthScaleComparison.LESS:
        return sp.S.Zero, LimitEvidence(
            "growth_comparison",
            "certified numerator scale is asymptotically smaller than denominator scale",
            value=sp.S.Zero,
        )
    if proof.relation is GrowthScaleComparison.GREATER:
        positive = bounded_ask(sp.Q.positive(expr), side_assumptions)
        if positive is not True:
            # GrowthScale comparisons are magnitude comparisons.  For the
            # positive-side Hardy germs accepted here, certify sign by refining
            # numerator and denominator under the same side assumptions.
            rn, rd = sp.refine(num, side_assumptions), sp.refine(den, side_assumptions)

            def _positive_hardy_germ(e):
                if bounded_ask(sp.Q.positive(e), side_assumptions) is True:
                    return True
                if e.is_Pow:
                    base_positive = (
                        bounded_ask(sp.Q.positive(e.base), side_assumptions) is True
                    )
                    # On the 0+ germ, log(1/x) is eventually positive.
                    if e.base == sp.log(1 / variable) and domain == (variable > target):
                        base_positive = True
                    return (
                        base_positive
                        and bounded_ask(sp.Q.real(e.exp), side_assumptions) is True
                    )
                if e.is_Mul:
                    return all(_positive_hardy_germ(f) for f in e.args)
                return (
                    e.func is sp.exp
                    and bounded_ask(sp.Q.real(e.args[0]), side_assumptions) is True
                )

            positive = _positive_hardy_germ(rn) and _positive_hardy_germ(rd)
        if positive is True:
            return sp.oo, LimitEvidence(
                "growth_comparison",
                "certified numerator scale dominates denominator and quotient is positive",
                value=sp.oo,
            )
    return None


def _fast_univariate_expansion_limit(expr, variable, point):
    """Certify a two-sided scalar limit from matching adaptive local expansions."""
    from .local_expansion import adaptive_path_limit

    # Adaptive path expansions do not carry branch/remainder certificates for
    # these heads or for variable exponents. Dedicated scalar theorems must
    # handle them; an unsigned leading magnitude is not a side limit.
    if expr.has(
        sp.gamma,
        sp.hankel1,
        sp.hankel2,
        sp.Ei,
        sp.Ci,
        sp.expint,
        sp.polylog,
        sp.LambertW,
        sp.sign,
        sp.Heaviside,
        sp.floor,
        sp.ceiling,
        sp.arg,
        sp.asech,
    ):
        return None
    if any(p.exp.has(variable) for p in expr.atoms(sp.Pow)):
        return None

    t = sp.Dummy("_univariate_germ_t", positive=True)
    values = []
    signs = (1,) if variable.is_positive is True and point == 0 else (-1, 1)
    for sign in signs:
        along = sp.sympify(expr).subs(variable, sp.sympify(point) + sign * t)
        expanded = adaptive_path_limit(along, t)
        if expanded is None:
            return None
        values.append(expanded[0])
    if len(values) == 1 or _exact_equal(values[0], values[1]) is True:
        return values[0]
    return None


def unresolved_origin_parameter(expr, variables, target):
    """Identify origin germs whose free parameter changes branch or order.

    Call after exact recurrences and supported parameter stratification. An
    unresolved path expression containing this parameter cannot establish a
    universal nonexistence claim.
    """
    parameters = set(variables)
    center = dict(zip(variables, target, strict=True))
    for power in sp.sympify(expr).atoms(sp.Pow):
        if (
            power.exp.free_symbols - parameters
            and power.exp.is_positive is None
            and power.exp.is_zero is None
            and power.base.subs(center) == 0
        ):
            return "undetermined_singular_power"
    for atom in sp.sympify(expr).atoms(sp.besselj, sp.bessely):
        order = atom.args[0]
        if (
            order.free_symbols - parameters
            and order.is_positive is None
            and order.is_zero is None
            and atom.args[1].subs(center) == 0
        ):
            return "undetermined_bessel_order"
    return None
