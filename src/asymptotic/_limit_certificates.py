"""Proof-aware simultaneous real multivariate limits.

A simultaneous Euclidean limit ranges over the full admissible approach domain,
not an iterated sequence of one-variable limits. The solver combines exact
reductions, uniform bounds, path certificates, and local geometric methods;
unsupported cases return an explicit unknown status.
"""

from __future__ import annotations

import sympy as sp

from ._limit_paths import _newton_weight_vectors
from ._symbolic_policy import bounded_ask
from .angular_extrema import _angular_squared_extremum
from .limit_models import (
    LimitEvidence,
)


def _lowest_homogeneous_part(poly_expr, variables):
    try:
        poly = sp.Poly(sp.expand(poly_expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if poly.is_zero:
        return sp.oo, sp.S.Zero
    terms = poly.terms()
    order = min(sum(monom) for monom, _ in terms)
    leading = sp.Add(
        *(
            coeff * sp.Mul(*(v**e for v, e in zip(variables, monom, strict=True)))
            for monom, coeff in terms
            if sum(monom) == order
        )
    )
    return order, sp.expand(leading)


def _rational_order_certificate(expr, variables, target, candidate, domain):
    """Prove a rational limit from exact local orders when denominator is elliptic.

    After translating the target to the origin and cancelling common factors,
    if ``ord(p-Lq) > ord(q)`` and the lowest homogeneous denominator part has
    no zero on the unit sphere, compactness gives a uniform denominator lower
    bound and hence ``p/q -> L``.  The sphere test is exact semialgebraic
    satisfiability, not numerical sampling.
    """
    shifted = sp.cancel(
        sp.together(
            expr.subs(
                {v: v + a for v, a in zip(variables, target, strict=True)},
                simultaneous=True,
            )
        )
    )
    numerator, denominator = sp.fraction(shifted)
    residual = sp.expand(numerator - candidate * denominator)
    if domain is not sp.S.true:
        try:
            from semialg import implies

            original_num, original_den = sp.fraction(sp.cancel(sp.together(expr)))
            original_residual = sp.expand(original_num - candidate * original_den)
            if implies(domain, sp.Eq(original_residual, 0), variables):
                return LimitEvidence(
                    "rational_domain_identity",
                    "domain implies exact equality to the candidate on the rational function domain",
                    value=candidate,
                )
        except (
            ImportError,
            ModuleNotFoundError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
        ):
            pass
        return None
    rpart = _lowest_homogeneous_part(residual, variables)
    qpart = _lowest_homogeneous_part(denominator, variables)
    if rpart is None or qpart is None:
        return None
    rorder, _ = rpart
    qorder, qleading = qpart
    if residual == 0:
        return LimitEvidence(
            "rational_exact_cancellation",
            "candidate equals the rational function on its punctured domain",
            value=candidate,
        )
    if rorder is sp.oo or qorder is sp.oo or rorder <= qorder:
        return None
    if qorder == 0:
        nonvanishing = qleading != 0
    else:
        try:
            from semialg import is_satisfiable

            sphere = sp.Eq(sp.Add(*(v**2 for v in variables)), 1)
            nonvanishing = not is_satisfiable(
                sp.And(sphere, sp.Eq(qleading, 0)), variables
            )
        except (
            ImportError,
            ModuleNotFoundError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
        ):
            return None
    if not nonvanishing:
        return None
    return LimitEvidence(
        "rational_dominant_order",
        f"residual order {rorder} exceeds denominator order {qorder} with nonvanishing leading denominator form",
        value=candidate,
    )


def _weighted_homogeneous_factor(poly_expr, variables, weights, radius, angular):
    """Return ``(order, angular_form)`` for an exactly weighted-homogeneous polynomial."""
    try:
        poly = sp.Poly(sp.expand(poly_expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    if poly.is_zero:
        return sp.oo, sp.S.Zero
    degrees = {
        sum(e * w for e, w in zip(monom, weights, strict=True))
        for monom, _ in poly.terms()
    }
    if len(degrees) != 1:
        return None
    order = next(iter(degrees))
    substitutions = {
        variable: radius**weight * angle
        for variable, weight, angle in zip(variables, weights, angular, strict=True)
    }
    transformed = sp.expand(poly.as_expr().subs(substitutions, simultaneous=True))
    angular_form = sp.cancel(transformed / radius**order)
    if angular_form.has(radius):
        return None
    return order, sp.expand(angular_form)


def _positive_definite_pure_even_form(expr, variables):
    """Cheap exact positivity certificate on the unit sphere.

    An even-monomial polynomial with nonnegative coefficients and, for every variable, a
    positive pure even power can vanish only at the origin.  Hence it has a
    strictly positive minimum on the compact unit sphere.  This recognizes only a sufficient class; it avoids invoking optimization/CAD.
    """
    try:
        poly = sp.Poly(sp.expand(expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return False
    if poly.is_zero:
        return False
    covered = set()
    for monom, coeff in poly.terms():
        if coeff.is_nonnegative is not True or any(exponent % 2 for exponent in monom):
            return False
        nonzero = [i for i, exponent in enumerate(monom) if exponent]
        if len(nonzero) == 1:
            i = nonzero[0]
            if monom[i] % 2 == 0 and coeff.is_positive is True:
                covered.add(i)
    return len(covered) == len(variables)


def _sphere_polynomial_bounded(expr, variables):
    """Certify boundedness of a polynomial on the unit sphere.

    Every polynomial is continuous and the sphere is compact.  Keeping this
    as a named certificate avoids computing an unnecessary exact maximum when
    only finiteness is required for a squeeze argument.
    """
    try:
        sp.Poly(sp.expand(expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return False
    return True


def _weighted_blowup_certificate(expr, variables, target, candidate, domain):
    """Certify a rational limit by a radial or weighted homogeneous blow-up.

    With ``x_i-a_i = r**w_i*u_i`` and ``sum(u_i**2)=1``, an exactly
    weighted-homogeneous residual/denominator pair becomes
    ``r**m R(u) / (r**n Q(u))``.  When ``m>n``, a certified positive minimum
    of ``Q(u)**2`` and finite maximum of ``R(u)**2`` give a *uniform* angular
    bound, so the transformed expression converges independently of angle.

    This is stronger than testing individual weighted paths: optimization on
    the compact angular sphere certifies all directions simultaneously.
    """
    if domain is not sp.S.true:
        return None
    shifted = sp.cancel(
        sp.together(
            expr.subs(
                {v: v + a for v, a in zip(variables, target, strict=True)},
                simultaneous=True,
            )
        )
    )
    numerator, denominator = sp.fraction(shifted)
    residual = sp.expand(numerator - candidate * denominator)
    if residual == 0:
        return None
    try:
        sp.Poly(residual, *variables)
        sp.Poly(denominator, *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None

    weight_candidates = {(1,) * len(variables)}
    weight_candidates.update(
        _newton_weight_vectors(expr - candidate, variables, target)
    )
    radius = sp.Dummy("_blowup_r", positive=True)
    angular = tuple(sp.Dummy(f"_u{i}", real=True) for i in range(len(variables)))

    for weights in sorted(weight_candidates, key=lambda w: (sum(w), w)):
        rpart = _weighted_homogeneous_factor(
            residual, variables, weights, radius, angular
        )
        qpart = _weighted_homogeneous_factor(
            denominator, variables, weights, radius, angular
        )
        if rpart is None or qpart is None:
            continue
        rorder, rangular = rpart
        qorder, qangular = qpart
        if rorder is sp.oo or qorder is sp.oo or rorder <= qorder:
            continue

        # For the common positive-definite weighted denominator case we only
        # need existence of a positive sphere minimum and a finite numerator
        # maximum.  Compactness proves both without solving the extrema.
        if _positive_definite_pure_even_form(
            qangular, angular
        ) and _sphere_polynomial_bounded(rangular, angular):
            qmin = sp.Symbol("m_Q", positive=True)
            rmax = sp.Symbol("M_R", finite=True, nonnegative=True)
            qprovider = "positive_definite_even_form+compact_sphere"
            rprovider = "polynomial_compact_sphere"
        else:
            qmin_result = _angular_squared_extremum(
                sp.expand(qangular**2), angular, kind="min"
            )
            if qmin_result is None:
                continue
            qmin, qprovider = qmin_result
            positive = bounded_ask(sp.Q.positive(qmin))
            if positive is not True:
                continue
            rmax_result = _angular_squared_extremum(
                sp.expand(rangular**2), angular, kind="max"
            )
            if rmax_result is None:
                continue
            rmax, rprovider = rmax_result
            if rmax.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                continue
        method = (
            "radial_uniform_angular_optimization"
            if all(weight == 1 for weight in weights)
            else "weighted_blowup_uniform_angular_optimization"
        )
        return LimitEvidence(
            method,
            "uniform angular certificate for weights "
            f"{weights}: residual order {rorder} > denominator order {qorder}; "
            f"min(Q(u)^2)={sp.sstr(qmin)} via {qprovider}, "
            f"max(R(u)^2)={sp.sstr(rmax)} via {rprovider}",
            value=candidate,
        )
    return None


def _rational_bad_set(expr, variables, candidate, domain, epsilon):
    """Polynomialize ``|expr-candidate| >= epsilon`` for a rational function."""
    try:
        together = sp.cancel(sp.together(expr))
        numerator, denominator = sp.fraction(together)
        sp.Poly(numerator, *variables)
        sp.Poly(denominator, *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    residual = sp.expand(numerator - candidate * denominator)
    return sp.And(
        domain,
        sp.Ne(denominator, 0),
        sp.Ge(sp.expand(residual**2 - epsilon**2 * denominator**2), 0),
    )


def _substitute_explicit_graph_auxiliaries(formula, auxiliaries):
    """Eliminate graph auxiliaries that are explicitly defined by equalities.

    Function-graph construction keeps branch/domain information
    explicit.  Many auxiliaries, however, are merely names for polynomial
    subexpressions.  Substituting those definitions before CAD can remove
    entire lifting dimensions without changing the represented set.
    """
    formula = sp.sympify(formula)
    remaining = list(auxiliaries)
    changed = True
    while changed:
        changed = False
        # Equalities inside an alternative branch do not define an auxiliary
        # globally. Only unconditional conjuncts can be substituted here.
        equalities = tuple(
            term for term in sp.And.make_args(formula) if isinstance(term, sp.Equality)
        )
        for variable in tuple(remaining):
            replacement = None
            for equality in equalities:
                try:
                    polynomial = sp.Poly(equality.lhs - equality.rhs, variable)
                except sp.PolynomialError:
                    continue
                if polynomial.degree() != 1:
                    continue
                coefficient = polynomial.coeff_monomial(variable)
                # Dividing by a moving coefficient loses the whole fiber
                # where it vanishes, even if a solver reports one formula.
                if (
                    coefficient.free_symbols
                    or coefficient.is_finite is not True
                    or coefficient.is_zero is not False
                ):
                    continue
                replacement = -polynomial.coeff_monomial(1) / coefficient
                break
            if replacement is None:
                continue
            formula = formula.subs(variable, replacement, simultaneous=True)
            remaining.remove(variable)
            changed = True
            break
    return formula, tuple(remaining)


def _reduced_semialgebraic_bad_set(expr, variables, candidate, domain, epsilon):
    """Construct the epsilon-bad set, eliminating graph auxiliaries first."""
    try:
        from semialg import (
            UnsupportedFunctionGraph,
            quantifier_eliminate,
            semialgebraic_function_graph,
        )
    except (ImportError, ModuleNotFoundError):
        return None

    rational = _rational_bad_set(expr, variables, candidate, domain, epsilon)
    if rational is not None:
        return rational, "rational_polynomial_bad_set"

    value_symbol = sp.Dummy("_limit_value", real=True)
    try:
        graph = semialgebraic_function_graph(expr, value_symbol)
    except UnsupportedFunctionGraph:
        return None
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        return None
    graph_formula, remaining_auxiliaries = _substitute_explicit_graph_auxiliaries(
        graph.formula, graph.auxiliary_variables
    )
    bad_graph = sp.And(
        domain,
        graph_formula,
        sp.Ge((value_symbol - candidate) ** 2 - epsilon**2, 0),
    )
    eliminated = (value_symbol, *remaining_auxiliaries)
    if not eliminated:
        return bad_graph, "semialgebraic_bad_set"
    try:
        try:
            reduced = quantifier_eliminate(
                bad_graph,
                quantifiers=[*(("exists", variable) for variable in eliminated)],
                variables=(*variables, epsilon, *eliminated),
            )
        except ArithmeticError:
            # Reduced CAD can occasionally fail while isolating an algebraic
            # fiber even though complete Collins CAD can decide the same
            # projection.  The algebraic-graph path retries the exact complete backend instead of
            # weakening the certificate.
            reduced = quantifier_eliminate(
                bad_graph,
                quantifiers=[*(("exists", variable) for variable in eliminated)],
                variables=(*variables, epsilon, *eliminated),
                strategy="collins",
            )
    except (
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return None
    return reduced, "semialgebraic_graph_elimination"


def _safe_target_fiber(formula, substitutions):
    """Substitute a target into a Boolean formula without evaluating dead 0/0 branches."""
    if formula is sp.S.true or formula is sp.S.false:
        return formula
    if isinstance(formula, sp.And):
        kept = []
        for arg in formula.args:
            value = _safe_target_fiber(arg, substitutions)
            if value is sp.S.false:
                return sp.S.false
            if value is not sp.S.true:
                kept.append(value)
        return sp.And(*kept)
    if isinstance(formula, sp.Or):
        kept = []
        for arg in formula.args:
            value = _safe_target_fiber(arg, substitutions)
            if value is sp.S.true:
                return sp.S.true
            if value is not sp.S.false:
                kept.append(value)
        return sp.Or(*kept)
    if isinstance(formula, sp.Not):
        value = _safe_target_fiber(formula.args[0], substitutions)
        return sp.Not(value)
    try:
        return formula.subs(substitutions, simultaneous=True)
    except (TypeError, ValueError, ZeroDivisionError):
        return formula.xreplace(substitutions, hack2=True) if False else formula


def _rational_local_closure_certificate(expr, variables, target, candidate, domain):
    """Certify a rational limit by a local CAD closure computation.

    For epsilon > 0 let B be the exact semialgebraic set of approach points
    where ``|expr-candidate| >= epsilon``.  The limit holds exactly when the
    closure of B in ``(variables, epsilon)`` has no point over the target with
    positive epsilon.  This is the epsilon-delta condition with the local
    distance quantifier absorbed by semialgebraic closure, leaving only a
    one-parameter satisfiability question after CAD closure construction.
    """
    epsilon = sp.Dummy("_local_epsilon", real=True)
    bad_set = _rational_bad_set(expr, variables, candidate, domain, epsilon)
    if bad_set is None:
        return None
    distance2 = sp.Add(*((v - a) ** 2 for v, a in zip(variables, target, strict=True)))
    punctured_bad = sp.And(sp.Gt(epsilon, 0), sp.Gt(distance2, 0), bad_set)
    try:
        from semialg import is_satisfiable, region_closure

        closure = region_closure(punctured_bad, (*variables, epsilon))
        target_fiber = _safe_target_fiber(
            closure, dict(zip(variables, target, strict=True))
        )
        if target_fiber.has(*variables):
            return None
        obstruction = is_satisfiable(
            sp.And(sp.Gt(epsilon, 0), target_fiber), (epsilon,)
        )
    except (
        ImportError,
        ModuleNotFoundError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return None
    if obstruction is False:
        return (
            True,
            LimitEvidence(
                "rational_local_cad_closure",
                "no positive-error fiber remains in the CAD closure of the local rational bad set",
                value=candidate,
            ),
        )
    if obstruction is True:
        return (
            None,
            LimitEvidence(
                "rational_local_cad_closure_rejected",
                "a positive-error fiber remains in the CAD closure of the local rational bad set",
                value=candidate,
            ),
        )
    return None


def _semialgebraic_local_closure_certificate(
    expr, variables, target, candidate, domain
):
    """Certify an algebraic/semialgebraic limit by local CAD closure.

    The exact semialgebraic function graph is converted to the epsilon-bad
    inequality ``|f-L| >= epsilon`` and its value/branch auxiliary variables
    are existentially eliminated first.  CAD then computes the closure of the
    resulting bad set in ``(x, epsilon)``.  Absence of a positive-epsilon
    fiber over the target is precisely the local inequality certificate
    needed for the limit.

    Rational functions have a more direct polynomial bad-set implementation
    by the polynomial/rational certificate and are left to that cheaper path first.
    """
    epsilon = sp.Dummy("_local_epsilon", real=True)
    try:
        reduced = _reduced_semialgebraic_bad_set(
            expr, variables, candidate, domain, epsilon
        )
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        # Unsupported algebraic graph reductions are an inconclusive backend
        # route, not a user-visible limit failure.  Other proof engines may
        # still certify the candidate.
        return None
    if reduced is None:
        return None
    bad_set, reduction_method = reduced
    # The rational case is handled by the cheaper polynomial/rational certificate before this
    # function is called.  Avoid presenting its generic reduction as an
    # algebraic-graph certificate if the cheaper certificate declines.
    if reduction_method == "rational_polynomial_bad_set":
        return None

    distance2 = sp.Add(*((v - a) ** 2 for v, a in zip(variables, target, strict=True)))
    punctured_bad = sp.And(sp.Gt(epsilon, 0), sp.Gt(distance2, 0), bad_set)
    try:
        from semialg import is_satisfiable, region_closure

        closure = region_closure(punctured_bad, (*variables, epsilon))
        target_fiber = _safe_target_fiber(
            closure, dict(zip(variables, target, strict=True))
        )
        if target_fiber.has(*variables):
            return None
        obstruction = is_satisfiable(
            sp.And(sp.Gt(epsilon, 0), target_fiber), (epsilon,)
        )
    except (
        ImportError,
        ModuleNotFoundError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return None

    if obstruction is False:
        return (
            True,
            LimitEvidence(
                "semialgebraic_local_cad_closure",
                f"no positive-error target fiber remains after {reduction_method} and local CAD closure",
                value=candidate,
            ),
        )
    if obstruction is True:
        return (
            None,
            LimitEvidence(
                "semialgebraic_local_cad_closure_rejected",
                f"a positive-error target fiber remains after {reduction_method} and local CAD closure",
                value=candidate,
            ),
        )
    return None


def _semialgebraic_certify_candidate(expr, variables, target, candidate, domain):
    """Certify a finite candidate using a reduced epsilon-bad-set problem.

    The semialgebraic certificate first checks that the punctured approach domain accumulates at the
    target.  Rational functions are polynomialized directly.  General
    supported semialgebraic graphs have their value and auxiliary variables
    existentially eliminated *before* the final epsilon-delta QE call.  This
    avoids sending the full graph through the alternating quantifier block.
    """
    try:
        from semialg import quantifier_eliminate
    except (ImportError, ModuleNotFoundError):
        return None

    domain = sp.sympify(domain)
    distance2 = sp.Add(*((v - a) ** 2 for v, a in zip(variables, target, strict=True)))
    punctured_domain = sp.And(domain, sp.Gt(distance2, 0))
    try:
        from .domain_witnesses import domain_accumulates

        closure_result = domain_accumulates(punctured_domain, variables, target)
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        return None
    if not closure_result:
        return (
            False,
            LimitEvidence(
                "semialgebraic_approach_closure",
                "target is not in the closure of the punctured approach domain",
            ),
        )

    order_certificate = _rational_order_certificate(
        expr, variables, target, candidate, domain
    )
    if order_certificate is not None:
        return True, order_certificate

    blowup_certificate = _weighted_blowup_certificate(
        expr, variables, target, candidate, domain
    )
    if blowup_certificate is not None:
        return True, blowup_certificate

    local_closure = _rational_local_closure_certificate(
        expr, variables, target, candidate, domain
    )
    if local_closure is not None:
        verdict, evidence = local_closure
        if verdict is True:
            return local_closure
        # A rejected candidate is useful evidence, but another generated
        # candidate may still be certifiable.  Return the rejection without
        # promoting it to a nonexistence proof.
        return verdict, evidence

    algebraic_closure = _semialgebraic_local_closure_certificate(
        expr, variables, target, candidate, domain
    )
    if algebraic_closure is not None:
        verdict, evidence = algebraic_closure
        if verdict is True:
            return algebraic_closure
        return verdict, evidence

    epsilon = sp.Dummy("_epsilon", real=True)
    try:
        reduced = _reduced_semialgebraic_bad_set(
            expr, variables, candidate, domain, epsilon
        )
    except (sp.PolynomialError, TypeError, ValueError, NotImplementedError):
        # Unsupported algebraic graph reductions are an inconclusive backend
        # route, not a user-visible limit failure.  Other proof engines may
        # still certify the candidate.
        return None
    if reduced is None:
        return None
    bad_set, reduction_method = reduced

    delta = sp.Dummy("_delta", real=True)
    annulus = sp.And(sp.Gt(distance2, 0), sp.Lt(distance2, delta**2))

    # Eliminate approach variables before introducing the forall-epsilon /
    # exists-delta alternation.  The resulting obstruction is only a relation
    # in (epsilon, delta): it says that an epsilon-bad point exists inside the
    # punctured delta-ball.  This is substantially smaller than asking CAD to
    # process the original forall x block together with graph variables.
    try:
        obstruction = quantifier_eliminate(
            sp.And(annulus, bad_set),
            quantifiers=[*(("exists", variable) for variable in variables)],
            variables=(epsilon, delta, *variables),
        )
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        return None

    quantified_matrix = sp.Implies(
        sp.Gt(epsilon, 0),
        sp.And(sp.Gt(delta, 0), sp.Not(obstruction)),
    )
    try:
        certified = quantifier_eliminate(
            quantified_matrix,
            quantifiers=[("forall", epsilon), ("exists", delta)],
            variables=(epsilon, delta),
        )
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        return None
    if certified in (sp.true, True):
        return (
            True,
            LimitEvidence(
                "semialgebraic_reduced_epsilon_delta",
                f"exact epsilon-delta proof after {reduction_method}",
                value=candidate,
            ),
        )
    if certified in (sp.false, False):
        return (
            None,
            LimitEvidence(
                "semialgebraic_reduced_epsilon_delta_rejected",
                f"exact reduced epsilon-delta QE rejects candidate after {reduction_method}",
                value=candidate,
            ),
        )
    return None
