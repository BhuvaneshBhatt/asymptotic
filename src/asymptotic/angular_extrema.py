"""Exact low-dimensional extrema used by multivariate limit providers."""

from __future__ import annotations

import sympy as sp


def _certified_extremum(objective, constraints, variables, *, kind):
    """Return a certified exact extremum, preferring semialg and then symbopt.

    The symbopt route is optional: asymptotic must remain usable
    without the general optimizer, while angular optimization can exploit it for
    problems outside semialg's preferred optimization route.
    """
    try:
        from semialg import semialgebraic_maximize, semialgebraic_minimize

        solver = semialgebraic_maximize if kind == "max" else semialgebraic_minimize
        result = solver(objective, constraints, variables, return_result=True)
        value = getattr(result, "value", None)
        if getattr(result, "certified", False):
            return sp.sympify(value), "semialg"
        # Some exact critical-point enumerations find the right algebraic value
        # but conservatively leave ``certified=False`` when one enumeration
        # branch was incomplete.  Do not discard that candidate: independently
        # certify the global bound and attainment with semialg decision logic.
        if value is not None:
            from semialg import implies, is_satisfiable

            candidate = sp.sympify(value)
            bound = (
                sp.Le(objective, candidate)
                if kind == "max"
                else sp.Ge(objective, candidate)
            )
            if (
                implies(constraints, bound, variables) is True
                and is_satisfiable(
                    sp.And(constraints, sp.Eq(objective, candidate)), variables
                )
                is True
            ):
                return candidate, "semialg_verified_candidate"
    except (
        ImportError,
        ModuleNotFoundError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        pass

    # Keep optional optimization bounded before importing its heavier runtime.
    # The adapter supports quadratic objectives with linear constraints.
    # Algebraic lifts and Boolean alternatives require additional domain and
    # coverage certificates.
    relations = sp.And.make_args(sp.sympify(constraints))
    if relations == (sp.S.true,):
        relations = ()
    if (
        not 1 <= len(variables) <= 2
        or len(relations) > 4
        or any(not getattr(relation, "is_Relational", False) for relation in relations)
        or sp.count_ops(objective) + sp.count_ops(constraints) > 60
    ):
        return None
    try:
        if sp.Poly(objective, *variables).total_degree() > 2:
            return None
        if any(
            sp.Poly(r.lhs - r.rhs, *variables).total_degree() > 1 for r in relations
        ):
            return None
    except sp.PolynomialError:
        return None

    try:
        import symbopt

        solver = symbopt.maximize if kind == "max" else symbopt.minimize
        result = solver(objective, list(relations), variables=variables)
        if getattr(result, "certified", False):
            value = getattr(result, "optimum_value", getattr(result, "value", None))
            if value is not None:
                return sp.sympify(value), "symbopt"
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        pass
    return None


def _sector_simplex_constraints(sector, angular, z):
    """Map square-invariant angular sector constraints through ``z_i=u_i**2``."""
    if sector is sp.S.true or sector is True:
        return sp.S.true
    replacements = {u**2: zi for u, zi in zip(angular, z, strict=True)}

    def convert(node):
        if node.func in (sp.And, sp.Or):
            return node.func(*(convert(arg) for arg in node.args))
        if node.func is sp.Not:
            return sp.Not(convert(node.args[0]))
        if getattr(node, "is_Relational", False):
            lhs = sp.expand(node.lhs).xreplace(replacements)
            rhs = sp.expand(node.rhs).xreplace(replacements)
            if lhs.has(*angular) or rhs.has(*angular):
                return None
            return node.func(lhs, rhs)
        mapped = sp.expand(node).xreplace(replacements)
        return None if mapped.has(*angular) else mapped

    return convert(sp.sympify(sector))


def _two_angular_sector_extremum(objective, angular, sector, *, kind):
    """Optimize a two-angle sphere sector through its exact ratio coordinate.

    On ``u_i**2 >= u_j**2`` put ``t=u_j/u_i``.  Then ``-1 <= t <= 1`` and
    ``u_i**2=1/(1+t**2)``.  For the squared homogeneous angular objectives used
    by relation certification, the sign of ``u_i`` cancels and this gives an
    exact compact univariate rational optimization problem.
    """
    if len(angular) != 2:
        return None
    u0, u1 = angular
    sectors = (
        (sp.Ge(u0**2, u1**2), u0, u1),
        (sp.Ge(u1**2, u0**2), u1, u0),
    )
    match = next(
        ((pivot, other) for expected, pivot, other in sectors if sector == expected),
        None,
    )
    if match is None:
        return None
    pivot, other = match
    t = sp.Dummy("_t", real=True)
    q = sp.Dummy("_q", positive=True)
    transformed = sp.cancel(
        sp.together(objective.subs({pivot: q, other: t * q}, simultaneous=True))
    )
    # Squared objectives are invariant under the pivot sign.  Replace q**2 by
    # the sphere value and reject anything that still depends on q.
    transformed = sp.powdenest(transformed, force=True)
    num, den = map(sp.expand, sp.fraction(transformed))
    try:
        pnum = sp.Poly(num, q)
        pden = sp.Poly(den, q)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    # Homogeneity gives one q-degree in each polynomial.  The degrees need not
    # agree because an angular coefficient may only become homogeneous after
    # imposing the sphere equation.  Substitute q**2 = 1/(1+t**2) exactly.
    if pnum.is_zero:
        return sp.S.Zero, "sector_ratio_identity"
    if (
        len({m[0] for m, _ in pnum.terms()}) != 1
        or len({m[0] for m, _ in pden.terms()}) != 1
    ):
        return None
    dn = pnum.monoms()[0][0]
    dd = pden.monoms()[0][0]
    if dn % 2 or dd % 2:
        return None
    num_coeff = sp.cancel(num / q**dn)
    den_coeff = sp.cancel(den / q**dd)
    sphere_scale = 1 + t**2
    reduced = sp.cancel(
        sp.together(
            (num_coeff * sphere_scale ** (dd // 2))
            / (den_coeff * sphere_scale ** (dn // 2))
        )
    )
    if reduced.has(q):
        return None
    constraints = sp.And(sp.Ge(t, -1), sp.Le(t, 1))
    result = _certified_extremum(reduced, constraints, (t,), kind=kind)
    if result is None:
        return None
    return result[0], f"sector_ratio_{result[1]}"


def _angular_squared_extremum(objective, angular, *, kind, sector=sp.S.true):
    """Optimize an angular square on a sphere, retaining sector inequalities.

    Even objectives use ``z_i=u_i**2``.  Max-coordinate sector inequalities
    become ordinary simplex inequalities such as ``z_i >= z_j`` before the
    final simplex coordinate is eliminated.  In two dimensions, non-even
    squared homogeneous objectives use the exact ratio coordinate on each
    max-coordinate sector, reducing them to compact univariate optimization.
    """
    numerator, denominator = sp.fraction(sp.cancel(sp.together(objective)))
    try:
        pnum = sp.Poly(sp.expand(numerator), *angular)
        pden = sp.Poly(sp.expand(denominator), *angular)
    except (sp.PolynomialError, TypeError, ValueError):
        pnum = pden = None
    even = (
        pnum is not None
        and pden is not None
        and all(
            not any(exponent % 2 for exponent in monom)
            for poly in (pnum, pden)
            for monom, _ in poly.terms()
        )
    )
    if not even:
        # Square-simplex reduction is limited to sign-invariant
        # objectives.  Let the caller use its inexpensive sphere coefficient
        # bound rather than forcing a substantially harder CAD optimization.
        return None
    z = tuple(sp.Dummy(f"_z{i}", real=True) for i in range(len(angular)))

    def square_map(poly):
        return sp.Add(
            *(
                coeff
                * sp.Mul(
                    *(
                        zi ** (exponent // 2)
                        for zi, exponent in zip(z, monom, strict=True)
                    )
                )
                for monom, coeff in poly.terms()
            )
        )

    squared = sp.cancel(sp.together(square_map(pnum) / square_map(pden)))
    mapped_sector = _sector_simplex_constraints(sector, angular, z)
    if mapped_sector is None:
        return _certified_extremum(
            objective,
            sp.And(sp.Eq(sp.Add(*(u**2 for u in angular)), 1), sector),
            angular,
            kind=kind,
        )
    if len(z) == 1:
        if mapped_sector is sp.S.false:
            return None
        return sp.simplify(squared.subs(z[0], 1)), "sphere_identity"
    free = z[:-1]
    last = 1 - sp.Add(*free)
    reduced = sp.expand(squared.subs(z[-1], last))
    mapped_sector = sp.sympify(mapped_sector).subs(z[-1], last)
    constraints = sp.And(
        *(sp.Ge(zi, 0) for zi in free),
        sp.Ge(last, 0),
        mapped_sector,
    )
    result = _certified_extremum(reduced, constraints, free, kind=kind)
    if result is None:
        return None
    return result[0], f"sector_simplex_{result[1]}"
