"""Shackell-style relative-growth regimes as generalized valuation cones."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._symbolic_policy import (
    bounded_ask,
    bounded_limit,
    bounded_refine,
    bounded_solve_system,
)
from .blowup_geometry import Valuation
from .coverage import CoverageCertificate


@dataclass(frozen=True)
class RelativeGrowthRelation:
    dependent: sp.Symbol
    reference: sp.Symbol
    exponent: sp.Expr
    kind: str = "power_comparable"
    condition: sp.Expr = sp.S.true


@dataclass(frozen=True)
class GrowthValuationRegion:
    """Growth metadata around the common geometric Valuation object."""

    relations: tuple[RelativeGrowthRelation, ...]
    valuation: Valuation | None
    symbolic_weights: tuple[sp.Expr, ...]
    log_weights: tuple[sp.Expr, ...]
    condition: sp.Expr
    coverage: CoverageCertificate
    provider: str = "relative_growth_valuation"

    @property
    def weights(self):
        return (
            self.valuation.weights
            if self.valuation is not None
            else self.symbolic_weights
        )


def _parse_relation(rel, variables):
    # Explicit Eq(log|y|/log|x|, alpha) and Eq(y, x**alpha) are accepted.
    if isinstance(rel, RelativeGrowthRelation):
        return rel
    if not isinstance(rel, sp.Equality):
        return None
    a, b = rel.lhs, rel.rhs
    for y in variables:
        for x in variables:
            if x == y:
                continue
            ratio = sp.log(sp.Abs(y)) / sp.log(sp.Abs(x))
            if sp.simplify(a - ratio) == 0 and not b.has(*variables):
                return RelativeGrowthRelation(y, x, sp.sympify(b), "log_ratio", rel)
            if a == y and b.is_Pow and b.base == x and not b.exp.has(*variables):
                return RelativeGrowthRelation(y, x, b.exp, "power_relation", rel)
    return None


def relative_growth_valuation_fan(variables, conditions):
    """Translate certified relative-growth relations into candidate valuation cones."""
    variables = tuple(variables)
    conds = sp.And.make_args(sp.sympify(conditions))
    relations = tuple(
        r for c in conds if (r := _parse_relation(c, variables)) is not None
    )
    if not relations:
        return (), CoverageCertificate.unknown(
            "relative_growth", "no supported relative-growth relations"
        )
    # Solve w_y = alpha*w_x; normalize one free reference weight to 1.
    ws = sp.symbols(f"_w0:{len(variables)}", positive=True)
    equations = []
    for r in relations:
        equations.append(
            sp.Eq(
                ws[variables.index(r.dependent)],
                r.exponent * ws[variables.index(r.reference)],
            )
        )
    equations.append(sp.Eq(ws[0], 1))
    sol = bounded_solve_system(equations, ws, allow_general=True)
    cones = []
    for s in sol:
        weights = tuple(sp.simplify(s.get(w, w)) for w in ws)
        if any(w.free_symbols & set(ws) for w in weights):
            continue
        cones.append(
            GrowthValuationRegion(
                relations,
                Valuation(tuple(variables), weights),
                weights,
                (sp.S.Zero,) * len(weights),
                sp.And(*[r.condition for r in relations]),
                CoverageCertificate.complete(
                    "relative_growth_relation",
                    "explicit relative-growth assumptions determine this valuation cone",
                    (weights,),
                ),
            )
        )
    coverage = (
        CoverageCertificate.complete(
            "relative_growth_fan",
            "all explicitly supplied supported growth relations were translated",
            tuple(c.weights for c in cones),
        )
        if cones
        else CoverageCertificate.unknown(
            "relative_growth", "relations did not determine a valuation cone"
        )
    )
    return tuple(cones), coverage


def exp_log_order(expr, variables, cone):
    """Generalized leading order: radial valuation plus exponential/log layers."""
    expr = sp.sympify(expr)
    weights = cone.weights

    # Polynomial/rational base order.
    def poly_order(e):
        try:
            p = sp.Poly(e, *variables)
            return sp.Min(
                *[
                    sum(m[i] * weights[i] for i in range(len(variables)))
                    for m, c in p.terms()
                    if c
                ]
            )
        except (sp.PolynomialError, ValueError, TypeError):
            return None

    if expr.func is sp.exp:
        inner = exp_log_order(expr.args[0], variables, cone)
        return ("exp", inner)
    if expr.func is sp.log:
        inner = exp_log_order(expr.args[0], variables, cone)
        return ("log", inner)
    num, den = sp.fraction(sp.cancel(expr))
    no, do = poly_order(num), poly_order(den)
    if no is not None and do is not None:
        return ("power", sp.simplify(no - do))
    return ("unknown", expr)


@dataclass(frozen=True)
class GrowthAtom:
    expression: sp.Expr
    kind: str
    depth: int
    variables: tuple[sp.Symbol, ...]


@dataclass(frozen=True)
class GrowthCell:
    relations: tuple[sp.Expr, ...]
    dominant_atoms: tuple[GrowthAtom, ...]
    valuation_cones: tuple[GrowthValuationRegion, ...]
    condition: sp.Expr
    coverage: CoverageCertificate
    provider: str = "exp_log_growth_cell"


@dataclass(frozen=True)
class ExpLogValuationFan:
    atoms: tuple[GrowthAtom, ...]
    cells: tuple[GrowthCell, ...]
    coverage: CoverageCertificate
    provider: str = "exp_log_valuation_fan"


def _growth_depth(e):
    if not e.args:
        return 0
    extra = 1 if e.func in (sp.exp, sp.log) else 0
    return extra + max((_growth_depth(a) for a in e.args), default=0)


def extract_growth_atoms(expr, variables):
    variables = tuple(variables)
    out = []
    for e in sp.preorder_traversal(sp.sympify(expr)):
        if not (e.free_symbols & set(variables)):
            continue
        kind = None
        if e.func is sp.exp:
            kind = "exp"
        elif e.func is sp.log:
            kind = "log"
        elif e.is_Pow and e.exp.is_number:
            kind = "power"
        elif e in variables:
            kind = "variable"
        if kind is not None:
            atom = GrowthAtom(
                e, kind, _growth_depth(e), tuple(v for v in variables if e.has(v))
            )
            if atom not in out:
                out.append(atom)
    return tuple(out)


def _pair_regimes(a, b, variables):
    """Candidate comparability cells. Conditions are explicit and exhaustive."""
    A = sp.Abs(a.expression)
    B = sp.Abs(b.expression)
    # Ratio limits encode the three Hardy-style comparability regimes without
    # pretending that their limits have already been proved.
    ratio = sp.simplify(A / B)
    return (
        (sp.Eq(sp.Limit(ratio, variables[0], 0, dir="+"), 0), "a_smaller"),
        (
            sp.And(
                sp.Gt(sp.Limit(ratio, variables[0], 0, dir="+"), 0),
                sp.Lt(sp.Limit(ratio, variables[0], 0, dir="+"), sp.oo),
            ),
            "comparable",
        ),
        (sp.Eq(sp.Limit(ratio, variables[0], 0, dir="+"), sp.oo), "a_larger"),
    )


def automatic_exp_log_valuation_fan(expr, variables, *, domain=sp.S.true):
    """Extract competing exp-log scales and construct exhaustive candidate cells.

    Cells whose comparison cannot be certified are retained with UNKNOWN
    coverage rather than discarded.
    """
    variables = tuple(variables)
    atoms = extract_growth_atoms(expr, variables)
    if not atoms:
        return ExpLogValuationFan(
            (), (), CoverageCertificate.unknown("exp_log_fan", "no growth atoms")
        )
    # Polynomial powers induce Newton-like candidate weights. Cross-variable
    # exp/log atoms additionally create explicit relative-growth cells.
    relations = []
    for a in atoms:
        for b in atoms:
            if a is b or not a.variables or not b.variables:
                continue
            if a.variables[0] != b.variables[0]:
                x = b.variables[0]
                y = a.variables[0]
                # log-magnitude slope is the common coordinate for powers and
                # exp/log scales; alpha remains symbolic until a cell specializes it.
                alpha = sp.Symbol(
                    f"_alpha_{variables.index(y)}_{variables.index(x)}", positive=True
                )
                rel = RelativeGrowthRelation(
                    y,
                    x,
                    alpha,
                    "generated_log_ratio",
                    sp.Eq(sp.log(sp.Abs(y)) / sp.log(sp.Abs(x)), alpha),
                )
                if rel not in relations:
                    relations.append(rel)
    cells = []
    # Baseline cell always exists; it is complete only when no cross-variable
    # comparability is needed.
    if not relations:
        cones, _cov = relative_growth_valuation_fan(variables, sp.S.true)
        cells.append(
            GrowthCell(
                (),
                atoms,
                cones,
                sp.sympify(domain),
                CoverageCertificate.complete(
                    "exp_log_single_scale",
                    "all extracted atoms use one independent scale",
                    (domain,),
                ),
            )
        )
    else:
        # Generate one symbolic cone per relation; the union is a candidate fan.
        for rel in relations:
            ws = [sp.S.One] * len(variables)
            ws[variables.index(rel.dependent)] = rel.exponent
            cone = GrowthValuationRegion(
                (rel,),
                None,
                tuple(ws),
                (sp.S.Zero,) * len(ws),
                rel.condition,
                CoverageCertificate.complete(
                    "generated_growth_relation",
                    "symbolic relative-growth cone",
                    (tuple(ws),),
                ),
            )
            cells.append(
                GrowthCell(
                    (rel.condition,),
                    atoms,
                    (cone,),
                    sp.And(domain, rel.condition),
                    CoverageCertificate.complete(
                        "generated_growth_cell",
                        "explicit symbolic comparability cell",
                        (rel.condition,),
                    ),
                )
            )
    coverage = CoverageCertificate.unknown(
        "exp_log_candidate_fan",
        "generated symbolic log-ratio cells are candidates; exhaustiveness requires an exact comparability partition",
    )
    return ExpLogValuationFan(atoms, tuple(cells), coverage)


def growth_cell_blowup_atlases(fan, variables, target=None):
    """Feed generated valuation cells through the common weighted blow-up API."""
    from .blowup_geometry import weighted_spherical_atlas

    target = tuple(sp.S.Zero for _ in variables) if target is None else tuple(target)
    out = []
    for cell in fan.cells:
        for cone in cell.valuation_cones:
            # Only rational positive weights define algebraic weighted blow-ups.
            if all(w.is_Rational and w > 0 for w in cone.weights):
                out.append(
                    (
                        cell,
                        weighted_spherical_atlas(
                            tuple(variables), target, tuple(cone.weights)
                        ),
                    )
                )
    return tuple(out)


class GrowthScaleComparison(str, Enum):
    LESS = "less"
    EQUIVALENT = "equivalent"
    GREATER = "greater"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GrowthScale:
    """Normalized recursive positive-magnitude scale tree.

    ``kind`` is semantic rather than syntactic: constant, variable, power,
    product, exp, log, compose, or atom.  Products are flattened and positive
    constants are combined, so comparison rules see a stable tree.
    """

    kind: str
    args: tuple[GrowthScale, ...] = ()
    exponent: sp.Expr = sp.S.One
    constant: sp.Expr = sp.S.One
    expression: sp.Expr = sp.S.One
    variable: sp.Symbol | None = None
    target: sp.Expr = sp.S.Zero

    @property
    def depth(self):
        if self.kind in {"exp", "log", "compose"}:
            return 1 + max((a.depth for a in self.args), default=-1)
        return max((a.depth for a in self.args), default=0)


@dataclass(frozen=True)
class GrowthProofStep:
    rule: str
    statement: str
    premises: tuple[GrowthProofStep, ...] = ()


@dataclass(frozen=True)
class GrowthComparisonProof:
    left: GrowthScale
    right: GrowthScale
    relation: GrowthScaleComparison
    certified: bool
    proof: GrowthProofStep | None = None
    assumptions: sp.Expr = sp.S.true
    provider: str = "structural_exp_log_comparison"


def _ask(predicate, assumptions):
    """Three-valued assumption query; never turns an unknown into a guess."""
    predicate = sp.sympify(predicate)
    assumptions = sp.sympify(assumptions)
    direct = bounded_ask(predicate, assumptions)
    if direct is not None:
        return direct
    # SymPy's predicate engine misses some elementary relational implications.
    condition = getattr(predicate, "function", None)
    if condition is not None:
        refined = bounded_refine(condition, assumptions)
        if refined is sp.S.true:
            return True
        if refined is sp.S.false:
            return False
    return None


def _positive(e, assumptions):
    e = sp.sympify(e)
    if e.is_positive is True:
        return True
    if e.is_positive is False:
        return False
    return _ask(sp.Q.positive(e), assumptions)


def _negative(e, assumptions):
    e = sp.sympify(e)
    if e.is_negative is True:
        return True
    if e.is_negative is False:
        return False
    return _ask(sp.Q.negative(e), assumptions)


def _real(e, assumptions):
    e = sp.sympify(e)
    if e.is_real is not None:
        return e.is_real
    return _ask(sp.Q.real(e), assumptions)


def _positive_constant(e, x, assumptions=sp.S.true):
    return not e.has(x) and _positive(e, assumptions) is True


def _scale_key(s):
    return (
        s.kind,
        sp.srepr(s.expression),
        sp.srepr(s.exponent),
        tuple(_scale_key(a) for a in s.args),
    )


def growth_scale(expr, variable, target=0, assumptions=sp.S.true):
    """Build a normalized recursive scale tree.

    Positive constants are separated, nested products flattened, powers retain
    symbolic exponents, and exp/log composition remains explicit.
    """
    e = bounded_refine(sp.sympify(expr), sp.sympify(assumptions))
    x = sp.sympify(variable)
    target = sp.sympify(target)
    if _positive_constant(e, x, assumptions):
        return GrowthScale("constant", (), sp.S.One, e, e, x, target)
    if e == x or e == sp.Abs(x):
        return GrowthScale("variable", (), sp.S.One, sp.S.One, e, x, target)
    if e.func is sp.exp:
        return GrowthScale(
            "exp",
            (growth_scale(e.args[0], x, target, assumptions),),
            expression=e,
            variable=x,
            target=target,
        )
    if e.func is sp.log:
        return GrowthScale(
            "log",
            (growth_scale(e.args[0], x, target, assumptions),),
            expression=e,
            variable=x,
            target=target,
        )
    if e.is_Pow:
        return GrowthScale(
            "power",
            (growth_scale(e.base, x, target, assumptions),),
            sp.sympify(e.exp),
            expression=e,
            variable=x,
            target=target,
        )
    if e.is_Mul:
        constant = sp.S.One
        children = []
        for factor in e.args:
            if _positive_constant(factor, x, assumptions):
                constant *= factor
                continue
            child = growth_scale(factor, x, target, assumptions)
            if child.kind == "product":
                constant *= child.constant
                children.extend(child.args)
            elif child.kind == "constant":
                constant *= child.constant
            else:
                children.append(child)
        children = tuple(sorted(children, key=_scale_key))
        if not children:
            return GrowthScale("constant", (), constant, constant, e, x, target)
        return GrowthScale(
            "product",
            children,
            constant=sp.simplify(constant),
            expression=e,
            variable=x,
            target=target,
        )
    if len(e.args) == 1 and e.has(x):
        return GrowthScale(
            "compose",
            (growth_scale(e.args[0], x, target, assumptions),),
            expression=e,
            variable=x,
            target=target,
        )
    return GrowthScale("atom", (), expression=e, variable=x, target=target)


def _invert(r):
    return {
        GrowthScaleComparison.LESS: GrowthScaleComparison.GREATER,
        GrowthScaleComparison.GREATER: GrowthScaleComparison.LESS,
    }.get(r, r)


def _ratio_fallback(A, B, assumptions):
    try:
        ratio = bounded_refine(
            sp.simplify(sp.Abs(A.expression) / sp.Abs(B.expression)), assumptions
        )
        lim = bounded_limit(
            ratio, A.variable, A.target, direction="+", allow_general=True
        )
    except (
        ValueError,
        TypeError,
        NotImplementedError,
        RecursionError,
        sp.PolynomialError,
    ):
        return GrowthScaleComparison.UNKNOWN, None
    if lim is None:
        return GrowthScaleComparison.UNKNOWN, None
    if lim == 0:
        return GrowthScaleComparison.LESS, lim
    if lim in (sp.oo, -sp.oo, sp.zoo):
        return GrowthScaleComparison.GREATER, lim
    if lim.is_finite is True and lim.is_zero is False:
        return GrowthScaleComparison.EQUIVALENT, lim
    return GrowthScaleComparison.UNKNOWN, lim


def _power_of_variable(s):
    if s.kind == "variable":
        return sp.S.One
    if s.kind == "power" and s.args[0].kind == "variable":
        return s.exponent
    return None


def _inverse_log_power(s):
    # |log x|**(-M) may normalize through Abs/log/power in several equivalent ways.
    e = sp.sympify(s.expression)
    x = s.variable
    w = sp.Wild("w", exclude=[x])
    for base in (sp.Abs(sp.log(x)), -sp.log(x), sp.log(1 / x)):
        m = e.match(base**w)
        if m and w in m:
            return -sp.sympify(m[w])
    return None


def _flat_exp_power(s):
    """Return q for exp(-1/x**q), otherwise None."""
    if s.kind != "exp":
        return None
    x = s.variable
    q = sp.Wild("q", exclude=[x])
    m = sp.sympify(s.expression).match(sp.exp(-1 / x**q))
    return sp.sympify(m[q]) if m and q in m else None


def _nested_flat_depth(s):
    """Count exp(-...) layers above a positive divergent inner scale."""
    e = sp.sympify(s.expression)
    depth = 0
    while e.func is sp.exp:
        inner = sp.expand_power_base(e.args[0], force=False)
        if not inner.could_extract_minus_sign():
            break
        depth += 1
        e = -inner
    return depth if depth else None


def _structural_compare(A, B, assumptions):
    if sp.simplify(A.expression - B.expression) == 0:
        return GrowthScaleComparison.EQUIVALENT, GrowthProofStep(
            "identity", "identical normalized scales"
        )
    if A.kind == "constant" and B.kind == "constant":
        return GrowthScaleComparison.EQUIVALENT, GrowthProofStep(
            "positive_constants", "positive constants are scale-equivalent"
        )

    # Symbolic powers of the same vanishing base: x**N << x**M iff N>M.
    ap, bp = _power_of_variable(A), _power_of_variable(B)
    if ap is not None and bp is not None:
        d = sp.simplify(ap - bp)
        if _positive(d, assumptions) is True:
            return GrowthScaleComparison.LESS, GrowthProofStep(
                "symbolic_power_order", f"{ap} > {bp} under assumptions"
            )
        if _negative(d, assumptions) is True:
            return GrowthScaleComparison.GREATER, GrowthProofStep(
                "symbolic_power_order", f"{ap} < {bp} under assumptions"
            )
        if sp.simplify(d) == 0:
            return GrowthScaleComparison.EQUIVALENT, GrowthProofStep(
                "symbolic_power_order", "equal exponents"
            )

    # Common real exponent preserves/reverses an already certified base order.
    if A.kind == B.kind == "power" and sp.simplify(A.exponent - B.exponent) == 0:
        exponent = A.exponent
        if _real(exponent, assumptions) is True:
            r, p = _structural_compare(A.args[0], B.args[0], assumptions)
            if r is not GrowthScaleComparison.UNKNOWN:
                if _negative(exponent, assumptions) is True:
                    r = _invert(r)
                elif _positive(exponent, assumptions) is not True:
                    return GrowthScaleComparison.UNKNOWN, None
                return r, GrowthProofStep(
                    "power_monotonicity",
                    f"common exponent {exponent}",
                    (p,) if p else (),
                )

    # exp(-1/x**q) is flat relative to every positive power x**N for q,N>0.
    aq, bq = _flat_exp_power(A), _flat_exp_power(B)
    if (
        aq is not None
        and bp is not None
        and _positive(aq, assumptions) is True
        and _positive(bp, assumptions) is True
    ):
        return GrowthScaleComparison.LESS, GrowthProofStep(
            "flat_exponential_vs_power",
            "positive flat exponential order dominates logarithmic power order",
        )
    if (
        bq is not None
        and ap is not None
        and _positive(bq, assumptions) is True
        and _positive(ap, assumptions) is True
    ):
        return GrowthScaleComparison.GREATER, GrowthProofStep(
            "flat_exponential_vs_power",
            "positive flat exponential order dominates logarithmic power order",
        )

    # Every positive power tends to zero faster than an inverse positive log power.
    al, bl = _inverse_log_power(A), _inverse_log_power(B)
    if (
        ap is not None
        and bl is not None
        and _positive(ap, assumptions) is True
        and _positive(bl, assumptions) is True
    ):
        return GrowthScaleComparison.LESS, GrowthProofStep(
            "power_vs_log", "positive power is smaller than inverse positive log power"
        )
    if (
        bp is not None
        and al is not None
        and _positive(bp, assumptions) is True
        and _positive(al, assumptions) is True
    ):
        return GrowthScaleComparison.GREATER, GrowthProofStep(
            "power_vs_log", "positive power is smaller than inverse positive log power"
        )

    # Inverse log powers: larger positive exponent is the smaller scale.
    if al is not None and bl is not None:
        d = sp.simplify(al - bl)
        if _positive(d, assumptions) is True:
            return GrowthScaleComparison.LESS, GrowthProofStep(
                "symbolic_log_power_order", f"{al} > {bl}"
            )
        if _negative(d, assumptions) is True:
            return GrowthScaleComparison.GREATER, GrowthProofStep(
                "symbolic_log_power_order", f"{al} < {bl}"
            )
        if d == 0:
            return GrowthScaleComparison.EQUIVALENT, GrowthProofStep(
                "symbolic_log_power_order", "equal inverse-log exponents"
            )

    # Structural nested flat exponentials: an additional certified negative-exp
    # layer is smaller. This is narrow and one-sided.
    ad, bd = _nested_flat_depth(A), _nested_flat_depth(B)
    if ad and bd and ad != bd:
        if ad > bd:
            return GrowthScaleComparison.LESS, GrowthProofStep(
                "nested_flat_exponential",
                "additional negative exponential layer is smaller",
            )
        return GrowthScaleComparison.GREATER, GrowthProofStep(
            "nested_flat_exponential",
            "additional negative exponential layer is smaller",
        )

    # Products can be compared after exact cancellation.  This lifts known
    # pairwise scale rules through products/quotients without sampling paths.
    ratio = sp.cancel(A.expression / B.expression)
    if ratio != A.expression / B.expression or ratio.is_Mul or ratio.is_Pow:
        # Recognize a flat exponential multiplied by any finite power/log power:
        # exp(-1/x**q) * x**(-N) * log(1/x)**M -> 0 for positive concrete q.
        x = A.variable
        for eleft, direction in (
            (ratio, GrowthScaleComparison.LESS),
            (1 / ratio, GrowthScaleComparison.GREATER),
        ):
            factors = sp.Mul.make_args(sp.factor_terms(eleft))
            flats = [
                f
                for f in factors
                if f.func is sp.exp
                and _flat_exp_power(growth_scale(f, x, A.target, assumptions))
                is not None
            ]
            if flats:
                q = _flat_exp_power(growth_scale(flats[0], x, A.target, assumptions))
                if _positive(q, assumptions) is True:
                    rest = sp.simplify(eleft / flats[0])
                    # Restrict the residual to algebraic powers and finite log powers.
                    allowed = True
                    for f in sp.Mul.make_args(rest):
                        if not (
                            f.is_number
                            or _power_of_variable(
                                growth_scale(f, x, A.target, assumptions)
                            )
                            is not None
                            or _inverse_log_power(
                                growth_scale(f, x, A.target, assumptions)
                            )
                            is not None
                            or (
                                f.is_Pow
                                and f.base in (sp.log(1 / x), sp.Abs(sp.log(x)))
                            )
                        ):
                            allowed = False
                            break
                    if allowed:
                        return direction, GrowthProofStep(
                            "flat_exp_product_dominance",
                            "flat exponential dominates finite algebraic/logarithmic product factors",
                        )

    # exp(-1/x**q) after an algebraic positive composition remains flat.
    x = A.variable

    def composed_flat(s):
        e = sp.sympify(s.expression)
        q = sp.Wild("q", exclude=[x])
        m = e.match(sp.exp(-1 / (x**q)))
        return sp.sympify(m[q]) if m and q in m else None

    aq2, bq2 = composed_flat(A), composed_flat(B)
    if (
        aq2 is not None
        and bp is not None
        and _positive(aq2, assumptions) is True
        and _positive(bp, assumptions) is True
    ):
        return GrowthScaleComparison.LESS, GrowthProofStep(
            "composed_flat_exponential",
            "positive algebraic composition preserves flatness",
        )
    if (
        bq2 is not None
        and ap is not None
        and _positive(bq2, assumptions) is True
        and _positive(ap, assumptions) is True
    ):
        return GrowthScaleComparison.GREATER, GrowthProofStep(
            "composed_flat_exponential",
            "positive algebraic composition preserves flatness",
        )

    # Multiplication by positive constants does not alter scale class.
    if A.kind == "product" and A.constant != 1:
        stripped = GrowthScale(
            "product",
            A.args,
            expression=sp.Mul(*(a.expression for a in A.args)),
            variable=A.variable,
            target=A.target,
        )
        r, p = _structural_compare(stripped, B, assumptions)
        if r is not GrowthScaleComparison.UNKNOWN:
            return r, GrowthProofStep(
                "positive_constant_factor",
                "discard positive constant factor",
                (p,) if p else (),
            )
    if B.kind == "product" and B.constant != 1:
        stripped = GrowthScale(
            "product",
            B.args,
            expression=sp.Mul(*(a.expression for a in B.args)),
            variable=B.variable,
            target=B.target,
        )
        r, p = _structural_compare(A, stripped, assumptions)
        if r is not GrowthScaleComparison.UNKNOWN:
            return r, GrowthProofStep(
                "positive_constant_factor",
                "discard positive constant factor",
                (p,) if p else (),
            )

    r, lim = _ratio_fallback(A, B, assumptions)
    if r is not GrowthScaleComparison.UNKNOWN:
        return r, GrowthProofStep(
            "certified_ratio", f"absolute ratio has exact limit {lim}"
        )
    return GrowthScaleComparison.UNKNOWN, None


def compare_growth_scales(a, b, variable=None, target=0, assumptions=sp.S.true):
    A = (
        a
        if isinstance(a, GrowthScale)
        else growth_scale(a, variable, target, assumptions)
    )
    B = (
        b
        if isinstance(b, GrowthScale)
        else growth_scale(b, variable or A.variable, target, assumptions)
    )
    if A.variable != B.variable:
        return GrowthScaleComparison.UNKNOWN
    return _structural_compare(A, B, sp.sympify(assumptions))[0]


def prove_growth_comparison(a, b, variable, target=0, assumptions=sp.S.true):
    A = growth_scale(a, variable, target, assumptions)
    B = growth_scale(b, variable, target, assumptions)
    rel, step = _structural_compare(A, B, sp.sympify(assumptions))
    return GrowthComparisonProof(
        A,
        B,
        rel,
        rel is not GrowthScaleComparison.UNKNOWN,
        step,
        sp.sympify(assumptions),
    )


def certify_exp_log_fan(
    expr, variables, *, domain=sp.S.true, assumptions=sp.S.true, max_cells=243
):
    """Certify a finite exp/log comparability fan when its cells are exhaustive."""
    variables = tuple(variables)
    domain = sp.sympify(domain)
    assumptions = sp.sympify(assumptions)
    fan = automatic_exp_log_valuation_fan(expr, variables, domain=domain)
    proofs = []
    unknown = []
    for i, a in enumerate(fan.atoms):
        for b in fan.atoms[i + 1 :]:
            if len(a.variables) == len(b.variables) == 1 and a.variables == b.variables:
                pr = prove_growth_comparison(
                    a.expression, b.expression, a.variables[0], assumptions=assumptions
                )
                proofs.append(pr)
                if not pr.certified:
                    unknown.append((a.expression, b.expression))
    if unknown:
        return ExpLogValuationFan(
            fan.atoms,
            fan.cells,
            CoverageCertificate.partial(
                "exp_log_comparability",
                "some same-scale comparisons remain unknown",
                tuple(
                    (p.left.expression, p.right.expression, p.relation.value)
                    for p in proofs
                    if p.certified
                ),
                tuple(unknown),
            ),
        ), tuple(proofs)
    # Cross-variable generated log-ratio parameters are positive and every one
    # lies in exactly one of alpha<1, alpha=1, alpha>1.  Taking the Cartesian
    # product therefore gives an exact finite comparability partition.
    rels = []
    for c in fan.cells:
        for cone in c.valuation_cones:
            for r in cone.relations:
                if r.kind == "generated_log_ratio" and r not in rels:
                    rels.append(r)
    if not rels:
        return ExpLogValuationFan(
            fan.atoms,
            fan.cells,
            CoverageCertificate.complete(
                "exp_log_comparability",
                "all extracted scales use one approach variable and every pairwise comparison is certified",
                (domain,),
            ),
        ), tuple(proofs)
    if 3 ** len(rels) > max_cells:
        return ExpLogValuationFan(
            fan.atoms,
            fan.cells,
            CoverageCertificate.unknown(
                "exp_log_comparability",
                "cross-variable comparability arrangement exceeds cell budget",
                tuple(r.exponent for r in rels),
            ),
        ), tuple(proofs)
    cells = [(sp.And(domain, assumptions), ())]
    for r in rels:
        nxt = []
        a = r.exponent
        for cond, rr in cells:
            for relation in (sp.Lt(a, 1), sp.Eq(a, 1), sp.Gt(a, 1)):
                q = sp.And(cond, r.condition, relation)
                if _feasible_cell(sp.And(cond, relation)):
                    nxt.append((q, rr + (r,)))
        cells = nxt
    gcells = []
    for cond, rr in cells:
        # Use all generated relations together; equalities in cond specialize
        # the symbolic weights before downstream Newton geometry.
        ws = [sp.S.One] * len(variables)
        for r in rr:
            ws[variables.index(r.dependent)] = r.exponent
        _, sws, _subs = specialize_parameter_cell(sp.S.Zero, tuple(ws), cond)
        cone = GrowthValuationRegion(
            tuple(rr),
            None,
            sws,
            (sp.S.Zero,) * len(sws),
            cond,
            CoverageCertificate.complete(
                "exp_log_comparability_cell", "exact log-ratio cell", (cond,)
            ),
        )
        gcells.append(
            GrowthCell(
                tuple(r.condition for r in rr),
                fan.atoms,
                (cone,),
                cond,
                CoverageCertificate.complete(
                    "exp_log_comparability_cell", "exact relative-growth cell", (cond,)
                ),
            )
        )
    cov = CoverageCertificate.complete(
        "exp_log_comparability_partition",
        "positive log-ratio parameters are exhaustively partitioned by <,=,> against every critical comparison boundary",
        tuple(c.condition for c in gcells),
    )
    return ExpLogValuationFan(fan.atoms, tuple(gcells), cov), tuple(proofs)


@dataclass(frozen=True)
class SymbolicWeightCell:
    condition: sp.Expr
    weights: tuple[sp.Expr, ...]
    certified: bool
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()


def _feasible_cell(condition):
    try:
        ans = sp.satisfiable(sp.sympify(condition), use_lra_theory=True)
        return ans is not False
    except (TypeError, ValueError, NotImplementedError):
        try:
            return sp.satisfiable(sp.sympify(condition)) is not False
        except (TypeError, ValueError, NotImplementedError):
            return True


def certify_symbolic_weights(weights, *, assumptions=sp.S.true, max_cells=243):
    """Construct an exhaustive real sign/order partition for Newton weights.

    The partition contains the positivity boundaries and pairwise order
    boundaries needed by Newton-face selection.  Logical trichotomy is exact;
    infeasible cells are removed only when satisfiability proves them empty.
    """
    assumptions = sp.sympify(assumptions)
    weights = tuple(map(sp.sympify, weights))
    comparisons = []
    for w in weights:
        if bounded_ask(sp.Q.real(w), assumptions) is not True and w.is_real is not True:
            return (), CoverageCertificate.unknown(
                "symbolic_weight_partition",
                "a Newton weight is not certified real",
                (w,),
            )
        comparisons.append(w)
    for i, a in enumerate(weights):
        for b in weights[i + 1 :]:
            comparisons.append(sp.simplify(a - b))
    # Remove constants and duplicate boundaries up to sign.
    boundaries = []
    for d in comparisons:
        if d.is_number:
            continue
        if any(sp.simplify(d - e) == 0 or sp.simplify(d + e) == 0 for e in boundaries):
            continue
        boundaries.append(d)
    if 3 ** len(boundaries) > max_cells:
        return (), CoverageCertificate.unknown(
            "symbolic_weight_partition",
            "exact sign arrangement exceeds cell budget",
            tuple(boundaries),
        )
    cells = [assumptions]
    for d in boundaries:
        nxt = []
        for c in cells:
            for rel in (sp.Lt(d, 0), sp.Eq(d, 0), sp.Gt(d, 0)):
                q = sp.And(c, rel)
                if _feasible_cell(q):
                    nxt.append(q)
        cells = nxt
    out = []
    for c in cells:
        # Newton weights themselves must be positive on an admissible cell.
        positive = all(
            bounded_ask(sp.Q.positive(w), c) is True
            or _feasible_cell(sp.And(c, sp.Le(w, 0))) is False
            for w in weights
        )
        if not positive:
            continue
        subs = (
            _equality_substitution_map(c)
            if "_equality_substitution_map" in globals()
            else {}
        )
        ws = tuple(sp.simplify(w.subs(subs)) for w in weights)
        out.append(
            SymbolicWeightCell(
                c,
                ws,
                True,
                tuple(sorted(subs.items(), key=lambda kv: sp.default_sort_key(kv[0]))),
            )
        )
    if not out:
        return (), CoverageCertificate.unknown(
            "symbolic_weight_partition",
            "no feasible positive-weight cell was certified",
        )
    return tuple(out), CoverageCertificate.complete(
        "symbolic_weight_partition",
        "real trichotomy of every zero/order boundary exhausts the admissible positive-weight parameter domain",
        tuple(c.condition for c in out),
    )


def _equality_substitution_map(condition):
    eqs = [
        q for q in sp.And.make_args(sp.sympify(condition)) if isinstance(q, sp.Equality)
    ]
    if not eqs:
        return {}
    symbols = sorted(
        set().union(*(e.free_symbols for e in eqs)), key=sp.default_sort_key
    )
    try:
        sols = bounded_solve_system(
            [e.lhs - e.rhs for e in eqs], symbols, allow_general=True
        )
    except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
        return {}
    if len(sols) != 1:
        return {}
    raw = sols[0]
    out = {}
    for k, v in raw.items():
        v = sp.simplify(v)
        for _ in range(len(raw) + 1):
            nv = sp.simplify(v.xreplace(raw))
            if nv == v:
                break
            v = nv
        if k != v:
            out[k] = v
    return out


def specialize_parameter_cell(expression, weights, condition):
    subs = _equality_substitution_map(condition)
    return (
        sp.simplify(sp.sympify(expression).subs(subs)),
        tuple(sp.simplify(sp.sympify(w).subs(subs)) for w in weights),
        subs,
    )


__all__ = [
    "ExpLogValuationFan",
    "GrowthAtom",
    "GrowthCell",
    "GrowthComparisonProof",
    "GrowthProofStep",
    "GrowthScale",
    "GrowthScaleComparison",
    "GrowthValuationRegion",
    "RelativeGrowthRelation",
    "automatic_exp_log_valuation_fan",
    "certify_exp_log_fan",
    "compare_growth_scales",
    "exp_log_order",
    "extract_growth_atoms",
    "growth_cell_blowup_atlases",
    "growth_scale",
    "prove_growth_comparison",
    "relative_growth_valuation_fan",
    "specialize_parameter_cell",
]
