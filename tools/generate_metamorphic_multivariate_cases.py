"""Generate deterministic metamorphic tests from the curated 320-case corpus."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).parents[1]
SRC = ROOT / "tests/data/multivariate_limit_comprehensive_cases.json"
OUT = ROOT / "tests/data/multivariate_limit_metamorphic_cases.json"


def symbols(c):
    return {n: sp.Symbol(n, real=True) for n in c["variables"]}


FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "arg": sp.arg,
    "gamma": sp.gamma,
    "erf": sp.erf,
    "erfi": sp.erfi,
    "besselj": sp.besselj,
    "bessely": sp.bessely,
    "airyai": sp.airyai,
    "Piecewise": sp.Piecewise,
    "Eq": sp.Eq,
    "Ne": sp.Ne,
    "Mod": sp.Mod,
    "sign": sp.sign,
    "Tuple": sp.Tuple,
    "Rational": sp.Rational,
    "floor": sp.floor,
    "Derivative": sp.Derivative,
    "f": sp.Function("f"),
    "re": sp.re,
    "im": sp.im,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "I": sp.I,
    "pi": sp.pi,
    "oo": sp.oo,
    "True": sp.S.true,
    "False": sp.S.false,
}


def parse(s, loc):
    names = set(__import__("re").findall(r"\b[A-Za-z_]\w*\b", str(s)))
    extra = {
        n: sp.Symbol(n, real=True) for n in names if n not in FUNCS and n not in loc
    }
    return sp.sympify(s, locals={**FUNCS, **extra, **loc})


def emit(seed, kind, expr, variables, target, domain, expected_kind, expected=None):
    d = {
        "id": f"meta_{seed['id']}_{kind}",
        "seed_id": seed["id"],
        "transformation": kind,
        "capability": seed["capability"],
        "expression": str(expr),
        "variables": [str(v) for v in variables],
        "target": [str(v) for v in target],
        "domain": str(domain),
        "expected_kind": expected_kind,
        "oracle": "SAME_RESULT",
    }
    if expected is not None:
        d["expected"] = str(expected)
    return d


def generate(c):
    loc = symbols(c)
    vs = tuple(loc[n] for n in c["variables"])
    e = parse(c["expression"], loc)
    tar = tuple(parse(x, loc) for x in c["target"])
    domain_text = c.get("domain", "True")
    dom = (
        sp.And(*(parse(part.strip(), loc) for part in domain_text.split("&")))
        if "&" in domain_text
        else sp.sympify(parse(domain_text, loc))
    )
    try:
        expected = parse(c["expected"], loc) if c.get("expected") is not None else None
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        expected = None
    out = []
    # Opaque derivative/function-expression seeds use representation-preserving identities.
    if e.has(sp.Derivative):
        return [
            emit(c, f"identity_{i}", e, vs, tar, dom, c["expected_kind"], expected)
            for i in range(3)
        ]
    # 1. Variable permutation: reverse all coordinates consistently.
    # perform simultaneous permutation using temporary symbols then restore canonical names
    perm = tuple(reversed(vs))
    sub = {v: perm[i] for i, v in enumerate(vs)}
    out.append(
        emit(
            c,
            "permute",
            e.xreplace(sub),
            vs,
            tuple(reversed(tar)),
            dom.xreplace(sub),
            c["expected_kind"],
            expected,
        )
    )
    # 2. Positive rational coordinate rescaling around finite targets only.
    if all(t not in (sp.oo, -sp.oo) for t in tar):
        sub = {
            v: tar[i] + sp.Rational(i + 2, i + 1) * (v - tar[i])
            for i, v in enumerate(vs)
        }
        out.append(
            emit(
                c,
                "positive_scale",
                e.xreplace(sub),
                vs,
                tar,
                dom.xreplace(sub),
                c["expected_kind"],
                expected,
            )
        )
    # 3. Add a vanishing polynomial perturbation only for finite-value seeds.
    if (
        c["expected_kind"] == "value"
        and expected not in (sp.oo, -sp.oo)
        and all(t not in (sp.oo, -sp.oo) for t in tar)
    ):
        h = sum((v - tar[i]) ** 2 for i, v in enumerate(vs))
        out.append(
            emit(
                c,
                "vanishing_add",
                sp.Tuple(*(q + h for q in e)) if isinstance(e, sp.Tuple) else e + h,
                vs,
                tar,
                dom,
                "value",
                expected,
            )
        )
    # Fill to exactly 3 using algebraic identity rewrite, which preserves domain.
    while len(out) < 3:
        out.append(
            emit(
                c,
                f"identity_{len(out)}",
                (
                    e
                    if isinstance(e, sp.Tuple)
                    else sp.Add(e, sp.S.Zero, evaluate=False)
                ),
                vs,
                tar,
                dom,
                c["expected_kind"],
                expected,
            )
        )
    return out


cases = json.loads(SRC.read_text())
rows = []
for c in cases:
    try:
        rows.extend(generate(c))
    except Exception:
        print("FAILED", c["id"], c["expression"], c.get("domain"), c.get("expected"))
        raise
assert len(rows) == 960
assert len({r["id"] for r in rows}) == 960
OUT.write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} metamorphic cases to {OUT}")
