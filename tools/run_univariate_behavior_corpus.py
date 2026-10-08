"""Run neutral mathematical behavior cases derived from univariate limit categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from asymptotic.algebraic_series import algebraic_series
from asymptotic.limits import LimitStatus, limit
from asymptotic.oscillatory_normal_form import oscillatory_cluster_interval
from asymptotic.special_function_infinity import special_function_infinity

ROOT = Path(__file__).parents[1]
FUN = {
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "sqrt": sp.sqrt,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "besselj": sp.besselj,
    "bessely": sp.bessely,
    "airyai": sp.airyai,
    "zeta": sp.zeta,
    "primepi": sp.primepi,
    "pi": sp.pi,
    "oo": sp.oo,
    "Interval": sp.Interval,
}


def parse(s, loc):
    return sp.sympify(s, locals={**FUN, **loc})


def eq(a, b):
    try:
        return bool(a == b or sp.simplify(a - b) == 0)
    except (TypeError, ValueError, NotImplementedError):
        return False


def run(c):
    x = sp.Symbol(c["variable"], real=True)
    loc = {c["variable"]: x}
    e = parse(c["expression"], loc)
    cat = c["category"]
    try:
        if cat in ("algebraic_series", "puiseux_series"):
            y = sp.Symbol("y", real=True)
            e = parse(c["expression"], {**loc, "y": y})
            r = algebraic_series(e, y, x, point=0, dependent_limit=0, order=6)
            return "PASS" if r.expansion is not None else "MISSING"
        if cat == "oscillatory_interval":
            r = oscillatory_cluster_interval(e, x)
            ex = parse(c["expected"], loc)
            return "PASS" if r is not None and r.set == ex else "MISSING"
        if c["expected_kind"] == "leading":
            r = special_function_infinity(e, x)
            return "PASS" if r is not None else "MISSING"
        point = parse(c["point"], loc)
        r = limit(e, (x,), (point,), return_result=True)
        if r.status is LimitStatus.UNKNOWN:
            return "MISSING"
        if r.status is not LimitStatus.PROVED:
            return "WRONG"
        return "PASS" if eq(r.value, parse(c["expected"], loc)) else "WRONG"
    except (
        TypeError,
        ValueError,
        NotImplementedError,
        RecursionError,
        sp.PolynomialError,
    ):
        return "ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    cases = json.loads((ROOT / "tests/data/univariate_behavior_cases.json").read_text())
    rows = []
    for c in cases:
        rows.append({"id": c["id"], "category": c["category"], "outcome": run(c)})
    Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
    from collections import Counter

    print(Counter(r["outcome"] for r in rows))
    for cat in sorted({r["category"] for r in rows}):
        print(cat, Counter(r["outcome"] for r in rows if r["category"] == cat))


if __name__ == "__main__":
    main()
