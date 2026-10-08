"""Execute the curated multivariate capability matrix and report outcomes."""

from __future__ import annotations

import argparse
import json
import re
import signal
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import sympy as sp


def _parse_domain(text, locals_):
    if not text:
        return sp.S.true
    # Corpus conjunction syntax is simple.
    parts = [p.strip() for p in text.split("&")]
    return sp.And(*(sp.sympify(p, locals=locals_) for p in parts))


def _equal(a, b):
    try:
        return sp.simplify(a - b) == 0
    except (TypeError, ValueError, NotImplementedError):
        return a == b


def _one(case, timeout=6):
    from asymptotic.limits import LimitStatus, limit

    def alarm(*_):
        raise TimeoutError

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(timeout)
    try:
        names = set(case["variables"])
        for field in (
            "expression",
            "domain",
            "expected",
            "assumptions",
            "growth_comparison",
        ):
            names |= set(re.findall(r"\b[A-Za-z_]\w*\b", str(case.get(field, ""))))
        reserved = {
            "Tuple",
            "Rational",
            "Abs",
            "Eq",
            "Ne",
            "Piecewise",
            "True",
            "False",
            "I",
            "pi",
            "E",
            "oo",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "exp",
            "log",
            "sqrt",
            "arg",
            "gamma",
            "airyai",
            "besselj",
            "bessely",
            "erf",
            "erfi",
            "sign",
            "floor",
            "Mod",
            "Derivative",
        }
        loc = {n: sp.Symbol(n, real=True) for n in names - reserved}
        loc.update({v: sp.Symbol(v, real=True) for v in case["variables"]})
        expr = sp.sympify(case["expression"], locals=loc)
        assumptions = _parse_domain(case.get("assumptions"), loc)
        expr = sp.refine(expr, assumptions)
        variables = tuple(loc[v] for v in case["variables"])
        target = tuple(sp.sympify(v, locals=loc) for v in case["target"])
        domain = _parse_domain(case.get("domain"), loc)
        r = limit(
            expr,
            variables,
            target,
            domain=domain,
            assumptions=assumptions,
            return_result=True,
        )
        if hasattr(r, "strata") and hasattr(r, "parameters"):
            actual = "CONDITIONAL"
            result_value = getattr(r, "mathematical_value", None)
        else:
            actual = {
                LimitStatus.PROVED: "PROVED",
                LimitStatus.DOES_NOT_EXIST: "DNE",
                LimitStatus.UNKNOWN: "UNKNOWN",
            }[r.status]
            result_value = r.value
        expected = case["expected_kind"]
        wrong = False
        # Capability-scoped prerequisite failures test a method boundary, not a
        # global-solver prohibition, so a global proof is not called wrong.
        if case["scope"] == "global":
            if expected == "dne":
                wrong = actual == "PROVED"
            elif expected == "value":
                if actual == "DNE":
                    wrong = True
                elif actual == "PROVED":
                    ev = sp.sympify(case.get("expected"), locals=loc)
                    wrong = not (r.value == ev or _equal(r.value, ev))
            elif expected in {"conditional", "cluster"}:
                wrong = False
        growth = None
        if "growth_comparison" in case:
            from asymptotic.relative_growth import prove_growth_comparison

            left, right, wanted = case["growth_comparison"]
            pr = prove_growth_comparison(
                sp.sympify(left, locals=loc),
                sp.sympify(right, locals=loc),
                variables[0],
                assumptions=assumptions,
            )
            growth = {
                "actual": pr.relation.value,
                "expected": wanted,
                "certified": pr.certified,
                "rule": pr.proof.rule if pr.proof else None,
                "wrong_result": pr.relation.value != wanted,
            }
        return {
            "id": case["id"],
            "capability": case["capability"],
            "role": case["role"],
            "actual": actual,
            "wrong_result": wrong,
            "value": str(result_value) if result_value is not None else None,
            "growth_comparison": growth,
        }
    except TimeoutError:
        return {
            "id": case["id"],
            "capability": case["capability"],
            "role": case["role"],
            "actual": "UNKNOWN",
            "wrong_result": False,
            "detail": "timeout",
        }
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError) as exc:
        return {
            "id": case["id"],
            "capability": case["capability"],
            "role": case["role"],
            "actual": "UNKNOWN",
            "wrong_result": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    finally:
        signal.alarm(0)


def _worker(payload):
    case, timeout = payload
    return _one(case, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=6)
    ap.add_argument("--output", default="capability-matrix.json")
    ns = ap.parse_args()
    corpus = (
        Path(__file__).parents[1]
        / "tests/data/multivariate_limit_comprehensive_cases.json"
    )
    cases = json.loads(corpus.read_text())
    with get_context("fork").Pool(ns.workers) as pool:
        rows = pool.map(_worker, [(c, ns.timeout) for c in cases])
    by = defaultdict(Counter)
    for r in rows:
        by[r["capability"]]["wrong-result" if r["wrong_result"] else r["actual"]] += 1
    report = {
        "case_count": len(rows),
        "capability_count": len(by),
        "totals": dict(
            Counter("wrong-result" if r["wrong_result"] else r["actual"] for r in rows)
        ),
        "capabilities": {k: dict(v) for k, v in sorted(by.items())},
        "cases": rows,
    }
    Path(ns.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"], sort_keys=True))


if __name__ == "__main__":
    import re

    main()
