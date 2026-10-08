"""Bounded executable runner for generated metamorphic cases."""

import argparse
import json
import re
import signal
from pathlib import Path

import sympy as sp

from asymptotic.limits import limit


def alarm(*_):
    raise TimeoutError


signal.signal(signal.SIGALRM, alarm)
FUN = {
    "Tuple": sp.Tuple,
    "I": sp.I,
    "pi": sp.pi,
    "oo": sp.oo,
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
    "floor": sp.floor,
    "Derivative": sp.Derivative,
    "Rational": sp.Rational,
    "f": sp.Function("f"),
}


def parse(s, loc):
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", str(s)))
    extra = {n: sp.Symbol(n, real=True) for n in names if n not in FUN and n not in loc}
    return sp.sympify(s, locals={**FUN, **extra, **loc})


ap = argparse.ArgumentParser()
ap.add_argument("--timeout", type=int, default=1)
ap.add_argument("--output", required=True)
a = ap.parse_args()
root = Path(__file__).parents[1]
rows = json.loads(
    (root / "tests/data/multivariate_limit_metamorphic_cases.json").read_text()
)
out = []
for c in rows:
    loc = {n: sp.Symbol(n, real=True) for n in c["variables"]}
    try:
        e = parse(c["expression"], loc)
        vs = tuple(loc[n] for n in c["variables"])
        tar = tuple(parse(x, loc) for x in c["target"])
        ds = c.get("domain", "True")
        dom = (
            sp.And(*(parse(q.strip(), loc) for q in ds.split("&")))
            if "&" in ds
            else parse(ds, loc)
        )
        signal.alarm(a.timeout)
        r = limit(e, vs, tar, domain=dom, return_result=True)
        signal.alarm(0)
        actual = r.status.name
        if c["expected_kind"] == "dne":
            verdict = (
                "PASS"
                if actual == "DOES_NOT_EXIST"
                else ("UNKNOWN" if actual == "UNKNOWN" else "WRONG")
            )
        elif c.get("expected") is None:
            verdict = "UNKNOWN" if actual == "UNKNOWN" else "PASS"
        else:
            ex = parse(c["expected"], loc)
            if actual == "UNKNOWN":
                verdict = "UNKNOWN"
            elif actual != "PROVED":
                verdict = "WRONG"
            else:
                try:
                    verdict = (
                        "PASS"
                        if bool(r.value == ex or sp.simplify(r.value - ex) == 0)
                        else "WRONG"
                    )
                except (TypeError, ValueError, NotImplementedError):
                    verdict = "WRONG"
    except TimeoutError:
        verdict = "TIMEOUT"
    except (
        TypeError,
        ValueError,
        NotImplementedError,
        RecursionError,
        sp.PolynomialError,
        ZeroDivisionError,
    ):
        verdict = "ERROR"
    finally:
        signal.alarm(0)
    out.append({"id": c["id"], "verdict": verdict})
Path(a.output).write_text(json.dumps(out, indent=2) + "\n")
print(__import__("collections").Counter(x["verdict"] for x in out))
