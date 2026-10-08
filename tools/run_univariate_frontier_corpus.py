import json
from collections import Counter
from pathlib import Path

import sympy as sp

from asymptotic.algebraic_series import implicit_map_series
from asymptotic.generalized_series import generalized_series
from asymptotic.oscillatory_normal_form import oscillatory_cluster_interval
from asymptotic.special_function_infinity import special_function_infinity

ROOT = Path(__file__).parents[1]
F = {
    "sin": sp.sin,
    "cos": sp.cos,
    "zeta": sp.zeta,
    "primepi": sp.primepi,
    "polygamma": sp.polygamma,
    "airyaiprime": sp.airyaiprime,
    "airybi": sp.airybi,
    "airybiprime": sp.airybiprime,
    "uppergamma": sp.uppergamma,
    "gamma": sp.gamma,
    "log": sp.log,
    "exp": sp.exp,
    "cot": sp.cot,
    "bessely": sp.bessely,
    "sqrt": sp.sqrt,
}
cases = json.loads((ROOT / "tests/data/univariate_frontier_cases.json").read_text())
rows = []
x = sp.Symbol("x", positive=True)
for c in cases:
    try:
        if c["category"] == "implicit_map_system":
            y1, y2 = sp.symbols("y1 y2", real=True)
            pair = sp.sympify(c["expression"], locals={"x": x, "y1": y1, "y2": y2})
            ok = (
                implicit_map_series(
                    pair, (y1, y2), x, point=0, dependent_limit=(0, 0), order=5
                )
                is not None
            )
        else:
            e = sp.sympify(c["expression"], locals={**F, "x": x})
            if c["category"] == "oscillatory_multifrequency":
                ok = oscillatory_cluster_interval(e, x) is not None
            elif c["category"] == "special_function_infinity_frontier":
                ok = special_function_infinity(e, x) is not None
            else:
                ok = generalized_series(e, x, point=0, terms=5) is not None
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        ok = False
    rows.append(
        {
            "id": c["id"],
            "category": c["category"],
            "outcome": "PASS" if ok else "MISSING",
        }
    )
(ROOT / "lim-txt-behavioral-frontier-report.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)

print(Counter(r["outcome"] for r in rows))
for cat in sorted({r["category"] for r in rows}):
    print(cat, Counter(r["outcome"] for r in rows if r["category"] == cat))
