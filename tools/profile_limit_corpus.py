"""Profile proof routes for the multivariate reference corpus."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sympy as sp

from asymptotic.limit_planner import limit_metrics, reset_limit_metrics
from asymptotic.limits import limit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    data = (
        Path(__file__).parents[1] / "tests/data/multivariate_limit_reference_cases.json"
    )
    cases = json.loads(data.read_text())
    timings = []
    reset_limit_metrics()
    for case in cases[: args.max_cases]:
        symbols = {n: sp.Symbol(n, real=True) for n in case["variables"]}
        symbols.update(
            {n: sp.Symbol(n, real=True) for n in case.get("real_parameters", [])}
        )
        local_symbols = {**symbols, "oo": sp.oo, "zoo": sp.zoo}

        def parse(value, namespace=local_symbols):
            return sp.sympify(value, locals=namespace)

        expr = parse(case["expression"])
        variables = tuple(symbols[n] for n in case["variables"])
        target = tuple(parse(v) for v in case["target"])
        start = time.perf_counter()
        try:
            result = limit(
                expr,
                variables,
                target,
                domain=parse(case.get("domain", "True")),
                return_result=True,
            )
            status = result.status.name
        except (NotImplementedError, RecursionError):
            status = "UNSUPPORTED"
        elapsed = time.perf_counter() - start
        timings.append((elapsed, case["id"], status))
        print(
            json.dumps({"seconds": elapsed, "id": case["id"], "status": status}),
            flush=True,
        )
    timings.sort(reverse=True)
    print(
        json.dumps(
            {
                "slowest": [
                    {"seconds": t, "id": i, "status": s} for t, i, s in timings[:25]
                ],
                "routes": limit_metrics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
