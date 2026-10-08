"""Microbenchmark lazy GermAnalysis dispatch versus uncached certificate dispatch."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import sympy as sp

from asymptotic._multivariate_germ import GermAnalysis
from asymptotic.multivariate_certificates import (
    analytic_leading_ratio_certificate,
    divided_difference_certificate,
    growth_scale_limit,
    rational_power_order_certificate,
    reciprocal_pole_certificate,
    sertoz_rational_limit,
)

CERTS = (
    reciprocal_pole_certificate,
    divided_difference_certificate,
    growth_scale_limit,
    rational_power_order_certificate,
    analytic_leading_ratio_certificate,
    sertoz_rational_limit,
)


def dispatch(expr, vars, target, cached):
    ctx = GermAnalysis.create(expr, vars, target) if cached else None
    for fn in CERTS:
        kw = {"_context": ctx} if cached else {}
        c = fn(expr, vars, target, **kw)
        if c.certified:
            return c.method
    return None


def bench(expr, vars, target, cached, n=20):
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        dispatch(expr, vars, target, cached)
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)


def main():
    x, y = sp.symbols("x y", real=True)
    cases = {
        "first_hit": (1 / (x**2 + y**2), (x, y), (0, 0)),
        "middle_hit": ((x**2 + y**2) * sp.log(x**2 + y**2), (x, y), (0, 0)),
        "last_hit": ((x**8 + y**8) / (x**4 + y**6), (x, y), (0, 0)),
        "fallthrough": (sp.sin(x * y) / (x**2 + y**4), (x, y), (0, 0)),
    }
    out = {}
    for name, args in cases.items():
        old = bench(*args, False)
        new = bench(*args, True)
        out[name] = {
            "uncached_seconds": old,
            "cached_seconds": new,
            "ratio": new / old if old else None,
            "method": dispatch(*args, True),
        }
    Path("docs/germ_analysis_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
