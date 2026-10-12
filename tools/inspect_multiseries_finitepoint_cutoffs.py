"""Read-only inspection of asymptotic.multiseries at finite points.

Run in an environment with `asymptotic` installed, from the repository root:
    python tools/inspect_multiseries_finitepoint_cutoffs.py

This reproduces the exact cutoffs *without* importing MultipleIntegrate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_multiseries_finitepoint_cutoffs import (
    canonical_tangent_cutoff,
    finite_part_of_series,
    weighted_tangent_cutoff,
)

from asymptotic import multiseries


def inspect_case(name: str, expr: sp.Expr, epsilon: sp.Symbol) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("expression:", sp.sstr(expr), flush=True)
    oracle = finite_part_of_series(expr, epsilon)
    print("SymPy finite constant:", oracle, flush=True)
    try:
        native = multiseries(
            expr,
            epsilon,
            point=0,
            terms=8,
            return_result=True,
            allow_series_fallback=True,
        )
        print("scale:", tuple(getattr(native.scale, "exprs", ())), flush=True)
        print("level:", getattr(native, "level", None), flush=True)
        try:
            formal, symbols = native._formal()
            print("formal symbols:", symbols, flush=True)
            print("formal expression:", sp.sstr(formal)[:900], flush=True)
        except Exception as exc:  # noqa: BLE001
            print("formalization ERROR:", type(exc).__name__, repr(exc), flush=True)
        try:
            terms = native.terms(8)
            print("number of terms:", len(terms), flush=True)
            for term in terms:
                print(
                    f"  exponent={term.exponent} coefficient={str(term.coefficient)[:450]}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print("native terms ERROR:", type(exc).__name__, repr(exc), flush=True)
        print(
            "obligations:",
            tuple(str(x)[:220] for x in getattr(native, "obligation_history", ())),
            flush=True,
        )
        try:
            prefix = sp.sympify(native.truncate(8))
            print("prefix:", sp.sstr(prefix)[:1100], flush=True)
            print(
                "prefix retains dilogarithm:", bool(prefix.has(sp.polylog)), flush=True
            )
            print(
                "prefix finite constant after SymPy re-expansion:",
                finite_part_of_series(prefix, epsilon),
                flush=True,
            )
            print(
                "prefix minus original tends to zero:",
                sp.limit(prefix - expr, epsilon, 0, dir="+"),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print("prefix ERROR:", type(exc).__name__, repr(exc), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("multiseries ERROR:", type(exc).__name__, repr(exc), flush=True)


def main() -> None:
    e = sp.Symbol("epsilon", positive=True)
    print("SymPy:", sp.__version__)
    import asymptotic

    print("asymptotic:", getattr(asymptotic, "__version__", "?"))
    inspect_case("finite analytic germ 1+epsilon", 1 + e, e)
    inspect_case(
        "Laurent log 1/epsilon+log**2+constant", 1 / e + sp.log(e) ** 2 + 7 + e, e
    )
    inspect_case("exact tangent m=2 k=1", canonical_tangent_cutoff(e, 2, 1), e)
    inspect_case("exact tangent m=2 k=2", canonical_tangent_cutoff(e, 2, 2), e)
    inspect_case("weighted tangent m=2 k=-3", weighted_tangent_cutoff(e, 2, -3), e)


if __name__ == "__main__":
    main()
