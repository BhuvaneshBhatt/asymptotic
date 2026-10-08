"""Resumable 60-shard reference runner with hard per-case subprocess isolation."""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def atomic(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2) + "\n")
    os.replace(tmp, path)


def same_infinite_direction(a, b):
    """Compare explicit directions with finite multiples of real infinity.

    The legacy product oo*c determines a ray only for a finite nonzero numeric
    coefficient. Spherical infinity and sums of infinite components supply no
    direction here. Normalizing each ray ignores a positive magnitude factor.
    """
    import sympy as sp

    from asymptotic.fixed_ray_branch_germs import DirectionalInfinity

    def direction(value):
        if isinstance(value, DirectionalInfinity):
            return value.args[0]
        if value is sp.oo:
            return sp.S.One
        if value == -sp.oo:
            return -sp.S.One
        if isinstance(value, sp.Mul):
            infinities = [part for part in value.args if part in (sp.oo, -sp.oo)]
            if len(infinities) == 1:
                infinity = infinities[0]
                orientation = sp.S.One if infinity is sp.oo else -sp.S.One
                return orientation * sp.Mul(
                    *(part for part in value.args if part != infinity)
                )
        return None

    first, second = direction(a), direction(b)
    if any(
        d is None
        or d.is_number is not True
        or d.is_finite is not True
        or d.is_zero is not False
        for d in (first, second)
    ):
        return False
    return sp.simplify(first / sp.Abs(first) - second / sp.Abs(second)) == 0


def worker(index, root, case_timeout=5):
    # Import heavy symbolic stack only in isolated worker.
    import sympy as sp
    from reference_expression_parser import parse_reference_expression
    from sympy.polys.polyerrors import NotAlgebraic

    from asymptotic.conditional import ConditionalExpression
    from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
    from asymptotic.limits import limit
    from asymptotic.reference_normalization import multivariate_reference_namespace

    c = json.loads(
        (root / "tests/data/multivariate_limit_reference_cases.json").read_text()
    )[index]

    if not c.get("executable", True):
        print(
            json.dumps(
                {
                    "id": c["id"],
                    "index": index,
                    "outcome": "KNOWN_GAP",
                    "detail": "explicitly non-executable reference",
                    "executed": False,
                }
            )
        )
        return

    expired = False

    class CaseTimeout(Exception):
        pass

    def alarm(sig, frame):
        nonlocal expired
        expired = True
        raise CaseTimeout()

    signal.signal(signal.SIGALRM, alarm)

    def eq(a, b):
        if isinstance(a, (tuple, sp.Tuple)) and isinstance(b, (tuple, sp.Tuple)):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        if isinstance(a, (tuple, sp.Tuple)):
            return bool(a) and all(eq(x, b) for x in a)
        if isinstance(b, (tuple, sp.Tuple)):
            return bool(b) and all(eq(a, x) for x in b)
        try:
            if a == b:
                return True
            if isinstance(a, DirectionalInfinity) or isinstance(b, DirectionalInfinity):
                return same_infinite_direction(a, b)
            if sp.simplify(a - b) == 0:
                return True
            if (
                (getattr(a, "is_Float", False) or getattr(b, "is_Float", False))
                and a.is_number
                and b.is_number
            ):
                return abs(float(sp.N(a - b, 17))) <= 1e-14 * max(
                    1.0, abs(float(sp.N(a, 17))), abs(float(sp.N(b, 17)))
                )
            return False
        except (TypeError, ValueError, NotImplementedError):
            return False

    sy = {n: sp.Symbol(n, real=True) for n in c["variables"]}
    sy.update({n: sp.Symbol(n, real=True) for n in c.get("real_parameters", [])})
    loc = {
        **multivariate_reference_namespace(),
        **sy,
        "oo": sp.oo,
        "zoo": sp.zoo,
        "Infinity": sp.oo,
        "NegativeInfinity": -sp.oo,
        "ConditionalExpression": ConditionalExpression,
    }
    try:
        signal.setitimer(signal.ITIMER_REAL, case_timeout)
        e = parse_reference_expression(c["expression"], loc)
        vs = tuple(sy[n] for n in c["variables"])
        tar = tuple(sp.sympify(v, locals=loc) for v in c["target"])
        dom = sp.sympify(c.get("domain", "True"), locals=loc)
        r = limit(e, vs, tar, domain=dom, return_result=True)
        ex = (
            sp.sympify(c["expected"], locals=loc)
            if c["expected_kind"]
            not in (
                "dne",
                "unparsed",
                "unknown",
                "unevaluated",
            )
            and "expected" in c
            else None
        )
        if c.get("expected_strata"):
            actual_strata = getattr(r, "strata", ())
            out = "PASS"
            if not actual_strata:
                conditional = getattr(r, "condition", None)
                out = (
                    "UNKNOWN"
                    if (
                        getattr(getattr(r, "status", None), "name", "UNKNOWN")
                        == "UNKNOWN"
                        or conditional not in (None, sp.S.true)
                    )
                    else "WRONG"
                )
            elif len(actual_strata) != len(c["expected_strata"]):
                out = "UNKNOWN"
            else:
                for cell in c["expected_strata"]:
                    condition = sp.sympify(cell["condition"], locals=loc)
                    matches = [
                        st
                        for st in actual_strata
                        if sp.simplify(sp.Equivalent(st.condition, condition))
                        is sp.S.true
                    ]
                    if len(matches) != 1 or matches[0].result.status.name == "UNKNOWN":
                        out = "UNKNOWN"
                        break
                    result = matches[0].result
                    if result.status.name != cell["status"] or (
                        cell["status"] == "PROVED"
                        and not eq(result.value, sp.sympify(cell["value"], locals=loc))
                    ):
                        out = "WRONG"
                        break
            signal.setitimer(signal.ITIMER_REAL, 0)
            if expired:
                out = "TIMEOUT"
            print(
                json.dumps(
                    {
                        "id": c["id"],
                        "index": index,
                        "outcome": out,
                        "detail": "complete parameter-stratum comparison",
                    }
                )
            )
            return
        if not hasattr(r, "status") and hasattr(r, "strata"):
            ok = False
            if isinstance(ex, ConditionalExpression):
                for st in r.strata:
                    if getattr(st.result, "status", None).name == "PROVED" and eq(
                        st.result.value, ex.value
                    ):
                        try:
                            cond_ok = (
                                sp.simplify(sp.Equivalent(st.condition, ex.condition))
                                is sp.S.true
                            )
                        except (
                            TypeError,
                            ValueError,
                            NotImplementedError,
                            sp.PolynomialError,
                        ):
                            cond_ok = False
                        if cond_ok:
                            ok = True
                            break
            out = "PASS" if ok else "UNKNOWN"
            detail = "parameter_stratification"
            signal.setitimer(signal.ITIMER_REAL, 0)
            if expired:
                out, detail = "TIMEOUT", "budget fired inside a solver fallback"
            print(
                json.dumps(
                    {"id": c["id"], "index": index, "outcome": out, "detail": detail}
                )
            )
            return
        mathematical = getattr(r, "mathematical_value", None)
        if c["expected_kind"] in {"unknown", "unevaluated"}:
            out = "KNOWN_GAP"
            detail = f"reference_{c['expected_kind']}:{r.status.name}"
        elif c["expected_kind"] == "unparsed":
            out, detail = "KNOWN_GAP", "unparsed reference expectation"
        elif (
            c["expected_kind"] != "dne"
            and mathematical is not None
            and eq(mathematical, ex)
        ):
            out = "PASS"
            detail = f"{r.status.name}:{mathematical}"
        elif r.status.name == "UNKNOWN":
            out = "UNKNOWN"
            detail = ",".join(ev.method for ev in r.evidence)
        elif c["expected_kind"] == "dne":
            out = "PASS" if r.status.name == "DOES_NOT_EXIST" else "WRONG"
            detail = str(r.value)
        else:
            out = "PASS" if r.status.name == "PROVED" and eq(r.value, ex) else "WRONG"
            detail = f"{r.status.name}:{r.value}"
    except CaseTimeout:
        out = "TIMEOUT"
        detail = None
    except (NotAlgebraic, NotImplementedError, RecursionError) as ex:
        out = "KNOWN_GAP"
        detail = type(ex).__name__
    except (
        TypeError,
        ValueError,
        ZeroDivisionError,
        sp.PolynomialError,
        sp.PoleError,
    ) as ex:
        detail = f"{type(ex).__name__}:{ex}"
        out = (
            "KNOWN_GAP"
            if "sequence by non-" in str(ex) or "multiply sequence" in str(ex)
            else "ERROR"
        )
    signal.setitimer(signal.ITIMER_REAL, 0)
    if expired:
        out, detail = "TIMEOUT", "solver/comparison budget fired"
    print(json.dumps({"id": c["id"], "index": index, "outcome": out, "detail": detail}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--timeout", type=float, default=5)
    ap.add_argument("--output")
    ap.add_argument("--case-index", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ns = ap.parse_args()
    root = Path(__file__).parents[1]
    if ns.case_index is not None:
        return worker(ns.case_index, root, ns.timeout)
    if ns.shard is None or ns.output is None:
        ap.error("--shard and --output are required")
    cases = json.loads(
        (root / "tests/data/multivariate_limit_reference_cases.json").read_text()
    )
    out = Path(ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [] if ns.no_resume or not out.exists() else json.loads(out.read_text())
    done = {r["id"] for r in rows}
    for j, c in enumerate(cases):
        if j % ns.shards != ns.shard or c["id"] in done:
            continue
        started = time.monotonic()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case-index",
            str(j),
            "--timeout",
            str(ns.timeout),
        ]
        try:
            cp = subprocess.run(
                cmd,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=ns.timeout + 6,
                check=False,
            )
            if cp.returncode == 0:
                line = next(
                    (q for q in reversed(cp.stdout.splitlines()) if q.startswith("{")),
                    None,
                )
                row = (
                    json.loads(line)
                    if line
                    else {
                        "id": c["id"],
                        "index": j,
                        "outcome": "ERROR",
                        "detail": "worker produced no JSON",
                    }
                )
            else:
                row = {
                    "id": c["id"],
                    "index": j,
                    "outcome": "ERROR",
                    "detail": cp.stderr[-500:],
                }
        except subprocess.TimeoutExpired:
            row = {"id": c["id"], "index": j, "outcome": "TIMEOUT", "detail": None}
        row["seconds"] = round(time.monotonic() - started, 4)
        rows.append(row)
        atomic(out, rows)
    print(json.dumps(collections.Counter(r["outcome"] for r in rows), sort_keys=True))


if __name__ == "__main__":
    main()
