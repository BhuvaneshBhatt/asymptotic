"""Process-isolated, resumable univariate corpus audit; never xfail a discrepancy.

Each executable row receives a disposable isolated process. In fork mode,
imports are preloaded without solver calls; fresh-interpreter mode includes
imports in the wall budget. The soft budget covers parsing, solver and comparison. A
discrepancy is evidence for review, not proof that either side is mathematically
correct. Reports preserve that distinction in ``review_required``.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/univariate_limit_reference_cases.json"
CATEGORIES = (
    "pass",
    "solver_wrong_result",
    "bad_reference_or_missing_assumptions",
    "unsupported_capability",
    "timeout_performance_failure",
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def evaluate_reference_case(row, namespace, record):
    import sympy as sp

    from asymptotic import complex_limit, complex_ray_limit
    from asymptotic.analytic_limits import analytic_limit, normalize_direction
    from asymptotic.reference_contracts import conditional_reference, parse_reference
    from asymptotic.reference_normalization import (
        scalar_reference_assumptions,
        scalar_reference_equal,
    )

    row = dict(row)
    row["direction"] = normalize_direction(row.get("direction"))
    variable = sp.Symbol(row["variable"])
    expression = parse_reference(row["expression"], namespace)
    point = parse_reference(row["point"], namespace)
    assumptions = scalar_reference_assumptions(
        row.get("assumptions", "True"), namespace
    )
    expected = None
    if row.get("expected_kind") == "value":
        expected = parse_reference(row["expected"], namespace)
        expected, condition = conditional_reference(expected)
        if condition is not sp.S.true:
            if condition.has(variable):
                raise ValueError(
                    "expected parameter condition cannot constrain the approach variable"
                )
            assumptions = sp.And(assumptions, condition)
            record["comparison_condition"] = str(condition)
    if row.get("expected_condition"):
        condition = scalar_reference_assumptions(row["expected_condition"], namespace)
        if condition.has(variable):
            raise ValueError(
                "expected parameter condition cannot constrain the approach variable"
            )
        assumptions = sp.And(assumptions, condition)
        record["comparison_condition"] = str(condition)
    substitutions = {
        sp.Symbol(k): sp.sympify(v, locals=namespace)
        for k, v in row.get("parameter_substitutions", {}).items()
    }
    if variable in substitutions:
        raise ValueError("approach variable cannot be substituted")
    if substitutions:
        expression = expression.subs(substitutions)
        assumptions = assumptions.subs(substitutions)
    if (
        not row.get("ray")
        and row.get("approach") != "complex"
        and row.get("direction") not in (None, "+", "-")
    ):
        record.update(
            classification="unsupported_capability",
            detail="unsupported direction: " + row["direction"],
            review_required=True,
        )
    else:
        record["executed"] = True
        if row.get("expected_kind") == "cluster_set":
            from asymptotic.cluster_limits import Circle, cluster_set

            result = cluster_set(
                expression,
                variable,
                point,
                direction=row.get("direction"),
                assumptions=assumptions,
                return_result=True,
            )
            record.update(
                status="proved" if result.certified else "unknown",
                actual=str(result.cluster_set),
                evidence=[item.method for item in result.evidence],
            )
            if not result.certified:
                record.update(
                    classification="unsupported_capability",
                    detail="exact cluster set unresolved",
                )
                return
            target = row["expected_cluster"]
            if target["type"] == "circle":
                expected_set = Circle(
                    parse_reference(target["center"], namespace),
                    parse_reference(target["radius"], namespace),
                )
            elif target["type"] == "interval":
                expected_set = sp.Interval(
                    parse_reference(target["low"], namespace),
                    parse_reference(target["high"], namespace),
                )
            else:
                raise ValueError("unsupported cluster comparison schema")
            equal = result.cluster_set == expected_set
            record.update(
                classification="pass" if equal else "solver_wrong_result",
                detail="exact attained-cluster comparison",
                review_required=not equal,
            )
            return
        if row.get("approach") == "complex":
            result = complex_limit(
                expression, variable, point, assumptions=assumptions, return_result=True
            )
        elif row.get("ray"):
            result = complex_ray_limit(
                expression,
                variable,
                point,
                ray=sp.sympify(row["ray"], locals=namespace),
                assumptions=assumptions,
                return_result=True,
            )
        else:
            result = analytic_limit(
                expression,
                variable,
                point,
                direction=row.get("direction"),
                analytic_functions=row.get("analytic_functions", ()),
                assumptions=assumptions,
                return_result=True,
            )
        status = getattr(getattr(result, "status", None), "value", "")
        actual = getattr(result, "value", result)
        record.update(
            status=status,
            actual=str(actual),
            evidence=[e.method for e in getattr(result, "evidence", ())],
        )
        from asymptotic.stratification import AsymptoticStratification

        if isinstance(result, AsymptoticStratification):
            record["solver_parameter_strata"] = [
                dict(
                    condition=str(cell.condition),
                    status=cell.result.status.value,
                    actual=str(cell.result.value),
                    evidence=[e.method for e in cell.result.evidence],
                )
                for cell in result.strata
            ]
            record["evidence"] = ["parameter_strata_reference_contract"] + [
                e.method for cell in result.strata for e in cell.result.evidence
            ]
        if not status:
            record.update(
                classification="unsupported_capability",
                detail="parameter-stratified/non-scalar result",
                review_required=True,
            )
        elif status == "unknown":
            record.update(
                classification="unsupported_capability", detail="solver unresolved"
            )
        elif status in ("does_not_exist", "dne"):
            record.update(
                classification="pass"
                if row["expected_kind"] == "dne"
                else "solver_wrong_result",
                detail="solver proves nonexistence",
                review_required=row["expected_kind"] != "dne",
            )
        elif row["expected_kind"] == "unevaluated":
            record.update(
                classification="bad_reference_or_missing_assumptions",
                detail="historically unevaluated reference now has a solver result",
                review_required=True,
            )
        elif row["expected_kind"] == "dne":
            record.update(
                classification="solver_wrong_result",
                detail="value versus DNE discrepancy",
                review_required=True,
            )
        else:
            if expected is None:
                expected = parse_reference(row["expected"], namespace)
            if substitutions:
                expected = expected.subs(substitutions)
            equal = scalar_reference_equal(
                actual,
                expected,
                row.get("comparison_codomain", "exact"),
                assumptions=assumptions,
            )
            record["comparison_codomain"] = row.get("comparison_codomain", "exact")
            record.update(
                classification="pass" if equal else "solver_wrong_result",
                detail="symbolic equality" if equal else "value discrepancy",
                review_required=not equal,
            )


def worker(index, budget):
    started = time.monotonic()
    row = json.loads(DATA.read_text())[index]
    record = {
        "id": row["id"],
        "index": index,
        "reference": row,
        "review_required": False,
        "executed": False,
    }
    if not row.get("executable", True):
        record.update(
            classification="unsupported_capability",
            detail="explicitly non-executable reference; solver not invoked",
            review_required=True,
        )
        print(json.dumps(record))
        return
    sys.path.insert(0, str(ROOT / "src"))

    class BudgetExceeded(Exception):
        pass

    budget_expired = [False]

    def alarm(*args):
        budget_expired[0] = True
        raise BudgetExceeded()

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, budget)
    from asymptotic.reference_normalization import (
        scalar_reference_namespace,
    )

    namespace = scalar_reference_namespace()
    # Retain the corpus parser vocabulary. Unknown source-language heads must
    # remain explicit capability gaps rather than acquire invented semantics.
    try:
        if row.get("expected_kind") == "stratified" or row.get("applications"):
            outcomes = []
            record["strata"] = outcomes
            field = "applications" if row.get("applications") else "parameter_cases"
            record["evaluation_group"] = (
                "source_applications" if field == "applications" else "parameter_strata"
            )
            for stratum in row[field]:
                case = dict(row)
                case.pop(field)
                case.update(stratum)
                item = {
                    "stratum_id": stratum.get("stratum_id", stratum.get("case_id")),
                    "reference": case,
                    "executed": False,
                }
                outcomes.append(item)
                evaluate_reference_case(case, namespace, item)
                record["executed"] = any(
                    child.get("executed", False) for child in outcomes
                )
            classes = [item["classification"] for item in outcomes]
            category = (
                "solver_wrong_result"
                if "solver_wrong_result" in classes
                else "unsupported_capability"
                if "unsupported_capability" in classes
                else "pass"
            )
            record.update(
                classification=category,
                executed=any(item["executed"] for item in outcomes),
                status="proved" if category == "pass" else "unknown",
                actual="source-mapped results"
                if field == "applications"
                else "parameter-stratified results",
                review_required=category == "solver_wrong_result",
                detail="all reconstructed applications share the row budget; sub-results retained"
                if field == "applications"
                else "all parameter strata share the row budget; sub-results retained",
                evidence=[
                    method for item in outcomes for method in item.get("evidence", [])
                ],
            )
        else:
            evaluate_reference_case(row, namespace, record)
    except BudgetExceeded:
        record.update(
            classification="timeout_performance_failure",
            detail="solver/comparison budget exceeded",
        )
    except (
        ValueError,
        TypeError,
        AttributeError,
        NameError,
        SyntaxError,
        ArithmeticError,
        NotImplementedError,
        RuntimeError,
    ) as exc:
        record.update(
            classification="unsupported_capability",
            detail=f"{type(exc).__name__}: {exc}",
            exception=type(exc).__name__,
            review_required=True,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    record["soft_budget_fired"] = budget_expired[0]
    if budget_expired[0]:
        record["classification_before_budget_check"] = record.get("classification")
        record.update(
            classification="timeout_performance_failure",
            detail="soft solver/comparison budget fired; preserved even if an internal fallback caught the interruption",
        )
    record["elapsed_seconds"] = round(time.monotonic() - started, 6)
    print(json.dumps(record))


def run_one(index, budget, wall):
    started = time.monotonic()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(index),
        "--budget",
        str(budget),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=wall)
        record = json.loads(stdout.splitlines()[-1])
        if stderr:
            record["stderr"] = stderr[-2000:]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        record = {
            "index": index,
            "classification": "timeout_performance_failure",
            "detail": "hard process wall budget exceeded",
            "review_required": False,
            "executed": True,
        }
    except (ValueError, IndexError):
        record = {
            "index": index,
            "classification": "unsupported_capability",
            "detail": "worker failed: " + stderr[-2000:],
            "review_required": True,
            "executed": False,
        }
    record["wall_seconds"] = round(time.monotonic() - started, 6)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--budget", type=float, default=5)
    parser.add_argument("--wall-budget", type=float, default=15)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output", default="audit/corpus-run.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--indices", help="comma-separated zero-based indices")
    parser.add_argument(
        "--fresh-interpreter",
        action="store_true",
        help="use independent interpreters instead of independent fork children",
    )
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.budget)
        return
    rows = json.loads(DATA.read_text())
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    records = {}
    if args.resume and Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text())
        current_contract = {
            "source_sha256": source_hash.hexdigest(),
            "corpus_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
            "solver_budget_seconds": args.budget,
            "process_wall_budget_seconds": args.wall_budget,
        }
        mismatched = [
            key for key, value in current_contract.items() if previous.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                "cannot resume an incompatible audit: " + ", ".join(mismatched)
            )
        records = {r["index"]: r for r in previous["rows"]}
    indices = (
        list(map(int, args.indices.split(",")))
        if args.indices
        else list(range(len(rows)))
    )
    started = time.monotonic()

    def save():
        ordered = sorted(records.values(), key=lambda r: r["index"])
        counts = collections.Counter(r["classification"] for r in ordered)
        report = {
            "schema_version": 1,
            "corpus_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "soft_budget_accounting": "deadline is recorded even if an internal fallback catches its exception",
            "total_reference_rows": len(rows),
            "completed_rows": len(ordered),
            "source_sha256": source_hash.hexdigest(),
            "isolation": "fresh interpreter"
            if args.fresh_interpreter
            else "fresh fork child per row; no solver calls in parent",
            "complete": len(ordered) == len(rows),
            "solver_budget_seconds": args.budget,
            "process_wall_budget_seconds": args.wall_budget,
            "jobs": args.jobs,
            "counts": {k: counts[k] for k in CATEGORIES},
            "ids_by_classification": {
                k: [r["id"] for r in ordered if r["classification"] == k]
                for k in CATEGORIES
            },
            "review_required_count": sum(
                r.get("review_required", False) for r in ordered
            ),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "rows": ordered,
            "interpretation": "Unreviewed discrepancies are provisionally solver_wrong_result; agreement is not an independent proof of the reference. Explicitly non-executable rows are inventoried without solver execution.",
        }
        atomic(args.output, report)

    if not args.fresh_interpreter:
        # Preload only imports in the parent. Every solver invocation runs in a
        # disposable process with separate caches and a separate wall watchdog.
        sys.path.insert(0, str(ROOT / "src"))
        for module in ("sympy", "asymptotic", "semialg", "exprtest"):
            importlib.import_module(module)
        pending = iter(i for i in indices if i not in records)
        active = {}
        exhausted = False
        with tempfile.TemporaryDirectory(prefix="univariate-audit-") as directory:
            while active or not exhausted:
                while len(active) < args.jobs and not exhausted:
                    try:
                        i = next(pending)
                    except StopIteration:
                        exhausted = True
                        break
                    path = Path(directory) / f"{i}.json"
                    pid = os.fork()
                    if pid == 0:
                        try:
                            with path.open("w") as out, contextlib.redirect_stdout(out):
                                worker(i, args.budget)
                            os._exit(0)
                        except (OSError, RuntimeError):
                            os._exit(1)
                    active[pid] = (i, path, time.monotonic())
                for pid, (i, path, launched) in list(active.items()):
                    ended, exit_status = os.waitpid(pid, os.WNOHANG)
                    if not ended and time.monotonic() - launched < args.wall_budget:
                        continue
                    if not ended:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                        record = {
                            "classification": "timeout_performance_failure",
                            "detail": "hard process wall budget exceeded",
                            "executed": True,
                            "review_required": False,
                        }
                    else:
                        try:
                            record = json.loads(path.read_text().splitlines()[-1])
                        except (ValueError, IndexError, OSError):
                            record = {
                                "classification": "unsupported_capability",
                                "detail": f"worker exited without a record: {exit_status}",
                                "executed": False,
                                "review_required": True,
                            }
                    record.update(
                        index=i,
                        id=rows[i]["id"],
                        reference=rows[i],
                        wall_seconds=round(time.monotonic() - launched, 6),
                    )
                    records[i] = record
                    del active[pid]
                    if len(records) % 50 == 0:
                        save()
                        print(
                            f"{len(records)}/{len(rows)} {dict(collections.Counter(r['classification'] for r in records.values()))}",
                            flush=True,
                        )
                time.sleep(0.02)
        save()
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, i, args.budget, args.wall_budget): i
            for i in indices
            if i not in records
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            record = future.result()
            record.setdefault("id", rows[i]["id"])
            record.setdefault("reference", rows[i])
            records[i] = record
            if len(records) % 25 == 0:
                save()
                print(
                    f"{len(records)}/{len(rows)} {dict(collections.Counter(r['classification'] for r in records.values()))}",
                    flush=True,
                )
    save()


if __name__ == "__main__":
    main()
