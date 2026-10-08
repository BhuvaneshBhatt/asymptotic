"""Robust resumable executor for the 1,318-case multivariate reference corpus.

Coordinator never imports asymptotic. Every case runs in a fresh child process,
so SIGALRM, SymPy state, recursion failures, and hard worker timeouts cannot
poison later cases. Results are appended as JSONL after every completed case;
aggregation is deterministic and restartable.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_reference_corpus_shard.py"
CORPUS = ROOT / "tests" / "data" / "multivariate_limit_reference_cases.json"


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_atomic(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND + fsync means a killed coordinator loses at most the active case.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def child_env(extra_pythonpath):
    env = os.environ.copy()
    parts = [str(ROOT / "src")]
    if extra_pythonpath:
        parts.extend(extra_pythonpath)
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_case(index, case_timeout, hard_grace, extra_pythonpath):
    cmd = [
        sys.executable,
        str(RUNNER),
        "--case-index",
        str(index),
        "--timeout",
        str(case_timeout),
    ]
    started = time.monotonic()
    try:
        cp = subprocess.run(
            cmd,
            cwd=ROOT,
            env=child_env(extra_pythonpath),
            text=True,
            capture_output=True,
            timeout=case_timeout + hard_grace,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "index": index,
            "outcome": "TIMEOUT",
            "detail": "hard worker timeout",
            "seconds": round(time.monotonic() - started, 4),
        }
    lines = [x for x in cp.stdout.splitlines() if x.startswith("{")]
    if cp.returncode or not lines:
        detail = (cp.stderr or cp.stdout)[-1000:]
        return {
            "index": index,
            "outcome": "ERROR",
            "detail": f"worker exit {cp.returncode}: {detail}",
            "seconds": round(time.monotonic() - started, 4),
        }
    row = json.loads(lines[-1])
    row["seconds"] = round(time.monotonic() - started, 4)
    return row


def run(ns):
    cases = json.loads(CORPUS.read_text())
    out = Path(ns.output)
    if not 0 <= ns.shard < ns.shards:
        raise ValueError("shard must lie between zero and shards-1")
    if not ns.resume:
        out.unlink(missing_ok=True)
    existing = load_jsonl(out) if ns.resume else []
    done = {r["index"] for r in existing}
    indices = [
        i for i in range(len(cases)) if i % ns.shards == ns.shard and i not in done
    ]
    for i in indices:
        row = run_case(i, ns.timeout, ns.hard_grace, ns.pythonpath)
        row.setdefault("id", cases[i]["id"])
        row["shard"] = ns.shard
        append_atomic(out, row)
    rows = load_jsonl(out)
    print(json.dumps(collections.Counter(r["outcome"] for r in rows), sort_keys=True))


def aggregate_rows(rows, *, expected, expected_ids=None):
    """Build a deterministic integrity report from reference-corpus rows.

    Duplicate indices are reported instead of allowing the last shard
    encountered to win. Conflicting duplicates are a release-integrity error.
    """
    if type(expected) is not int or expected < 0:
        raise ValueError("expected corpus size must be a nonnegative integer")
    if expected_ids is not None and len(expected_ids) != expected:
        raise ValueError("expected IDs must match the corpus size")
    outcomes = {"PASS", "WRONG", "ERROR", "TIMEOUT", "UNKNOWN", "KNOWN_GAP"}
    grouped = collections.defaultdict(list)
    for row in rows:
        index = row.get("index")
        if type(index) is not int or not 0 <= index < expected:
            raise ValueError(f"reference index outside the corpus: {index!r}")
        if row.get("outcome") not in outcomes:
            raise ValueError(f"invalid reference outcome at index {index}")
        if expected_ids is not None and row.get("id") != expected_ids[index]:
            raise ValueError(f"reference ID does not match index {index}")
        grouped[index].append(row)
    duplicate_indices = sorted(i for i, group in grouped.items() if len(group) > 1)
    conflicting_duplicate_indices = sorted(
        i
        for i, group in grouped.items()
        if len(group) > 1
        and len({(r.get("id"), r.get("outcome"), r.get("detail")) for r in group}) > 1
    )
    by = {i: group[0] for i, group in grouped.items()}
    missing = [i for i in range(expected) if i not in by]
    counts = collections.Counter(r["outcome"] for r in by.values())
    return {
        "total_expected": expected,
        "total_completed": len(by),
        "missing_count": len(missing),
        "counts": dict(sorted(counts.items())),
        "missing_indices": missing,
        "duplicate_indices": duplicate_indices,
        "conflicting_duplicate_indices": conflicting_duplicate_indices,
        "wrong": [r for r in by.values() if r["outcome"] == "WRONG"],
        "errors": [r for r in by.values() if r["outcome"] == "ERROR"],
        "timeouts": [r for r in by.values() if r["outcome"] == "TIMEOUT"],
        "unknown": [r for r in by.values() if r["outcome"] == "UNKNOWN"],
        "known_gaps": [r for r in by.values() if r["outcome"] == "KNOWN_GAP"],
    }


def aggregate(ns):
    cases = json.loads(CORPUS.read_text())
    rows = []
    for p in sorted(Path(ns.directory).glob("shard_*.jsonl")):
        rows.extend(load_jsonl(p))
    report = aggregate_rows(
        rows, expected=len(cases), expected_ids=[case["id"] for case in cases]
    )
    Path(ns.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        k: report[k]
        for k in (
            "total_expected",
            "total_completed",
            "missing_count",
            "counts",
            "duplicate_indices",
            "conflicting_duplicate_indices",
        )
    }
    print(json.dumps(summary, sort_keys=True))
    if ns.strict and (
        report["missing_count"]
        or report["duplicate_indices"]
        or report["wrong"]
        or report["errors"]
        or report["timeouts"]
    ):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--shards", type=int, default=60)
    r.add_argument("--timeout", type=float, default=2)
    r.add_argument("--hard-grace", type=float, default=2)
    r.add_argument("--output", required=True)
    r.add_argument("--pythonpath", action="append", default=[])
    r.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--directory", required=True)
    a.add_argument("--output", required=True)
    a.add_argument(
        "--strict",
        action="store_true",
        help="fail on missing/conflicting rows or WRONG/ERROR/TIMEOUT outcomes",
    )
    ns = ap.parse_args()
    run(ns) if ns.cmd == "run" else aggregate(ns)


if __name__ == "__main__":
    main()
