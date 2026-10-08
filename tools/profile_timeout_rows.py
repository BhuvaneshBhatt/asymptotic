"""Profile every recorded timeout, saving snapshots before hard termination."""

import argparse
import contextlib
import cProfile
import hashlib
import importlib.util
import json
import os
import pstats
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit/timeout-profiles"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="audit/univariate-audit.json")
    parser.add_argument("--output", default="audit/timeout-profiles")
    parser.add_argument("--ids")
    parser.add_argument("--budget", type=float, default=30)
    parser.add_argument("--wall-budget", type=float, default=40)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1 or args.budget <= 0 or args.wall_budget <= 0:
        parser.error("jobs and diagnostic budgets must be positive")
    global OUT
    OUT = ROOT / args.output
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "corpus_audit", ROOT / "tools/audit_univariate_corpus.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for module in ("sympy", "asymptotic", "semialg", "exprtest"):
        __import__(module)

    original = json.loads((ROOT / args.input).read_text())
    selected = set(args.ids.split(",")) if args.ids else None
    pending = iter(
        r
        for r in original["rows"]
        if r["classification"] == "timeout_performance_failure"
        and (selected is None or r["id"] in selected)
    )
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    active = {}
    done = []
    exhausted = False
    while active or not exhausted:
        while len(active) < args.jobs and not exhausted:
            try:
                row = next(pending)
            except StopIteration:
                exhausted = True
                break
            stem = OUT / row["id"]
            pid = os.fork()
            if pid == 0:
                profile = cProfile.Profile()

                def snapshot(signum, frame, profile=profile, stem=stem):
                    profile.dump_stats(str(stem) + ".prof")
                    profile.enable()
                    Path(str(stem) + ".stack.json").write_text(
                        json.dumps(traceback.format_stack(frame), indent=2)
                    )

                signal.signal(signal.SIGUSR1, snapshot)
                profile.enable()
                try:
                    with (
                        Path(str(stem) + ".result.json").open("w") as f,
                        contextlib.redirect_stdout(f),
                    ):
                        runner.worker(row["index"], args.budget)
                finally:
                    profile.disable()
                    profile.dump_stats(str(stem) + ".prof")
                os._exit(0)
            active[pid] = (row, stem, time.monotonic(), False)
        for pid, (row, stem, start, saved) in list(active.items()):
            ended, status = os.waitpid(pid, os.WNOHANG)
            elapsed = time.monotonic() - start
            if not ended and elapsed > 4 and not saved:
                os.kill(pid, signal.SIGUSR1)
                active[pid] = (row, stem, start, True)
            if not ended and elapsed < args.wall_budget:
                continue
            if not ended:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            record = {
                "id": row["id"],
                "seconds": elapsed,
                "hard_timeout": not bool(ended),
                "diagnostic_budget_seconds": args.budget,
                "process_wall_budget_seconds": args.wall_budget,
                "profiling_enabled": True,
                "source_sha256": source_hash.hexdigest(),
                "reference": row["reference"],
            }
            path = Path(str(stem) + ".prof")
            if path.exists():
                stats = pstats.Stats(str(path))
                functions = []
                for (file, line, function), (
                    primitive,
                    calls,
                    total,
                    cumulative,
                    callers,
                ) in stats.stats.items():
                    functions.append(
                        dict(
                            file=file,
                            line=line,
                            function=function,
                            calls=calls,
                            total=total,
                            cumulative=cumulative,
                        )
                    )
                record["top_functions"] = sorted(
                    functions, key=lambda x: x["cumulative"], reverse=True
                )[:35]
            try:
                record["result"] = json.loads(
                    Path(str(stem) + ".result.json").read_text().splitlines()[-1]
                )
            except (ValueError, IndexError):
                pass
            done.append(record)
            del active[pid]
            (OUT / "summary.json").write_text(json.dumps(done, indent=2) + "\n")
            print(row["id"], round(elapsed, 2), flush=True)
        time.sleep(0.03)


if __name__ == "__main__":
    main()
