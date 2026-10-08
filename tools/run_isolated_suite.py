"""Run the complete pytest suite sequentially in fresh subprocesses.

The coordinator never imports the package under test.  It first asks pytest for
stable node IDs, then launches exactly one pytest child at a time.  Progress is
written atomically after every child so an interrupted run can be resumed.
Modules that exceed their budget can optionally be retried item-by-item, which
identifies a pathological test without losing the rest of the module.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / ".isolated-suite" / "state.json"
DEFAULT_LOG_DIR = ROOT / ".isolated-suite" / "logs"


@dataclass(frozen=True)
class RunResult:
    """Persistent result of one isolated pytest child process."""

    nodeid: str
    scope: str
    outcome: str
    returncode: int | None
    seconds: float
    log: str


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Do not let caller-specific pytest flags change release semantics.
    env.pop("PYTEST_ADDOPTS", None)
    return env


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _collect_nodeids(pytest_args: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *pytest_args,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, 5):
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"pytest collection failed with exit code {completed.returncode}"
        )
    nodeids = [
        line.strip()
        for line in completed.stdout.splitlines()
        if "::" in line
        and not line.startswith("=")
        and (ROOT / line.strip().split("::", 1)[0]).is_file()
    ]
    if not nodeids and completed.returncode != 5:
        raise RuntimeError("pytest collection succeeded but produced no test node IDs")
    return nodeids


def _module(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe[-180:]


def _run_child(
    target: str,
    *,
    scope: str,
    timeout: float,
    log_dir: Path,
    pytest_args: list[str],
    selected_nodeids: list[str] | None = None,
) -> RunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_slug(target)}.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *(selected_nodeids or [target]),
        *pytest_args,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        seconds = time.monotonic() - started
        log_path.write_text(
            f"$ {' '.join(command)}\n\n{completed.stdout}\n{completed.stderr}"
        )
        outcome = "passed" if completed.returncode == 0 else "failed"
        return RunResult(
            target,
            scope,
            outcome,
            completed.returncode,
            round(seconds, 4),
            str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        seconds = time.monotonic() - started
        stdout = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        )
        log_path.write_text(f"$ {' '.join(command)}\n\n{stdout}\n{stderr}")
        return RunResult(
            target, scope, "timeout", None, round(seconds, 4), str(log_path)
        )


def _load_results(path: Path, *, resume: bool) -> list[RunResult]:
    if not resume or not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [RunResult(**row) for row in payload.get("results", [])]


def _save(path: Path, *, nodeids: list[str], results: list[RunResult]) -> None:
    counts = Counter(result.outcome for result in results)
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "collected": len(nodeids),
            "completed_children": len(results),
            "counts": dict(sorted(counts.items())),
            "results": [asdict(result) for result in results],
        },
    )


def _summary(nodeids: list[str], results: list[RunResult]) -> dict[str, object]:
    counts = Counter(result.outcome for result in results)
    bad = [result.nodeid for result in results if result.outcome != "passed"]
    return {
        "collected": len(nodeids),
        "children": len(results),
        "counts": dict(sorted(counts.items())),
        "nonpass": bad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolation",
        choices=("module", "test"),
        default="module",
        help="fresh process per module (default) or per collected test item",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--split-timeouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retry a timed-out module as isolated test items (default: enabled)",
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="pytest selection arguments after '--', e.g. -- tests -m 'not slow'",
    )
    args = parser.parse_args()
    pytest_args = (
        args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    )
    if not pytest_args:
        pytest_args = ["tests"]

    nodeids = _collect_nodeids(pytest_args)
    by_module: dict[str, list[str]] = defaultdict(list)
    for nodeid in nodeids:
        by_module[_module(nodeid)].append(nodeid)

    results = _load_results(args.state, resume=args.resume)
    completed = {(result.scope, result.nodeid) for result in results}

    if args.isolation == "test":
        work = [("test", nodeid) for nodeid in nodeids]
    else:
        work = [("module", module) for module in sorted(by_module)]

    print(
        f"collected {len(nodeids)} tests; {len(work)} initial isolated children; "
        "concurrency=1",
        flush=True,
    )
    for index, (scope, target) in enumerate(work, start=1):
        if (scope, target) in completed:
            continue
        print(f"[{index}/{len(work)}] {scope}: {target}", flush=True)
        result = _run_child(
            target,
            scope=scope,
            timeout=args.timeout,
            log_dir=args.log_dir,
            pytest_args=[],
            selected_nodeids=by_module[target] if scope == "module" else None,
        )
        results.append(result)
        completed.add((scope, target))
        _save(args.state, nodeids=nodeids, results=results)
        print(f"  {result.outcome} in {result.seconds:.2f}s", flush=True)

        if scope == "module" and result.outcome == "timeout" and args.split_timeouts:
            items = by_module[target]
            print(f"  splitting timeout into {len(items)} isolated tests", flush=True)
            for item_index, nodeid in enumerate(items, start=1):
                if ("test", nodeid) in completed:
                    continue
                print(f"    [{item_index}/{len(items)}] {nodeid}", flush=True)
                item_result = _run_child(
                    nodeid,
                    scope="test",
                    timeout=args.timeout,
                    log_dir=args.log_dir,
                    pytest_args=[],
                )
                results.append(item_result)
                completed.add(("test", nodeid))
                _save(args.state, nodeids=nodeids, results=results)
                print(
                    f"      {item_result.outcome} in {item_result.seconds:.2f}s",
                    flush=True,
                )

    summary = _summary(nodeids, results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    # A module timeout is diagnostic rather than fatal when every split item passed.
    fatal = []
    for result in results:
        if result.outcome == "passed":
            continue
        if (
            result.scope == "module"
            and result.outcome == "timeout"
            and args.split_timeouts
            and all(
                ("test", nodeid) in completed
                and any(
                    r.scope == "test" and r.nodeid == nodeid and r.outcome == "passed"
                    for r in results
                )
                for nodeid in by_module.get(result.nodeid, ())
            )
        ):
            continue
        fatal.append(result)
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
