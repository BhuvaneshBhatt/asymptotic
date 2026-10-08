"""Profile individual failure regressions in fresh, bounded interpreters."""

import cProfile
import importlib.util
import json
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tests/limits/multivariate/algebraic/test_failure_regressions.py"
OUT = ROOT / "audit/failure-profiles"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        sys.path.insert(0, str(ROOT / "src"))
        spec = importlib.util.spec_from_file_location("failure_regressions", MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = sys.argv[1]
        profile = cProfile.Profile()
        start = time.perf_counter()
        profile.runcall(getattr(module, name))
        elapsed = time.perf_counter() - start
        profile.dump_stats(str(OUT / (name + ".prof")))
        stats = pstats.Stats(profile)
        functions = []
        for (file, line, function), (
            primitive,
            calls,
            total,
            cumulative,
            callers,
        ) in stats.stats.items():
            if "/asymptotic/" in file:
                functions.append(
                    dict(
                        file=file.split("/src/")[-1],
                        line=line,
                        function=function,
                        calls=calls,
                        total_seconds=total,
                        cumulative_seconds=cumulative,
                    )
                )
        functions.sort(key=lambda r: r["cumulative_seconds"], reverse=True)
        print(
            json.dumps(dict(test=name, seconds=elapsed, top_functions=functions[:25]))
        )
        return
    import ast

    names = [
        n.name
        for n in ast.parse(MODULE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    results = []
    for name in names:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        try:
            p = subprocess.run(
                [sys.executable, __file__, name],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            result = (
                json.loads(p.stdout.splitlines()[-1])
                if p.returncode == 0
                else dict(test=name, error=p.stderr[-2000:])
            )
        except subprocess.TimeoutExpired:
            result = dict(test=name, error="30-second process budget exceeded")
        results.append(result)
        print(name, result.get("seconds", result.get("error")), flush=True)
        (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
