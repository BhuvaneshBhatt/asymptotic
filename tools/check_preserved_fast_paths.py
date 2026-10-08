"""Check preserved fast paths under multiple independent hash seeds."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) > 1:
        sys.path.insert(0, str(ROOT / "src"))
        import sympy as sp

        from asymptotic import limit

        x, y = sp.symbols("x y", real=True)
        r2 = x * x + y * y
        examples = {
            "0721": (
                2 * r2 ** sp.Rational(3, 2) * sp.cos(1 / r2) - sp.sin(1 / r2),
                (x, y),
                (0, 0),
            ),
            "atanh_endpoint": (sp.atanh(2 * x / (1 + x * x + y * y)), (x, y), (1, 0)),
        }
        records = []
        for name, (expression, variables, point) in examples.items():
            start = time.perf_counter()
            result = limit(expression, variables, point, return_result=True)
            records.append(
                dict(
                    case=name,
                    seconds=time.perf_counter() - start,
                    status=result.status.value,
                    value=str(result.value),
                    evidence=[e.method for e in result.evidence],
                )
            )
        assert records[0]["status"] == "does_not_exist"
        assert records[0]["evidence"][0] == "vanishing_perturbation_cluster"
        assert records[1]["value"] == "oo"
        print(json.dumps(records))
        return
    records = []
    for seed in (0, 1, 2, 42, 117, 721):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        p = subprocess.run(
            [sys.executable, __file__, "worker"],
            env=env,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
            check=True,
        )
        for r in json.loads(p.stdout.splitlines()[-1]):
            records.append(dict(hash_seed=seed, **r))
    (ROOT / "audit/preserved-fast-paths.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
