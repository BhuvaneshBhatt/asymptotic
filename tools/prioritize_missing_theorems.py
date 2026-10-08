"""Rank theorem families from a capability classification report."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("audit/capability-classification.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("audit/theorem-priorities.json")
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    groups = defaultdict(list)
    for row in report["rows"]:
        if row["reason_group"] == "missing certified theorem in solver dispatch":
            groups[row["mathematical_family"]].append(row["id"])
    ranked = sorted(
        [dict(family=key, count=len(ids), ids=ids) for key, ids in groups.items()],
        key=lambda row: (-row["count"], row["family"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranked, indent=2) + "\n")


if __name__ == "__main__":
    main()
