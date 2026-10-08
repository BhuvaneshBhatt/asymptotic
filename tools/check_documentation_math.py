"""Check Markdown math delimiters and extract formulas for renderer validation."""

import argparse
import json
import re
from pathlib import Path


def documentation_files(root):
    """List maintained documentation, examples and benchmark explanations."""
    return sorted(
        {
            *root.glob("*.md"),
            *(root / "docs").rglob("*.md"),
            *(root / "examples").rglob("*.md"),
            *(root / "benchmarks" / "docs").rglob("*.md"),
        }
    )


def prose(text):
    """Exclude literal fenced blocks and inline code from math checks."""
    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            continue
        if fence is None:
            result.append(line)
    return re.sub(r"(`+)(.*?)\1", "", "".join(result), flags=re.S)


def check(root):
    """Return formula text and malformed delimiters, excluding literal code."""
    formulas, errors = [], []
    files = documentation_files(root)
    for path in files:
        text = prose(path.read_text())
        name = str(path.relative_to(root))
        if any(delimiter in text for delimiter in (r"\(", r"\)", r"\[", r"\]")):
            errors.append({"file": name, "error": "Use dollar math delimiters."})
        delimiters = list(re.finditer(r"(?<!\\)\${1,2}", text))
        pending = None
        for delimiter in delimiters:
            if pending is None:
                pending = delimiter
                continue
            if delimiter[0] != pending[0]:
                errors.append({"file": name, "error": "Mixed math delimiters."})
                pending = None
                continue
            formula = text[pending.end() : delimiter.start()]
            display = delimiter[0] == "$$"
            if not display and "\n" in formula:
                errors.append({"file": name, "error": "Inline formula crosses a line."})
            if display and "\n\n" in formula:
                errors.append(
                    {"file": name, "error": "Blank paragraph inside display math."}
                )
            formulas.append({"file": name, "display": display, "tex": formula.strip()})
            pending = None
        if pending is not None:
            errors.append({"file": name, "error": "Unclosed math delimiter."})
    return {
        "files": len(files),
        "formulas": len(formulas),
        "errors": errors,
        "formulas_checked": formulas,
    }


def main():
    """Write a reusable formula inventory and fail on malformed delimiters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.root)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(
        f"{report['files']} files, {report['formulas']} formulas, "
        f"{len(report['errors'])} delimiter errors"
    )
    for error in report["errors"]:
        print(f"{error['file']}: {error['error']}")
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
