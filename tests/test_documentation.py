"""Executable introductory documentation and maintained link contracts."""

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_examples():
    namespace = {"__name__": "documentation_example"}
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    assert blocks
    for index, code in enumerate(blocks):
        exec(compile(code, f"README example {index}", "exec"), namespace)


@pytest.mark.parametrize(
    "name",
    [
        "certified_limits",
        "branch_limits",
        "special_function_limits",
        "ordinary_expansion",
        "common_algebra",
    ],
)
def test_introductory_examples(name):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / f"{name}.py")],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_documentation_links():
    files = [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "examples").rglob("*.md"),
    ]
    failures = []
    for path in files:
        for link in re.findall(r"(?<![!\\\w])\[[^]]*\]\(([^)]+)\)", path.read_text()):
            target = link.split()[0].strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            if not destination.exists():
                failures.append(f"{path.relative_to(ROOT)}: {target}")
    assert failures == []


def test_readme_repository_links():
    links = re.findall(r"\[[^]]*\]\(([^)]+)\)", (ROOT / "README.md").read_text())
    assert links
    assert all(
        link.startswith("https://github.com/BhuvaneshBhatt/asymptotic/")
        for link in links
    )
