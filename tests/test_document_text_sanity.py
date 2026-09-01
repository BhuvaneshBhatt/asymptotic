from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_contains_no_ascii_control_characters():
    files = [ROOT / "README.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    files.extend(sorted((ROOT / "docs").rglob("*.rst")))
    failures = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for offset, character in enumerate(text):
            if ord(character) < 32 and character != "\n":
                failures.append((path.relative_to(ROOT), offset, ord(character)))
    assert not failures, f"unexpected control characters: {failures[:10]}"
