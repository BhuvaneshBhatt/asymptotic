"""Generate the root API reference from live public docstrings."""

from __future__ import annotations

import inspect
import re
from enum import Enum
from pathlib import Path

import asymptotic

HEADER = """# Generated primary API reference\n\nThis file is generated from the live root API signatures and docstrings. Edit the public docstrings, then run `python tools/generate_api_reference.py`.\n\n"""


def _public_signature(obj: object) -> str:
    """Return a Python-version-stable signature for a public object."""

    if isinstance(obj, type) and issubclass(obj, Enum):
        return "(value)"
    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    signature = re.sub(
        r"<asymptotic\.function_properties\.registry\.FunctionPropertyRegistry object at 0x[0-9a-fA-F]+>",
        "DEFAULT_REGISTRY",
        signature,
    )
    return re.sub(r" at 0x[0-9a-fA-F]+", "", signature)


def _public_docstring(obj: object) -> str:
    """Return a Python-version-stable public docstring."""

    if isinstance(obj, type) and issubclass(obj, Enum):
        doc = obj.__dict__.get("__doc__")
        if not doc:
            return "No public docstring."
        return inspect.cleandoc(doc)
    wrapped = getattr(obj, "__wrapped__", None)
    original_doc = getattr(wrapped, "__doc__", None)
    if original_doc and obj.__doc__.startswith(original_doc):
        # Decorators can append unindented API notes to an indented docstring.
        # Clean the original and supplement separately: inspect.cleandoc differs
        # across Python versions when both indentation styles occur together.
        supplement = inspect.cleandoc(obj.__doc__[len(original_doc) :])
        return inspect.cleandoc(original_doc) + (
            "\n\n" + supplement if supplement else ""
        )
    return inspect.getdoc(obj) or "No public docstring."


def render() -> str:
    """Render the current root API as deterministic Markdown."""

    parts = [HEADER]
    for name in sorted(asymptotic.__all__):
        obj = getattr(asymptotic, name)
        parts.append(f"## `{name}`\n\n")
        if name == "__version__":
            parts.append("Current package version string.\n\n")
            continue
        signature = _public_signature(obj)
        parts.append(f"```python\n{name}{signature}\n```\n\n")
        parts.append(_public_docstring(obj) + "\n\n")
    return "".join(parts)


def main() -> None:
    """Regenerate ``docs/api-reference.md`` from the imported package."""

    (Path(__file__).resolve().parents[1] / "docs/api-reference.md").write_text(render())


if __name__ == "__main__":
    main()
