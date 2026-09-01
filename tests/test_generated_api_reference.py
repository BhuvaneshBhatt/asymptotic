"""The generated API reference must match current signatures/docstrings."""

from enum import Enum
from pathlib import Path

import asymptotic
from tools.generate_api_reference import _public_docstring, _public_signature, render


def test_generated_api_reference_is_current():
    assert Path("docs/api-reference.md").read_text() == render()


def test_enum_signatures_are_python_version_independent():
    assert _public_signature(asymptotic.GrowthComparison) == "(value)"
    assert _public_signature(asymptotic.RemainderKind) == "(value)"


def test_enum_docstrings_do_not_inherit_stdlib_enum_documentation():
    assert _public_docstring(asymptotic.GrowthComparison) == "No public docstring."
    assert _public_docstring(asymptotic.RemainderKind) == (
        "Semantic strength of an asymptotic remainder statement."
    )


def test_enum_placeholder_docstring_is_treated_as_not_public():
    class PlaceholderDocEnum(Enum):
        __doc__ = "An enumeration."
        ITEM = "item"

    assert _public_docstring(PlaceholderDocEnum) == "No public docstring."
