"""Native parsing and comparison contracts for frozen limit references."""

import ast

import sympy as sp

from .conditional import ConditionalExpression


class _PiecewiseSyntax(ast.NodeTransformer):
    """Flatten grouped scalar branches without changing priority or defaults."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "Piecewise":
            return node
        if node.args and isinstance(node.args[0], ast.Tuple):
            first = node.args[0]
            if first.elts and all(
                isinstance(arg, ast.Tuple) and len(arg.elts) == 2 for arg in first.elts
            ):
                node.args = list(first.elts) + node.args[1:]
        if node.args and not (
            isinstance(node.args[-1], ast.Tuple) and len(node.args[-1].elts) == 2
        ):
            node.args[-1] = ast.Tuple(
                elts=[node.args[-1], ast.Constant(value=True)], ctx=ast.Load()
            )
        return node


def parse_reference(text, namespace):
    """Parse native expressions, normalizing grouped Piecewise/default literals."""
    if "Piecewise(" in text:
        tree = _PiecewiseSyntax().visit(ast.parse(text, mode="eval"))
        text = ast.unparse(ast.fix_missing_locations(tree))
    return sp.sympify(text, locals=namespace)


def reference_membership(value, domain):
    """Translate scalar/tuple real or integer membership into native predicates."""
    value, domain = map(sp.sympify, (value, domain))
    if isinstance(value, sp.Tuple):
        return sp.And(*(reference_membership(item, domain) for item in value))
    if domain == sp.S.Reals:
        result = sp.Q.real(value)
        if value.is_Pow and value.exp == -1 and isinstance(value.base, sp.Symbol):
            result = sp.And(result, sp.Q.real(value.base), sp.Ne(value.base, 0))
        return result
    if domain == sp.S.Integers:
        return sp.Q.integer(value)
    return sp.Contains(value, domain)


def conditional_reference(value):
    """Return a reference's value and parameter condition without discarding it."""
    if isinstance(value, ConditionalExpression):
        return value.value, value.condition
    return value, sp.S.true
