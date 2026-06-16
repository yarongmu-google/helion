"""Pre-type-prop AST classifier for jagged_tile bodies.

Tags each outer ``for tile_b in hl.tile(...)`` for-node with:

  - ``_jagged_flat = True`` if a nested ``hl.jagged_tile(...)`` body
    has a single-element subscript via ``hl.load(x_flat, [<expr>])``
    or ``hl.store(...)`` — the canonical 2D form on a 1-D source.

  - ``_jagged_outer = True`` if a nested ``hl.jagged_tile(...)`` body
    has ``tensor[<expr>, :, :, ...]`` — multi-element subscript with
    the first slot a real expression and trailing slots bare ``:``
    slices (the natural N-D outer-jagged case at dim_from_end >= 2).

Read by ``type_propagation.visit_For`` to propagate onto the
``TileIndexType`` instance, which ``type_info.py`` reads for the
auto-collapse decision.

Read-only, no type info, no torch ops — pure syntactic AST walk.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_function import HostFunction


class _JaggedClassifier(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._outer_tile: ast.For | None = None
        self._jagged_depth: int = 0

    def visit_For(self, node: ast.For) -> None:
        attr = getattr(getattr(node.iter, "func", None), "attr", None)
        if attr == "tile":
            prev = self._outer_tile
            if prev is None:
                self._outer_tile = node
            self.generic_visit(node)
            self._outer_tile = prev
        elif attr == "jagged_tile":
            self._jagged_depth += 1
            self.generic_visit(node)
            self._jagged_depth -= 1
        else:
            self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._jagged_depth and self._outer_tile is not None:
            sl = node.slice
            if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                first, *rest = sl.elts
                # First slot must be a real expression (Name/BinOp/Attr/Call):
                # excludes ``starts[:, None]`` (first=Slice) and
                # ``mask[None, :, :]`` (first=Constant(None)).
                # Trailing slots must all be bare ``:`` slices:
                # excludes multi-tile dense indexing like
                # ``dense[tile_b, tile_d, tile_k]``.
                if isinstance(
                    first, (ast.Name, ast.BinOp, ast.Attribute, ast.Call)
                ) and all(
                    isinstance(e, ast.Slice)
                    and e.lower is None
                    and e.upper is None
                    and e.step is None
                    for e in rest
                ):
                    self._outer_tile._jagged_outer = True  # type: ignore[attr-defined]
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._jagged_depth and self._outer_tile is not None:
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in ("load", "store")
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.List)
                and len(node.args[1].elts) == 1
            ):
                self._outer_tile._jagged_flat = True  # type: ignore[attr-defined]
        self.generic_visit(node)


def classify(hf: HostFunction) -> None:
    """Run the classifier over the kernel body."""
    classifier = _JaggedClassifier()
    for stmt in hf.body:
        classifier.visit(stmt)
