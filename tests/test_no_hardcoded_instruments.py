"""Tickers and weights must live in YAML, not in Python source."""

from __future__ import annotations

import ast
from pathlib import Path

from src.paths import PROJECT_ROOT

SRC_DIR = PROJECT_ROOT / "src"
# Example tickers from the commented YAML template — must not appear as string literals in src/.
FORBIDDEN_TICKERS = {"IEF", "LQD", "EMB", "HYG", "FXY", "GLD", "AAXJ"}


def _string_constants(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def test_src_has_no_example_ticker_literals() -> None:
    offenders: list[str] = []
    for path in SRC_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        literals = _string_constants(tree)
        hits = sorted(FORBIDDEN_TICKERS & literals)
        if hits:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {hits}")
    assert not offenders, "Instrument tickers must not be hardcoded:\n" + "\n".join(offenders)
