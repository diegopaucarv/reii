"""Fast tests for the --mode CLI parsing in main_workflow.py.

Importing main_workflow pulls in torch/stanza/spacy/DB (~5 min), so we
extract the _parse_cli_mode function via AST and exec it in a minimal
namespace. This keeps the test fast and focused on the CLI logic.
"""

import ast
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "reii" / "main_workflow.py"


def _extract_parse_cli_mode() -> tuple:
    """Return (func, valid_modes) extracted from main_workflow.py via AST."""
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    func_src = None
    modes_src = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_cli_mode":
            func_src = ast.get_source_segment(_SRC.read_text(encoding="utf-8"), node)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CLI_MODES_VALIDOS"
            for t in node.targets
        ):
            modes_src = ast.get_source_segment(_SRC.read_text(encoding="utf-8"), node)
    assert func_src is not None, "_parse_cli_mode not found"
    assert modes_src is not None, "_CLI_MODES_VALIDOS not found"

    ns: dict = {"sys": sys, "Optional": Optional}
    exec(modes_src, ns)
    exec(func_src, ns)
    return ns["_parse_cli_mode"], ns["_CLI_MODES_VALIDOS"]


# Import typing.Optional for the namespace
from typing import Optional  # noqa: E402

_parse_cli_mode, _CLI_MODES_VALIDOS = _extract_parse_cli_mode()


def test_valid_modes_include_canonical_and_aliases():
    assert {
        "wc_only",
        "coref_only",
        "emb_only",
        "wc_coref",
        "wc_emb",
        "coref_emb",
        "all",
        "both",
        "sim_only",
        "wc_sim",
        "sim_emb",
    } <= _CLI_MODES_VALIDOS


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["main_workflow.py"], None),
        (["main_workflow.py", "--force"], None),
        (["main_workflow.py", "--mode", "wc_only"], "wc_only"),
        (["main_workflow.py", "--mode", "emb_only"], "emb_only"),
        (["main_workflow.py", "--mode", "all"], "all"),
        (["main_workflow.py", "--mode", "ALL"], "all"),  # case-insensitive
        (["main_workflow.py", "--mode", "both"], "both"),  # alias
        (["main_workflow.py", "--mode", "sim_emb"], "sim_emb"),  # alias
        (["main_workflow.py", "--force", "--mode", "wc_only"], "wc_only"),
    ],
)
def test_parse_cli_mode_valid(monkeypatch, argv, expected):
    monkeypatch.setattr(sys, "argv", argv)
    assert _parse_cli_mode() == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["main_workflow.py", "--mode"],  # missing value
        ["main_workflow.py", "--mode", "--force"],  # value is another flag
        ["main_workflow.py", "--mode", "emb"],  # invalid (emb alone is emb_only)
        ["main_workflow.py", "--mode", "bogus"],
    ],
)
def test_parse_cli_mode_invalid_exits(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        _parse_cli_mode()
    assert exc.value.code == 2
