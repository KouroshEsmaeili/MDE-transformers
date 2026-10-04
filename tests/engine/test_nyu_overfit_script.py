"""Static safety checks for the deliberately narrow real-NYU diagnostic."""

from __future__ import annotations

import ast
from pathlib import Path


def test_overfit_script_constructs_only_the_official_training_split() -> None:
    script_path = Path(__file__).parents[2] / "scripts" / "overfit_nyu.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    dataset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NYUDepthV2"
    ]

    assert len(dataset_calls) == 1
    split_keywords = [keyword for keyword in dataset_calls[0].keywords if keyword.arg == "split"]
    assert len(split_keywords) == 1
    assert isinstance(split_keywords[0].value, ast.Constant)
    assert split_keywords[0].value.value == "train"
