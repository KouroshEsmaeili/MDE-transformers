"""Static split-boundary and diagnostic-label safeguards for NYU commands."""

from __future__ import annotations

import ast
from pathlib import Path


def _dataset_splits(script_name: str) -> list[str]:
    script_path = Path(__file__).parents[2] / "scripts" / script_name
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "NYUDepthV2"
        ):
            continue
        split_keywords = [keyword for keyword in node.keywords if keyword.arg == "split"]
        assert len(split_keywords) == 1
        value = split_keywords[0].value
        assert isinstance(value, ast.Constant) and isinstance(value.value, str)
        values.append(value.value)
    return values


def test_training_command_can_construct_only_official_train_split() -> None:
    assert _dataset_splits("train_nyu.py") == ["train"]


def test_evaluation_command_explicitly_constructs_only_official_test_split() -> None:
    assert _dataset_splits("evaluate_nyu.py") == ["test"]


def test_truncated_test_evaluation_is_visibly_labeled() -> None:
    script = (Path(__file__).parents[2] / "scripts" / "evaluate_nyu.py").read_text(encoding="utf-8")
    assert "TRUNCATED_DIAGNOSTIC" in script
    assert "not full-test benchmark results" in script
