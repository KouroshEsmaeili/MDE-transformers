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
    splits = _dataset_splits("train_nyu.py")
    assert splits
    assert set(splits) == {"train"}


def test_evaluation_command_explicitly_constructs_only_official_test_split() -> None:
    assert _dataset_splits("evaluate_nyu.py") == ["test"]


def test_truncated_test_evaluation_is_visibly_labeled() -> None:
    script = (Path(__file__).parents[2] / "scripts" / "evaluate_nyu.py").read_text(encoding="utf-8")
    assert "TRUNCATED_DIAGNOSTIC" in script
    assert "not full-test benchmark results" in script


def test_commands_expose_and_record_explicit_crop_policies() -> None:
    root = Path(__file__).parents[2] / "scripts"
    train_script = (root / "train_nyu.py").read_text(encoding="utf-8")
    evaluation_script = (root / "evaluate_nyu.py").read_text(encoding="utf-8")

    assert '"--training-crop"' in train_script
    assert '"--evaluation-crop"' in train_script
    assert "training_crop=" in train_script
    assert "evaluation_crop=" in train_script
    assert '"--crop"' in evaluation_script
    assert "protocol={protocol.profile_name}" in evaluation_script
