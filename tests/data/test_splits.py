from __future__ import annotations

import pytest

from mde_transformers.data import split_train_dev_indices


def test_train_dev_split_is_seeded_disjoint_and_complete() -> None:
    source = tuple(range(10, 20))

    first = split_train_dev_indices(source, validation_fraction=0.3, seed=17)
    repeated = split_train_dev_indices(source, validation_fraction=0.3, seed=17)
    different = split_train_dev_indices(source, validation_fraction=0.3, seed=18)

    assert first == repeated
    assert first != different
    assert len(first.optimization) == 7
    assert len(first.validation) == 3
    assert set(first.optimization).isdisjoint(first.validation)
    assert set(first.optimization).union(first.validation) == set(source)
    assert first.optimization == tuple(index for index in source if index in first.optimization)
    assert first.validation == tuple(index for index in source if index in first.validation)


def test_train_dev_split_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        split_train_dev_indices([1, 1, 2], validation_fraction=0.5, seed=0)
    with pytest.raises(ValueError, match="strictly between"):
        split_train_dev_indices([1, 2], validation_fraction=0.0, seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        split_train_dev_indices([1, 2], validation_fraction=0.1, seed=0)
