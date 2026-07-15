import pytest

from sanitization_bench.sampling import (
    deterministic_sample,
    resolve_percentage,
)


def test_empty_variant_defaults_to_full():
    assert resolve_percentage("", None) == 100.0


def test_empty_variant_accepts_arbitrary_percentage():
    assert resolve_percentage("", 35) == 35.0
    assert resolve_percentage("", 12.5) == 12.5


def test_percentage_can_be_passed_as_variant():
    assert resolve_percentage("35%", None) == 35.0
    assert resolve_percentage("35", None) == 35.0
    assert resolve_percentage("0.35", None) == 35.0


def test_existing_aliases_still_work():
    assert resolve_percentage("10%", None) == 10.0
    assert resolve_percentage("20pct", None) == 20.0
    assert resolve_percentage("full", None) == 100.0
    assert resolve_percentage("custom", 33) == 33.0


def test_invalid_percentage_is_rejected():
    with pytest.raises(ValueError):
        resolve_percentage("", 0)

    with pytest.raises(ValueError):
        resolve_percentage("", 101)


def test_deterministic_sample():
    examples = [{"id": str(i)} for i in range(100)]

    first = deterministic_sample(
        examples,
        variant="",
        percentage=35,
        seed=42,
    )
    second = deterministic_sample(
        examples,
        variant="",
        percentage=35,
        seed=42,
    )

    assert first == second
    assert len(first) == 35


def test_sampling_is_applied_to_selected_split_size():
    train_examples = [{"id": f"train-{i}"} for i in range(80)]
    test_examples = [{"id": f"test-{i}"} for i in range(20)]

    sampled_train = deterministic_sample(
        train_examples,
        variant="",
        percentage=25,
        seed=42,
    )
    sampled_test = deterministic_sample(
        test_examples,
        variant="",
        percentage=25,
        seed=42,
    )

    assert len(sampled_train) == 20
    assert len(sampled_test) == 5
