from __future__ import annotations

import random
from typing import Any


VARIANT_ALIASES = {
    "10%": 10.0,
    "10pct": 10.0,
    "sample_10": 10.0,
    "0.1": 10.0,
    "20%": 20.0,
    "20pct": 20.0,
    "sample_20": 20.0,
    "0.2": 20.0,
    "100%": 100.0,
    "full": 100.0,
    "all": 100.0,
    "1.0": 100.0,
}


def _parse_variant_percentage(variant: str) -> float:
    """Parse arbitrary variants such as '35%', '35', or '0.35'."""
    value = variant.strip().lower()

    if value.endswith("%"):
        value = value[:-1].strip()

    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise ValueError(
            "Unsupported variant. Use an empty variant with percentage=<number>, "
            "a percentage such as '35%', or one of the predefined aliases."
        ) from exc

    # Variant fractions such as 0.35 mean 35%.
    if 0 < numeric_value <= 1:
        numeric_value *= 100

    return numeric_value


def resolve_percentage(
    variant: str = "",
    percentage: float | None = None,
) -> float:
    """Resolve the requested dataset percentage.

    Supported forms:

    variant="", percentage=35
    variant=""
    variant="35%"
    variant="35"
    variant="0.35"
    variant="custom", percentage=35
    variant="full"
    """
    variant_norm = str(variant or "").strip().lower()

    if variant_norm == "":
        # Empty variant means use the explicitly supplied percentage.
        # When no percentage is provided, load the full split.
        pct = 100.0 if percentage is None else float(percentage)

    elif variant_norm == "custom":
        if percentage is None:
            raise ValueError(
                "variant='custom' requires percentage=<float between 0 and 100>."
            )
        pct = float(percentage)

    elif variant_norm in VARIANT_ALIASES:
        if percentage is not None:
            raise ValueError(
                "Do not provide percentage when variant already defines a percentage. "
                "Use variant='' with percentage=<number> instead."
            )
        pct = VARIANT_ALIASES[variant_norm]

    else:
        if percentage is not None:
            raise ValueError(
                "Use either variant='<percentage>' or "
                "variant='' with percentage=<number>, not both."
            )
        pct = _parse_variant_percentage(variant_norm)

    if not (0 < pct <= 100):
        raise ValueError(f"percentage must be in (0, 100], got {pct}")

    return pct


def deterministic_sample(
    examples: list[dict[str, Any]],
    *,
    variant: str = "",
    percentage: float | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return a deterministic percentage of the selected split."""
    pct = resolve_percentage(variant, percentage)

    if pct >= 100:
        return list(examples)

    total = len(examples)
    if total == 0:
        return []

    sample_size = max(1, int(round(total * pct / 100.0)))

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    selected_indices = set(indices[:sample_size])

    # Preserve the original dataset order.
    return [
        example
        for index, example in enumerate(examples)
        if index in selected_indices
    ]


def deterministic_split(
    examples: list[dict[str, Any]],
    *,
    split: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    """Create deterministic train/dev/test splits for datasets without official splits."""
    split_norm = str(split).strip().lower()

    if split_norm in {"all", "full", "100%"}:
        return list(examples)

    if split_norm == "validation":
        split_norm = "dev"

    if split_norm not in {"train", "dev", "test"}:
        raise ValueError(
            "split must be one of 'all', 'train', 'dev'/'validation', or 'test'."
        )

    total = len(examples)
    indices = list(range(total))

    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(round(total * train_ratio))
    dev_end = train_end + int(round(total * dev_ratio))

    if split_norm == "train":
        chosen_indices = set(indices[:train_end])
    elif split_norm == "dev":
        chosen_indices = set(indices[train_end:dev_end])
    else:
        chosen_indices = set(indices[dev_end:])

    return [
        example
        for index, example in enumerate(examples)
        if index in chosen_indices
    ]
