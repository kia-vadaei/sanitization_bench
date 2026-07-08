from __future__ import annotations

import random
from typing import Any, Iterable


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


def resolve_percentage(variant: str, percentage: float | None) -> float:
    variant_norm = str(variant).strip().lower()
    if variant_norm == "custom":
        if percentage is None:
            raise ValueError("variant='custom' requires percentage=<float between 0 and 100>.")
        pct = float(percentage)
    elif variant_norm in VARIANT_ALIASES:
        pct = VARIANT_ALIASES[variant_norm]
    else:
        raise ValueError(
            "Unsupported variant. Use one of: '10%', '20%', '100%', 'custom'. "
            f"Received: {variant!r}"
        )

    if not (0 < pct <= 100):
        raise ValueError(f"percentage must be in (0, 100], got {pct}")
    return pct


def deterministic_sample(
    examples: list[dict[str, Any]],
    *,
    variant: str = "100%",
    percentage: float | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    pct = resolve_percentage(variant, percentage)
    if pct >= 100:
        return list(examples)

    n = len(examples)
    if n == 0:
        return []

    sample_size = max(1, int(round(n * pct / 100.0)))
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = set(indices[:sample_size])

    # Preserve original order after deterministic selection.
    return [example for idx, example in enumerate(examples) if idx in selected]


def deterministic_split(
    examples: list[dict[str, Any]],
    *,
    split: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    """Create a deterministic train/dev/test split for datasets without official splits.

    Split is applied at the already-normalized example level. Adapters choose the example
    level carefully: author-level for SynthPAI, document-level for SPIA, etc.
    """
    split_norm = str(split).strip().lower()
    if split_norm in {"all", "full", "100%"}:
        return list(examples)
    if split_norm == "validation":
        split_norm = "dev"
    if split_norm not in {"train", "dev", "test"}:
        raise ValueError("split must be one of 'all', 'train', 'dev'/'validation', or 'test'.")

    n = len(examples)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(round(n * train_ratio))
    dev_end = train_end + int(round(n * dev_ratio))

    if split_norm == "train":
        chosen = set(indices[:train_end])
    elif split_norm == "dev":
        chosen = set(indices[train_end:dev_end])
    else:
        chosen = set(indices[dev_end:])

    return [example for idx, example in enumerate(examples) if idx in chosen]
