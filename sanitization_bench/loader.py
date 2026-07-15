from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .dataset import SanitizationDataset
from .registry import get_adapter_class, list_dataset_names
from .sampling import deterministic_sample, resolve_percentage
from .schema import validate_example


def list_datasets() -> list[str]:
    """Return the canonical names of all registered datasets."""
    return list_dataset_names()


def _format_percentage(percentage: float) -> str:
    """Convert a percentage to a canonical variant label."""
    return f"{percentage:g}%"


def load_dataset(
    name: str,
    *,
    split: str = "train",
    variant: str = "",
    percentage: float | None = None,
    seed: int = 42,
    cache_dir: str | Path = "./data",
    source: str = "default",
    granularity: str | None = None,
    download: bool = True,
    show_progress: bool = True,
    keep_raw: bool = True,
    validate: bool = True,
) -> SanitizationDataset:
    """Load one normalized sanitization benchmark dataset.

    Sampling is applied after the requested split is loaded. Therefore:

    - split="train", percentage=35 returns 35% of train.
    - split="test", percentage=35 returns 35% of test.

    Parameters
    ----------
    name:
        Registered dataset name. Supported canonical names are:
        ``tab``, ``synthpai``, ``spia``, and ``qatd2k``.

    split:
        Requested dataset split. Official splits are used when available.
        Otherwise, the adapter creates a deterministic split.

    variant:
        Optional sampling alias.

        Preferred usage is::

            variant=""
            percentage=35

        Percentage aliases such as ``"35%"`` and ``"full"`` are also
        supported. ``"custom"`` remains supported for backward compatibility.

    percentage:
        Percentage of the selected split to retain. Must be greater than zero
        and less than or equal to 100.

        When ``variant=""`` and ``percentage=None``, the full selected split
        is returned.

    seed:
        Seed used for deterministic generated splits and percentage sampling.

    cache_dir:
        Directory containing or receiving the raw dataset files.

    source:
        Dataset-specific source selector. For SPIA, supported sources include
        ``panorama_151``, ``panorama_531``, ``tab_144``, and ``all``.

    granularity:
        Dataset-specific example granularity.

    download:
        Download missing public source files when enabled.

    show_progress:
        Show loading and validation progress bars when enabled.

    keep_raw:
        Preserve the original source record in ``example["raw"]``.

    validate:
        Validate every returned example against the canonical schema.

    Returns
    -------
    SanitizationDataset
        The normalized and deterministically sampled dataset.
    """
    normalized_name = str(name).strip()
    normalized_split = str(split).strip().lower()
    normalized_variant = str(variant or "").strip()

    if not normalized_name:
        raise ValueError("name must not be empty.")

    if not normalized_split:
        raise ValueError("split must not be empty.")

    if normalized_split == "validation":
        normalized_split = "dev"

    # Resolve all accepted sampling configurations into one percentage.
    resolved_percentage = resolve_percentage(
        variant=normalized_variant,
        percentage=percentage,
    )

    # The stored variant is always explicit, even when the caller used "".
    effective_variant = _format_percentage(resolved_percentage)

    resolved_cache_dir = Path(cache_dir).expanduser().resolve()

    adapter_cls = get_adapter_class(normalized_name)

    adapter = adapter_cls(
        cache_dir=resolved_cache_dir,
        split=normalized_split,
        source=source,
        granularity=granularity,
        download=download,
        show_progress=show_progress,
        keep_raw=keep_raw,
        seed=seed,
    )

    # The adapter first selects or creates the requested train/test/dev split.
    examples = list(adapter.load())
    original_size = len(examples)

    # Sampling is then applied only to that selected split.
    sampled_examples = deterministic_sample(
        examples,
        variant="",
        percentage=resolved_percentage,
        seed=seed,
    )

    if validate:
        for example in tqdm(
            sampled_examples,
            desc=f"Validating {adapter.name}/{normalized_split}",
            unit="example",
            disable=not show_progress,
        ):
            validate_example(example)

    metadata: dict[str, Any] = {
        "source": source,
        "granularity": granularity,
        "cache_dir": str(resolved_cache_dir),
        "requested_variant": normalized_variant,
        "requested_percentage": percentage,
        "effective_variant": effective_variant,
        "resolved_percentage": resolved_percentage,
        "original_size_before_sampling": original_size,
        "sampled_size": len(sampled_examples),
        "sampling_seed": seed,
        "sampling_scope": normalized_split,
    }

    return SanitizationDataset(
        sampled_examples,
        name=adapter.name,
        split=normalized_split,
        variant=effective_variant,
        seed=seed,
        metadata=metadata,
    )