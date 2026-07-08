from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .dataset import SanitizationDataset
from .registry import get_adapter_class, list_dataset_names
from .sampling import deterministic_sample
from .schema import validate_example


def list_datasets() -> list[str]:
    return list_dataset_names()


def load_dataset(
    name: str,
    *,
    split: str = "train",
    variant: str = "100%",
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
    """Load a normalized sanitization benchmark dataset.

    Parameters
    ----------
    name:
        One of: 'tab', 'synthpai', 'spia', 'qatd2k'.
    split:
        Official split if available, otherwise deterministic split from normalized examples.
    variant:
        One of: '10%', '20%', '100%', 'custom'. Aliases such as 'full' and '10pct' also work.
    percentage:
        Required when variant='custom'.
    seed:
        Fixed seed for deterministic generated splits and variant sampling.
    cache_dir:
        Directory where raw downloaded files are cached.
    source:
        Dataset-specific source. For SPIA: 'panorama_151', 'panorama_531', 'tab_144', or 'all'.
    granularity:
        Dataset-specific example granularity. Defaults preserve multi-hop structure.
    download:
        If True, download missing files from their public sources.
    show_progress:
        If True, show tqdm progress bars.
    keep_raw:
        If True, include original raw records in example['raw'].
    validate:
        If True, run lightweight schema validation.
    """
    adapter_cls = get_adapter_class(name)
    adapter = adapter_cls(
        cache_dir=cache_dir,
        split=split,
        source=source,
        granularity=granularity,
        download=download,
        show_progress=show_progress,
        keep_raw=keep_raw,
        seed=seed,
    )

    examples = adapter.load()

    if validate:
        for example in tqdm(
            examples,
            desc=f"Validating {adapter.name}",
            unit="example",
            disable=not show_progress,
        ):
            validate_example(example)

    sampled = deterministic_sample(
        examples,
        variant=variant,
        percentage=percentage,
        seed=seed,
    )

    metadata: dict[str, Any] = {
        "source": source,
        "granularity": granularity,
        "cache_dir": str(Path(cache_dir).resolve()),
        "original_size_before_sampling": len(examples),
    }

    return SanitizationDataset(
        sampled,
        name=adapter.name,
        split=split,
        variant=variant,
        seed=seed,
        metadata=metadata,
    )
