from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseAdapter(ABC):
    """Base class for dataset adapters."""

    name: str
    domain: str
    official_splits: set[str] = set()

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        split: str,
        source: str = "default",
        granularity: str | None = None,
        download: bool = True,
        show_progress: bool = True,
        keep_raw: bool = True,
        seed: int = 42,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.source = source
        self.granularity = granularity
        self.download = download
        self.show_progress = show_progress
        self.keep_raw = keep_raw
        self.seed = seed

    @property
    def dataset_dir(self) -> Path:
        return self.cache_dir / self.name

    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        """Return normalized examples before variant sampling."""
        raise NotImplementedError
