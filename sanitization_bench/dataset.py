from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


class SanitizationDataset(Sequence[dict[str, Any]]):
    """A small list-like container for normalized sanitization examples.

    This intentionally does not depend on PyTorch, TensorFlow, HuggingFace Datasets,
    or any other ML framework. It is designed to be used as the first step in a
    privacy/sanitization pipeline.
    """

    def __init__(
        self,
        examples: Iterable[dict[str, Any]],
        *,
        name: str,
        split: str,
        variant: str,
        seed: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.examples = list(examples)
        self.name = name
        self.split = split
        self.variant = variant
        self.seed = seed
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.examples)

    def __repr__(self) -> str:
        return (
            f"SanitizationDataset(name={self.name!r}, split={self.split!r}, "
            f"variant={self.variant!r}, size={len(self)}, seed={self.seed})"
        )

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.examples)

    def preview(self, n: int = 3) -> list[dict[str, Any]]:
        return self.examples[: max(0, n)]

    def map(self, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> "SanitizationDataset":
        return SanitizationDataset(
            [fn(x) for x in self.examples],
            name=self.name,
            split=self.split,
            variant=self.variant,
            seed=self.seed,
            metadata=dict(self.metadata),
        )

    def filter(self, fn: Callable[[dict[str, Any]], bool]) -> "SanitizationDataset":
        return SanitizationDataset(
            [x for x in self.examples if fn(x)],
            name=self.name,
            split=self.split,
            variant=self.variant,
            seed=self.seed,
            metadata=dict(self.metadata),
        )

    def to_jsonl(self, path: str | Path, *, ensure_ascii: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in self.examples:
                f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
        return path
