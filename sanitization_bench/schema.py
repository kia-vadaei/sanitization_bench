from __future__ import annotations

from typing import Any


def base_example(
    *,
    example_id: str,
    dataset: str,
    split: str,
    domain: str,
    text: str,
    units: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    subjects: list[dict[str, Any]] | None = None,
    attributes: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the canonical normalized example dict."""
    return {
        "id": str(example_id),
        "dataset": dataset,
        "split": split,
        "domain": domain,
        "text": text or "",
        "units": units or [],
        "spans": spans or [],
        "subjects": subjects or [],
        "attributes": attributes or [],
        "relations": relations or [],
        "metadata": metadata or {},
        "raw": raw or {},
    }


def validate_example(example: dict[str, Any]) -> None:
    """Lightweight validation for the canonical schema."""
    required = {
        "id",
        "dataset",
        "split",
        "domain",
        "text",
        "units",
        "spans",
        "subjects",
        "attributes",
        "relations",
        "metadata",
        "raw",
    }
    missing = required.difference(example)
    if missing:
        raise ValueError(f"Normalized example is missing required keys: {sorted(missing)}")
    if not isinstance(example["text"], str):
        raise TypeError("example['text'] must be a string")
    for key in ["units", "spans", "subjects", "attributes", "relations"]:
        if not isinstance(example[key], list):
            raise TypeError(f"example[{key!r}] must be a list")
