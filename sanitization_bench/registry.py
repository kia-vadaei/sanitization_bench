from __future__ import annotations

from typing import Type

from .datasets.base import BaseAdapter
from .datasets.qatd2k import QATD2KAdapter
from .datasets.spia import SPIAAdapter
from .datasets.synthpai import SynthPAIAdapter
from .datasets.tab import TABAdapter

DATASET_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "tab": TABAdapter,
    "synthpai": SynthPAIAdapter,
    "spia": SPIAAdapter,
    "qatd2k": QATD2KAdapter,
    "qatd-2k": QATD2KAdapter,
    "qatd_2k": QATD2KAdapter,
}


def get_adapter_class(name: str) -> Type[BaseAdapter]:
    key = str(name).strip().lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")
    return DATASET_REGISTRY[key]


def list_dataset_names() -> list[str]:
    return sorted({"tab", "synthpai", "spia", "qatd2k"})
