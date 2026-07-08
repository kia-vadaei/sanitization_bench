from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ..download import ensure_files
from ..io import read_json
from ..schema import base_example
from .base import BaseAdapter


class TABAdapter(BaseAdapter):
    name = "tab"
    domain = "legal"
    official_splits = {"train", "dev", "test"}

    URLS = {
        "train": "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/refs/heads/master/echr_train.json",
        "dev": "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/refs/heads/master/echr_dev.json",
        "test": "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/refs/heads/master/echr_test.json",
    }

    def _split_key(self) -> str:
        split = self.split.lower().strip()
        if split == "validation":
            split = "dev"
        if split not in self.URLS:
            raise ValueError("TAB supports split='train', 'dev'/'validation', or 'test'.")
        return split

    def _ensure(self) -> Path:
        split = self._split_key()
        if self.download:
            paths = ensure_files({split: self.URLS[split]}, self.dataset_dir, show_progress=self.show_progress)
            return paths[split]

        path = self.dataset_dir / Path(self.URLS[split]).name
        if not path.exists():
            raise FileNotFoundError(
                f"TAB file not found at {path}. Pass download=True or place the raw file there."
            )
        return path

    @staticmethod
    def _flatten_mentions(annotations: Any) -> list[dict[str, Any]]:
        """Robustly collect TAB entity mentions from possibly nested standoff JSON."""
        mentions: list[dict[str, Any]] = []

        def visit(obj: Any) -> None:
            if isinstance(obj, dict):
                if {
                    "start_offset",
                    "end_offset",
                    "span_text",
                }.issubset(obj.keys()) or "entity_mention_id" in obj:
                    mentions.append(obj)
                    return
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item)

        visit(annotations)
        return mentions

    def _normalize_doc(self, doc: dict[str, Any], split: str) -> dict[str, Any]:
        text = doc.get("text", "") or ""
        doc_id = str(doc.get("doc_id", "unknown-doc"))
        mentions = self._flatten_mentions(doc.get("annotations", {}))

        spans: list[dict[str, Any]] = []
        entity_to_spans: dict[str, list[str]] = defaultdict(list)
        subjects_by_id: dict[str, dict[str, Any]] = {}

        for idx, mention in enumerate(mentions):
            span_id = str(mention.get("entity_mention_id") or f"{doc_id}-span-{idx}")
            entity_id = mention.get("entity_id")
            subject_id = f"entity-{entity_id}" if entity_id is not None else None
            start = mention.get("start_offset")
            end = mention.get("end_offset")
            try:
                start_int = int(start) if start is not None else None
                end_int = int(end) if end is not None else None
            except (TypeError, ValueError):
                start_int = None
                end_int = None

            span_text = mention.get("span_text")
            if span_text is None and start_int is not None and end_int is not None:
                span_text = text[start_int:end_int]

            spans.append(
                {
                    "span_id": span_id,
                    "start": start_int,
                    "end": end_int,
                    "text": span_text or "",
                    "label": mention.get("entity_type"),
                    "identifier_type": mention.get("identifier_type"),
                    "subject_id": subject_id,
                    "replacement": None,
                    "metadata": {
                        "edit_type": mention.get("edit_type"),
                        "confidential_status": mention.get("confidential_status"),
                        "entity_mention_id": mention.get("entity_mention_id"),
                        "entity_id": entity_id,
                    },
                }
            )

            if subject_id:
                entity_to_spans[subject_id].append(span_id)
                if subject_id not in subjects_by_id:
                    subjects_by_id[subject_id] = {
                        "subject_id": subject_id,
                        "description": None,
                        "aliases": [],
                        "role": None,
                        "metadata": {"entity_id": entity_id},
                    }
                if span_text and span_text not in subjects_by_id[subject_id]["aliases"]:
                    subjects_by_id[subject_id]["aliases"].append(span_text)

        relations = []
        for subject_id, span_ids in entity_to_spans.items():
            if len(span_ids) <= 1:
                continue
            anchor = span_ids[0]
            for other in span_ids[1:]:
                relations.append(
                    {
                        "type": "coreference",
                        "source": anchor,
                        "target": other,
                        "metadata": {"subject_id": subject_id},
                    }
                )

        return base_example(
            example_id=doc_id,
            dataset=self.name,
            split=split,
            domain=self.domain,
            text=text,
            units=[
                {
                    "unit_id": f"{doc_id}-document",
                    "type": "document",
                    "text": text,
                    "speaker": None,
                    "start": 0,
                    "end": len(text),
                    "metadata": {},
                }
            ],
            spans=spans,
            subjects=list(subjects_by_id.values()),
            attributes=[],
            relations=relations,
            metadata={
                "document_id": doc_id,
                "dataset_type": doc.get("dataset_type"),
                "task": doc.get("task"),
                "quality_checked": doc.get("quality_checked"),
                "meta": doc.get("meta", {}),
            },
            raw=doc if self.keep_raw else {},
        )

    def load(self) -> list[dict[str, Any]]:
        split = self._split_key()
        path = self._ensure()
        docs = read_json(path)
        if not isinstance(docs, list):
            raise ValueError(f"TAB file must contain a list of documents: {path}")

        examples = []
        for doc in tqdm(docs, desc=f"Parsing TAB/{split}", unit="doc", disable=not self.show_progress):
            examples.append(self._normalize_doc(doc, split))
        return examples
