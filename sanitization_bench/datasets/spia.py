from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from ..download import ensure_files
from ..io import read_jsonl
from ..sampling import deterministic_split
from ..schema import base_example
from .base import BaseAdapter


class SPIAAdapter(BaseAdapter):
    name = "spia"
    domain = "mixed_privacy_inference"

    URLS = {
        "panorama_151": "https://raw.githubusercontent.com/maisonOP/spia/refs/heads/main/data/spia/spia_panorama_151.jsonl",
        "panorama_531": "https://raw.githubusercontent.com/maisonOP/spia/refs/heads/main/data/spia/spia_panorama_531.jsonl",
        "tab_144": "https://raw.githubusercontent.com/maisonOP/spia/refs/heads/main/data/spia/spia_tab_144.jsonl",
    }

    SOURCE_ALIASES = {
        "default": "all",
        "all": "all",
        "panorama151": "panorama_151",
        "panorama_151": "panorama_151",
        "panorama531": "panorama_531",
        "panorama_531": "panorama_531",
        "tab144": "tab_144",
        "tab_144": "tab_144",
        "tab": "tab_144",
    }

    def _source_keys(self) -> list[str]:
        key = self.SOURCE_ALIASES.get(self.source.lower().strip())
        if key is None:
            raise ValueError(
                "SPIA source must be one of: 'all', 'panorama_151', 'panorama_531', 'tab_144'."
            )
        if key == "all":
            return list(self.URLS)
        return [key]

    def _ensure(self):
        keys = self._source_keys()
        urls = {key: self.URLS[key] for key in keys}
        if self.download:
            return ensure_files(urls, self.dataset_dir, show_progress=self.show_progress)

        paths = {}
        for key, url in urls.items():
            path = self.dataset_dir / url.rsplit("/", 1)[-1]
            if not path.exists():
                raise FileNotFoundError(
                    f"SPIA file not found at {path}. Pass download=True or place the raw file there."
                )
            paths[key] = path
        return paths

    @staticmethod
    def _find_keyword_span(text: str, keyword: Any) -> tuple[int | None, int | None]:
        if keyword is None:
            return None, None
        needle = str(keyword)
        if not needle:
            return None, None
        idx = text.find(needle)
        if idx < 0:
            return None, None
        return idx, idx + len(needle)

    def _normalize_doc(self, row: dict[str, Any], split: str, source_key: str) -> dict[str, Any]:
        metadata = row.get("metadata", {}) or {}
        data_id = str(metadata.get("data_id") or row.get("id") or f"{source_key}-unknown")
        text = row.get("text", "") or ""

        subjects = []
        attributes = []
        spans = []
        relations = []

        for subject in row.get("subjects", []) or []:
            sid_raw = subject.get("id")
            subject_id = f"subject-{sid_raw}"
            subjects.append(
                {
                    "subject_id": subject_id,
                    "description": subject.get("description"),
                    "aliases": [],
                    "role": None,
                    "metadata": {"source_subject_id": sid_raw},
                }
            )

            for pii_idx, pii in enumerate(subject.get("PIIs", []) or []):
                tag = pii.get("tag")
                keyword = pii.get("keyword")
                certainty = pii.get("certainty")
                hardness = pii.get("hardness")
                start, end = self._find_keyword_span(text, keyword)
                span_id = f"{data_id}-subject-{sid_raw}-pii-{pii_idx}"
                evidence = []

                if start is not None and end is not None:
                    evidence = [span_id]
                    spans.append(
                        {
                            "span_id": span_id,
                            "start": start,
                            "end": end,
                            "text": text[start:end],
                            "label": tag,
                            "identifier_type": None,
                            "subject_id": subject_id,
                            "replacement": None,
                            "metadata": {
                                "certainty": certainty,
                                "hardness": hardness,
                                "source": "exact_keyword_match",
                            },
                        }
                    )

                attributes.append(
                    {
                        "subject_id": subject_id,
                        "attribute": tag,
                        "value": keyword,
                        "certainty": certainty,
                        "hardness": hardness,
                        "evidence": evidence,
                    }
                )

        # Multi-subject co-presence relations are useful for privacy leakage analysis.
        for i, source in enumerate(subjects):
            for target in subjects[i + 1 :]:
                relations.append(
                    {
                        "type": "co_occurs_in_document",
                        "source": source["subject_id"],
                        "target": target["subject_id"],
                        "metadata": {"document_id": data_id},
                    }
                )

        return base_example(
            example_id=data_id,
            dataset=self.name,
            split=split,
            domain=self.domain,
            text=text,
            units=[
                {
                    "unit_id": f"{data_id}-document",
                    "type": "document",
                    "text": text,
                    "speaker": None,
                    "start": 0,
                    "end": len(text),
                    "metadata": {"source": source_key},
                }
            ],
            spans=spans,
            subjects=subjects,
            attributes=attributes,
            relations=relations,
            metadata={**metadata, "document_id": data_id, "source": source_key},
            raw=row if self.keep_raw else {},
        )

    def load(self) -> list[dict[str, Any]]:
        paths = self._ensure()
        examples = []
        for source_key, path in paths.items():
            rows = read_jsonl(path)
            for row in tqdm(
                rows,
                desc=f"Parsing SPIA/{source_key}",
                unit="doc",
                disable=not self.show_progress,
            ):
                examples.append(self._normalize_doc(row, self.split, source_key))

        return deterministic_split(examples, split=self.split, seed=self.seed)
