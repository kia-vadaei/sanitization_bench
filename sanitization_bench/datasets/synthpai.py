from __future__ import annotations

from collections import defaultdict
from typing import Any

from tqdm.auto import tqdm

from ..download import ensure_files
from ..io import read_jsonl
from ..sampling import deterministic_split
from ..schema import base_example
from .base import BaseAdapter


class SynthPAIAdapter(BaseAdapter):
    name = "synthpai"
    domain = "online_forum"

    URLS = {
        "all": "https://raw.githubusercontent.com/eth-sri/SynthPAI/refs/heads/main/data/synthpai.jsonl"
    }

    def _ensure(self):
        if self.download:
            return ensure_files(self.URLS, self.dataset_dir, show_progress=self.show_progress)["all"]
        path = self.dataset_dir / "synthpai.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"SynthPAI file not found at {path}. Pass download=True or place synthpai.jsonl there."
            )
        return path

    @staticmethod
    def _safe_author(row: dict[str, Any]) -> str:
        return str(row.get("author") or row.get("username") or row.get("profile", {}).get("username") or "unknown")

    def _normalize_comment(self, row: dict[str, Any], split: str) -> dict[str, Any]:
        author = self._safe_author(row)
        comment_id = str(row.get("id") or f"comment-{author}")
        text = row.get("text", "") or ""
        profile = row.get("profile", {}) or {}

        attributes = []
        for key, value in profile.items():
            if key in {"style", "username"}:
                continue
            attributes.append(
                {
                    "subject_id": f"author-{author}",
                    "attribute": key,
                    "value": value,
                    "certainty": None,
                    "hardness": None,
                    "evidence": [comment_id],
                }
            )

        return base_example(
            example_id=comment_id,
            dataset=self.name,
            split=split,
            domain=self.domain,
            text=text,
            units=[
                {
                    "unit_id": comment_id,
                    "type": "comment",
                    "text": text,
                    "speaker": author,
                    "start": 0,
                    "end": len(text),
                    "metadata": {
                        "thread_id": row.get("thread_id"),
                        "parent_id": row.get("parent_id"),
                        "children": row.get("children", []),
                    },
                }
            ],
            spans=[],
            subjects=[
                {
                    "subject_id": f"author-{author}",
                    "description": "The synthetic online author",
                    "aliases": [str(row.get("username") or profile.get("username") or author)],
                    "role": "author",
                    "metadata": {"author": author, "profile_style": profile.get("style")},
                }
            ],
            attributes=attributes,
            relations=[],
            metadata={
                "author_id": author,
                "username": row.get("username") or profile.get("username"),
                "thread_id": row.get("thread_id"),
                "parent_id": row.get("parent_id"),
                "comment_id": comment_id,
                "reviews": row.get("reviews"),
                "guesses": row.get("guesses"),
            },
            raw=row if self.keep_raw else {},
        )

    def _normalize_author(self, author: str, rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
        rows = sorted(rows, key=lambda r: str(r.get("id") or ""))
        profile = rows[0].get("profile", {}) or {}
        username = rows[0].get("username") or profile.get("username") or author

        units = []
        text_parts = []
        cursor = 0
        relations = []

        for idx, row in enumerate(rows):
            comment_id = str(row.get("id") or f"{author}-comment-{idx}")
            comment_text = row.get("text", "") or ""
            header = f"[Comment {idx + 1} | thread={row.get('thread_id', '')}]\n"
            block = header + comment_text
            start = cursor
            end = start + len(block)
            text_parts.append(block)
            units.append(
                {
                    "unit_id": comment_id,
                    "type": "comment",
                    "text": comment_text,
                    "speaker": author,
                    "start": start + len(header),
                    "end": end,
                    "metadata": {
                        "thread_id": row.get("thread_id"),
                        "parent_id": row.get("parent_id"),
                        "children": row.get("children", []),
                    },
                }
            )
            parent_id = row.get("parent_id")
            if parent_id:
                relations.append(
                    {
                        "type": "reply_to",
                        "source": comment_id,
                        "target": str(parent_id),
                        "metadata": {"thread_id": row.get("thread_id")},
                    }
                )
            for child in row.get("children", []) or []:
                relations.append(
                    {
                        "type": "has_child",
                        "source": comment_id,
                        "target": str(child),
                        "metadata": {"thread_id": row.get("thread_id")},
                    }
                )
            cursor = end + 2
            text_parts.append("\n\n")

        full_text = "".join(text_parts).rstrip()

        attributes = []
        evidence = [unit["unit_id"] for unit in units]
        for key, value in profile.items():
            if key in {"style", "username"}:
                continue
            attributes.append(
                {
                    "subject_id": f"author-{author}",
                    "attribute": key,
                    "value": value,
                    "certainty": None,
                    "hardness": None,
                    "evidence": evidence,
                }
            )

        return base_example(
            example_id=f"author-{author}",
            dataset=self.name,
            split=split,
            domain=self.domain,
            text=full_text,
            units=units,
            spans=[],
            subjects=[
                {
                    "subject_id": f"author-{author}",
                    "description": "The synthetic online author",
                    "aliases": [str(username), str(author)],
                    "role": "author",
                    "metadata": {"author": author, "profile_style": profile.get("style")},
                }
            ],
            attributes=attributes,
            relations=relations,
            metadata={
                "author_id": author,
                "username": username,
                "num_comments": len(rows),
                "thread_ids": sorted({str(r.get("thread_id")) for r in rows if r.get("thread_id")}),
            },
            raw={"rows": rows} if self.keep_raw else {},
        )

    def load(self) -> list[dict[str, Any]]:
        path = self._ensure()
        rows = read_jsonl(path)
        granularity = (self.granularity or "author").lower().strip()

        if granularity not in {"author", "comment"}:
            raise ValueError("SynthPAI granularity must be 'author' or 'comment'.")

        if granularity == "comment":
            examples = [
                self._normalize_comment(row, self.split)
                for row in tqdm(rows, desc="Parsing SynthPAI/comments", unit="comment", disable=not self.show_progress)
            ]
        else:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[self._safe_author(row)].append(row)

            examples = [
                self._normalize_author(author, group, self.split)
                for author, group in tqdm(
                    sorted(grouped.items()),
                    desc="Parsing SynthPAI/authors",
                    unit="author",
                    disable=not self.show_progress,
                )
            ]

        # Generated split is safe because examples are already author-level by default.
        return deterministic_split(examples, split=self.split, seed=self.seed)
