from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ..download import ensure_files
from ..io import read_csv
from ..schema import base_example
from .base import BaseAdapter


class QATD2KAdapter(BaseAdapter):
    name = "qatd2k"
    domain = "educational_dialogue"
    official_splits = {"train", "test"}

    URLS = {
        "train": "https://raw.githubusercontent.com/Eedi/qatd-2k-anonymous-clone/refs/heads/main/anchored-dialogues/train.csv",
        "test": "https://raw.githubusercontent.com/Eedi/qatd-2k-anonymous-clone/refs/heads/main/anchored-dialogues/test.csv",
        "dialogue_subjects": "https://raw.githubusercontent.com/Eedi/qatd-2k-anonymous-clone/refs/heads/main/dialogue-subjects.csv",
        "question_metadata": "https://raw.githubusercontent.com/Eedi/qatd-2k-anonymous-clone/refs/heads/main/dq-question-metadata.csv",
    }

    def _split_key(self) -> str:
        split = self.split.lower().strip()
        if split == "validation":
            split = "dev"
        if split == "dev":
            # QATD-2k public clone exposes train/test. We create dev from train later if needed.
            return "train"
        if split not in {"train", "test"}:
            raise ValueError("QATD-2k supports split='train', 'dev'/'validation', or 'test'.")
        return split

    def _ensure(self) -> dict[str, Path]:
        split_key = self._split_key()
        needed = {
            split_key: self.URLS[split_key],
            "dialogue_subjects": self.URLS["dialogue_subjects"],
            "question_metadata": self.URLS["question_metadata"],
        }
        if self.download:
            return ensure_files(needed, self.dataset_dir, show_progress=self.show_progress)

        paths = {}
        for key, url in needed.items():
            path = self.dataset_dir / url.rsplit("/", 1)[-1]
            if not path.exists():
                raise FileNotFoundError(
                    f"QATD-2k file not found at {path}. Pass download=True or place the raw file there."
                )
            paths[key] = path
        return paths

    @staticmethod
    def _row_key(row: dict[str, str], granularity: str) -> str:
        intervention = row.get("InterventionId", "unknown")
        if granularity == "question_dialogue":
            question = row.get("QuestionId_DQ", "unknown")
            tutor = row.get("TutorId", "unknown")
            return f"intervention-{intervention}-question-{question}-tutor-{tutor}"
        return f"intervention-{intervention}"

    @staticmethod
    def _sort_messages(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        def key(row: dict[str, str]) -> tuple[int, str]:
            try:
                seq = int(float(row.get("MessageSequence", "0") or 0))
            except ValueError:
                seq = 0
            return seq, str(row.get("MessageString", ""))

        return sorted(rows, key=key)

    @staticmethod
    def _speaker(row: dict[str, str]) -> str:
        is_tutor = str(row.get("IsTutor", "")).lower().strip()
        if is_tutor in {"true", "1", "yes", "y"}:
            return "tutor"
        if is_tutor in {"false", "0", "no", "n"}:
            return "student"
        return "unknown"

    @staticmethod
    def _index_by_intervention(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        index: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            index[str(row.get("InterventionId", "unknown"))].append(row)
        return index

    @staticmethod
    def _index_questions(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        by_intervention: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_intervention[str(row.get("InterventionId", "unknown"))].append(row)
        return by_intervention

    def _normalize_message(self, row: dict[str, str], split: str) -> dict[str, Any]:
        msg = row.get("MessageString", "") or ""
        intervention = row.get("InterventionId", "unknown")
        sequence = row.get("MessageSequence", "unknown")
        message_id = f"intervention-{intervention}-message-{sequence}"
        speaker = self._speaker(row)
        return base_example(
            example_id=message_id,
            dataset=self.name,
            split=split,
            domain=self.domain,
            text=msg,
            units=[
                {
                    "unit_id": message_id,
                    "type": "message",
                    "text": msg,
                    "speaker": speaker,
                    "start": 0,
                    "end": len(msg),
                    "metadata": {
                        "message_sequence": sequence,
                        "talk_move_prediction": row.get("TalkMovePrediction"),
                    },
                }
            ],
            spans=[],
            subjects=[],
            attributes=[],
            relations=[],
            metadata={
                "intervention_id": intervention,
                "tutor_id": row.get("TutorId"),
                "question_id": row.get("QuestionId_DQ"),
                "message_sequence": sequence,
                "is_tutor": row.get("IsTutor"),
            },
            raw=row if self.keep_raw else {},
        )

    def _normalize_dialogue(
        self,
        group_key: str,
        rows: list[dict[str, str]],
        split: str,
        subjects_by_intervention: dict[str, list[dict[str, str]]],
        questions_by_intervention: dict[str, list[dict[str, str]]],
    ) -> dict[str, Any]:
        rows = self._sort_messages(rows)
        first = rows[0] if rows else {}
        intervention = str(first.get("InterventionId", "unknown"))
        question_ids = sorted({str(r.get("QuestionId_DQ")) for r in rows if r.get("QuestionId_DQ")})
        tutor_ids = sorted({str(r.get("TutorId")) for r in rows if r.get("TutorId")})

        units = []
        relations = []
        parts = []
        cursor = 0

        previous_unit_id = None
        for idx, row in enumerate(rows):
            speaker = self._speaker(row)
            sequence = row.get("MessageSequence", idx)
            unit_id = f"{group_key}-msg-{sequence}"
            msg = row.get("MessageString", "") or ""
            prefix = f"{speaker}: "
            block = prefix + msg
            start = cursor + len(prefix)
            end = cursor + len(block)
            parts.append(block)
            parts.append("\n")
            units.append(
                {
                    "unit_id": unit_id,
                    "type": "message",
                    "text": msg,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "metadata": {
                        "message_sequence": sequence,
                        "talk_move_prediction": row.get("TalkMovePrediction"),
                        "question_id": row.get("QuestionId_DQ"),
                        "tutor_id": row.get("TutorId"),
                    },
                }
            )
            if previous_unit_id is not None:
                relations.append(
                    {
                        "type": "next_turn",
                        "source": previous_unit_id,
                        "target": unit_id,
                        "metadata": {},
                    }
                )
            previous_unit_id = unit_id
            cursor += len(block) + 1

        full_text = "".join(parts).rstrip()

        subject_rows = subjects_by_intervention.get(intervention, [])
        question_rows = questions_by_intervention.get(intervention, [])

        return base_example(
            example_id=group_key,
            dataset=self.name,
            split=split,
            domain=self.domain,
            text=full_text,
            units=units,
            spans=[],
            subjects=[],
            attributes=[],
            relations=relations,
            metadata={
                "intervention_id": intervention,
                "question_ids": question_ids,
                "tutor_ids": tutor_ids,
                "num_messages": len(rows),
                "dialogue_subjects": subject_rows,
                "question_metadata": question_rows,
                "note": "Public QATD-2k anonymous clone is already anonymized; use this mainly for dialogue utility/coherence evaluation.",
            },
            raw={"messages": rows, "dialogue_subjects": subject_rows, "question_metadata": question_rows}
            if self.keep_raw
            else {},
        )

    def load(self) -> list[dict[str, Any]]:
        split_key = self._split_key()
        paths = self._ensure()

        dialogue_rows = read_csv(paths[split_key])
        subject_rows = read_csv(paths["dialogue_subjects"])
        question_rows = read_csv(paths["question_metadata"])

        subjects_by_intervention = self._index_by_intervention(subject_rows)
        questions_by_intervention = self._index_questions(question_rows)

        granularity = (self.granularity or "dialogue").lower().strip()
        if granularity not in {"dialogue", "question_dialogue", "message"}:
            raise ValueError(
                "QATD-2k granularity must be 'dialogue', 'question_dialogue', or 'message'."
            )

        # Public repo has train/test. For dev, make a deterministic dev slice from train.
        normalized_split = self.split.lower().strip()
        if normalized_split == "validation":
            normalized_split = "dev"

        if granularity == "message":
            examples = [
                self._normalize_message(row, normalized_split)
                for row in tqdm(
                    dialogue_rows,
                    desc=f"Parsing QATD-2k/{normalized_split}/messages",
                    unit="message",
                    disable=not self.show_progress,
                )
            ]
        else:
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in dialogue_rows:
                grouped[self._row_key(row, granularity)].append(row)

            examples = [
                self._normalize_dialogue(
                    key,
                    group,
                    normalized_split,
                    subjects_by_intervention,
                    questions_by_intervention,
                )
                for key, group in tqdm(
                    sorted(grouped.items()),
                    desc=f"Parsing QATD-2k/{normalized_split}/dialogues",
                    unit="dialogue",
                    disable=not self.show_progress,
                )
            ]

        if normalized_split == "dev":
            # Take the deterministic dev part from the official train file.
            from ..sampling import deterministic_split

            return deterministic_split(examples, split="dev", seed=self.seed)
        return examples
