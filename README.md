# Sanitization Bench

A lightweight, framework-agnostic unified dataset loader for text sanitization and multi-hop privacy leakage experiments.

The package currently supports:

- **TAB**: Text Anonymization Benchmark, ECHR legal documents
- **SynthPAI**: synthetic online-forum personal attribute inference data
- **SPIA**: subject-level privacy inference benchmark
- **QATD-2k with PIIvot**: anonymized educational tutoring dialogues

This package is meant to be the **first stage** of a larger sanitization-defense pipeline:

```text
Dataset Loader -> Sanitization Defense -> Attack / Inference Model -> Utility Evaluation -> Privacy Evaluation
```

It does **not** depend on PyTorch, TensorFlow, or HuggingFace Datasets.

---

## Installation

Development install:

```bash
git clone https://github.com/kia-vadaei/sanitization_bench.git
cd sanitization-bench
pip install -e .
```

Direct GitHub install:

```bash
pip install git+https://github.com/kia-vadaei/sanitization_bench.git
```

---

## Quickstart

```python
from sanitization_bench import load_dataset

# Deterministic 10 percent TAB sample
dataset = load_dataset(
    name="tab",
    split="train",
    variant="10%",
    seed=42,
    cache_dir="./data",
)

print(dataset)
print(len(dataset))
print(dataset[0].keys())

for example in dataset:
    text = example["text"]
    spans = example["spans"]
    # pass text and labels to your sanitizer / evaluator
```

Custom percentage:

```python
dataset = load_dataset(
    name="spia",
    split="all",
    source="all",
    variant="custom",
    percentage=35,
    seed=123,
)
```

Export normalized examples:

```python
dataset.to_jsonl("normalized_spia_35pct.jsonl")
```

---

## Supported datasets

| Name | Dataset | Default unit | Multi-hop privacy structure |
|---|---|---|---|
| `tab` | TAB / Text Anonymization Benchmark | legal document | long-document context, repeated entities, confidential attributes, coreference |
| `synthpai` | SynthPAI | author profile with all comments | attribute inference across multiple comments and profile clues |
| `spia` | SPIA | document with all subjects | multi-subject contextual PII inference |
| `qatd2k` | QATD-2k with PIIvot | full dialogue grouped by `InterventionId` | multi-turn tutor-student dialogue and anchored question metadata |

---

## Sampling variants

```python
variant="10%"       # deterministic 10 percent sample
variant="20%"       # deterministic 20 percent sample
variant="100%"      # full selected split/source
variant="custom"    # requires percentage=<float>
```

Aliases also work:

```python
"10pct", "sample_10", "20pct", "sample_20", "full", "all"
```

Sampling is deterministic with a fixed seed and preserves the original order after selection.

---

## Split behavior

| Dataset | Split behavior |
|---|---|
| `tab` | official `train`, `dev`, `test` |
| `synthpai` | generated deterministic `train`, `dev`, `test`, or `all`; default unit is author-level to avoid comment leakage |
| `spia` | generated deterministic `train`, `dev`, `test`, or `all`; split is document-level |
| `qatd2k` | official `train`, `test`; `dev` is deterministically carved from train |

---

## Dataset-specific options

### TAB

```python
dataset = load_dataset("tab", split="train", variant="20%")
```

Available splits: `train`, `dev`, `test`.

### SynthPAI

```python
# Default: one example per author, preserving multi-comment inference context
dataset = load_dataset("synthpai", split="all", granularity="author")

# Optional: one example per comment
dataset = load_dataset("synthpai", split="all", granularity="comment")
```

### SPIA

```python
# All SPIA source files
dataset = load_dataset("spia", split="all", source="all")

# Individual sources
dataset = load_dataset("spia", source="panorama_151", split="all")
dataset = load_dataset("spia", source="panorama_531", split="all")
dataset = load_dataset("spia", source="tab_144", split="all")
```

### QATD-2k

```python
# Default: one example per InterventionId dialogue
dataset = load_dataset("qatd2k", split="train", granularity="dialogue")

# More specific grouping
dataset = load_dataset("qatd2k", split="train", granularity="question_dialogue")

# One example per message
dataset = load_dataset("qatd2k", split="train", granularity="message")
```

Note: the public QATD-2k anonymous clone is already anonymized, so it is mainly useful for dialogue utility/coherence evaluation after additional sanitization.

---

## Unified output schema

Every item is a plain Python dictionary:

```python
{
    "id": "unique-example-id",
    "dataset": "tab",
    "split": "train",
    "domain": "legal",
    "text": "full document, profile text, or full dialogue",
    "units": [
        {
            "unit_id": "unit-1",
            "type": "document | comment | message",
            "text": "...",
            "speaker": None,
            "start": 0,
            "end": 120,
            "metadata": {},
        }
    ],
    "spans": [
        {
            "span_id": "span-1",
            "start": 15,
            "end": 26,
            "text": "John Smith",
            "label": "PERSON",
            "identifier_type": "DIRECT",
            "subject_id": "subject-1",
            "replacement": None,
            "metadata": {},
        }
    ],
    "subjects": [],
    "attributes": [],
    "relations": [],
    "metadata": {},
    "raw": {},
}
```

The schema intentionally preserves multi-hop structure such as `subject_id`, `document_id`, `author_id`, `thread_id`, `intervention_id`, `unit_id`, and relation links.

---

## Pipeline example

```python
from sanitization_bench import load_dataset

class DummyDefense:
    def sanitize(self, text: str) -> str:
        return text.replace("John", "[NAME]")

class DummyEvaluator:
    def evaluate(self, example, sanitized_text):
        return {
            "id": example["id"],
            "original_length": len(example["text"]),
            "sanitized_length": len(sanitized_text),
            "num_spans": len(example["spans"]),
            "num_attributes": len(example["attributes"]),
        }

examples = load_dataset("tab", split="train", variant="10%", seed=42)
defense = DummyDefense()
evaluator = DummyEvaluator()

results = []
for example in examples:
    sanitized = defense.sanitize(example["text"])
    results.append(evaluator.evaluate(example, sanitized))

print(results[:3])
```

---

## Raw sources

The loaders download from the public raw files you provided:

- TAB train/dev/test JSON files from `NorskRegnesentral/text-anonymization-benchmark`
- SynthPAI JSONL from `eth-sri/SynthPAI`
- SPIA JSONL files from `maisonOP/spia`
- QATD-2k anonymous clone CSV files from `Eedi/qatd-2k-anonymous-clone`

Files are cached under `cache_dir/<dataset-name>/`.
