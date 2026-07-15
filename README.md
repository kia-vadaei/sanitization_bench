# Sanitization Bench

A lightweight, framework-agnostic dataset loader for text sanitization, privacy leakage, re-identification, and multi-hop inference experiments.

The package currently supports:

* **TAB**: Text Anonymization Benchmark based on ECHR legal documents
* **SynthPAI**: synthetic online-forum data for personal-attribute inference
* **SPIA**: subject-level privacy-inference benchmark
* **QATD-2k with PIIvot**: anonymized educational tutoring dialogues

Sanitization Bench is intended to be the dataset-loading and normalization stage of a larger privacy evaluation pipeline:

```text
Dataset Loader
    -> Sanitization Defense
    -> Attack / Inference Model
    -> Utility Evaluation
    -> Privacy Evaluation
```

The package does not depend on PyTorch, TensorFlow, or Hugging Face Datasets.

---

## Installation

### Development installation

```bash
git clone https://github.com/kia-vadaei/sanitization_bench.git
cd sanitization_bench
pip install -e .
```

### Direct GitHub installation

```bash
pip install git+https://github.com/kia-vadaei/sanitization_bench.git
```

---

## Quickstart

```python
from sanitization_bench import load_dataset

dataset = load_dataset(
    name="tab",
    split="train",
    percentage=10,
    seed=42,
    cache_dir="./data",
)

print(dataset)
print(len(dataset))
print(dataset[0].keys())

for example in dataset:
    text = example["text"]
    spans = example["spans"]

    # Pass normalized text and privacy labels to a sanitizer,
    # attack model, or evaluation pipeline.
```

Sampling is applied after the requested split is loaded.

For example:

```python
train_dataset = load_dataset(
    name="tab",
    split="train",
    percentage=35,
)

test_dataset = load_dataset(
    name="tab",
    split="test",
    percentage=35,
)
```

This independently retains:

* 35% of the TAB training split
* 35% of the TAB test split

---

## Sampling

The preferred way to request a dataset portion is through `percentage`:

```python
dataset = load_dataset(
    name="spia",
    split="train",
    source="all",
    variant="",
    percentage=35,
    seed=123,
)
```

Because `variant=""` is the default, it can normally be omitted:

```python
dataset = load_dataset(
    name="spia",
    split="train",
    source="all",
    percentage=35,
    seed=123,
)
```

### Full selected split

When neither `variant` nor `percentage` specifies a subset, the complete selected split is returned:

```python
dataset = load_dataset(
    name="tab",
    split="train",
)
```

Equivalent explicit form:

```python
dataset = load_dataset(
    name="tab",
    split="train",
    variant="",
    percentage=100,
)
```

### Percentage aliases

A percentage can also be passed through `variant`:

```python
dataset = load_dataset(
    name="tab",
    split="train",
    variant="35%",
)
```

The following forms are supported:

```python
variant="", percentage=35
variant="35%"
variant="35"
variant="0.35"
variant="full"
variant="all"
variant="100%"
```

The legacy custom form remains supported:

```python
dataset = load_dataset(
    name="spia",
    split="train",
    variant="custom",
    percentage=35,
)
```

### Sampling rules

* Sampling occurs after train, dev, test, or all selection.
* Each split is sampled independently.
* Any percentage in the range `(0, 100]` is supported.
* Sampling is deterministic for a fixed seed.
* Original record order is preserved after selection.
* At least one example is returned for a non-empty dataset and a valid percentage.

Example:

```text
Train split size: 1,000
Test split size:    200
Requested portion:   35%

Returned train examples: 350
Returned test examples:   70
```

Do not specify conflicting sampling values:

```python
# Invalid: the percentage is defined twice
dataset = load_dataset(
    name="tab",
    split="train",
    variant="35%",
    percentage=20,
)
```

---

## Exporting normalized examples

```python
dataset = load_dataset(
    name="spia",
    split="train",
    source="all",
    percentage=35,
)

dataset.to_jsonl("normalized_spia_train_35pct.jsonl")
```

The exported JSONL file contains one normalized example per line.

---

## Supported datasets

| Name       | Dataset                            | Default example unit                   | Multi-hop privacy structure                                                        |
| ---------- | ---------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| `tab`      | TAB / Text Anonymization Benchmark | legal document                         | long-document context, repeated entities, confidential attributes, and coreference |
| `synthpai` | SynthPAI                           | author profile containing all comments | attribute inference across comments and profile clues                              |
| `spia`     | SPIA                               | document containing all subjects       | multi-subject contextual PII inference                                             |
| `qatd2k`   | QATD-2k with PIIvot                | dialogue grouped by `InterventionId`   | multi-turn tutor-student dialogue and anchored question metadata                   |

---

## Split behavior

| Dataset    | Split behavior                                                                              |
| ---------- | ------------------------------------------------------------------------------------------- |
| `tab`      | Uses official `train`, `dev`, and `test` splits                                             |
| `synthpai` | Creates deterministic `train`, `dev`, and `test` splits; also supports `all`                |
| `spia`     | Creates deterministic document-level `train`, `dev`, and `test` splits; also supports `all` |
| `qatd2k`   | Uses official `train` and `test`; `dev` is deterministically created from train             |

For generated splits, the same `seed` produces the same assignments.

Sampling is performed only after the requested split has been selected.

---

## Dataset-specific options

### TAB

```python
dataset = load_dataset(
    name="tab",
    split="train",
    percentage=20,
)
```

Available splits:

```text
train
dev
test
```

TAB examples preserve document-level text, annotated spans, entity references, identifier categories, and replacement information.

---

### SynthPAI

Default author-level loading:

```python
dataset = load_dataset(
    name="synthpai",
    split="train",
    granularity="author",
)
```

The author granularity combines all comments from one author into one example, preserving cross-comment personal-attribute inference context.

Optional comment-level loading:

```python
dataset = load_dataset(
    name="synthpai",
    split="train",
    granularity="comment",
)
```

Available splits:

```text
train
dev
test
all
```

---

### SPIA

Load all SPIA sources:

```python
dataset = load_dataset(
    name="spia",
    split="train",
    source="all",
)
```

Load one source:

```python
dataset = load_dataset(
    name="spia",
    split="train",
    source="panorama_151",
)

dataset = load_dataset(
    name="spia",
    split="train",
    source="panorama_531",
)

dataset = load_dataset(
    name="spia",
    split="train",
    source="tab_144",
)
```

Available sources:

```text
all
panorama_151
panorama_531
tab_144
```

Available splits:

```text
train
dev
test
all
```

---

### QATD-2k

Default dialogue-level loading:

```python
dataset = load_dataset(
    name="qatd2k",
    split="train",
    granularity="dialogue",
)
```

Group messages by question and dialogue:

```python
dataset = load_dataset(
    name="qatd2k",
    split="train",
    granularity="question_dialogue",
)
```

Load one example per message:

```python
dataset = load_dataset(
    name="qatd2k",
    split="train",
    granularity="message",
)
```

Available granularities:

```text
dialogue
question_dialogue
message
```

The public QATD-2k anonymous clone is already anonymized. It is primarily suitable for evaluating dialogue utility, coherence, and degradation after additional sanitization.

---

## Unified output schema

Every loaded item is returned as a plain Python dictionary with the same top-level schema:

```python
{
    "id": "unique-example-id",
    "dataset": "tab",
    "split": "train",
    "domain": "legal",
    "text": "full document, profile text, or dialogue",
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

### Common fields

| Field        | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| `id`         | Globally usable example identifier                                   |
| `dataset`    | Source dataset name                                                  |
| `split`      | Effective split name                                                 |
| `domain`     | Dataset domain such as legal, forum, privacy inference, or education |
| `text`       | Main normalized text consumed by sanitizers and models               |
| `units`      | Ordered source units such as documents, comments, or messages        |
| `spans`      | Character-level sensitive spans when available                       |
| `subjects`   | People or entities associated with private information               |
| `attributes` | Subject-level or profile-level private attributes                    |
| `relations`  | Links between units, spans, subjects, and attributes                 |
| `metadata`   | Normalized dataset-specific metadata                                 |
| `raw`        | Original source record when `keep_raw=True`                          |

The schema preserves multi-hop identifiers and relations such as:

```text
subject_id
document_id
author_id
thread_id
intervention_id
unit_id
span_id
relation links
```

Not every dataset contains every privacy annotation type. Missing structures are represented by empty lists rather than omitted fields.

---

## Dataset metadata

The returned `SanitizationDataset` includes sampling metadata:

```python
dataset.metadata
```

Example:

```python
{
    "source": "all",
    "granularity": None,
    "cache_dir": "/absolute/path/to/data",
    "requested_variant": "",
    "requested_percentage": 35,
    "effective_variant": "35%",
    "resolved_percentage": 35.0,
    "original_size_before_sampling": 1000,
    "sampled_size": 350,
    "sampling_seed": 42,
    "sampling_scope": "train",
}
```

The dataset-level `variant` property stores the effective normalized value:

```python
print(dataset.variant)
```

Output:

```text
35%
```

This remains explicit even when the caller used:

```python
variant=""
percentage=35
```

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
            "dataset": example["dataset"],
            "split": example["split"],
            "original_length": len(example["text"]),
            "sanitized_length": len(sanitized_text),
            "num_spans": len(example["spans"]),
            "num_attributes": len(example["attributes"]),
        }


examples = load_dataset(
    name="tab",
    split="train",
    percentage=10,
    seed=42,
)

defense = DummyDefense()
evaluator = DummyEvaluator()

results = []

for example in examples:
    sanitized_text = defense.sanitize(example["text"])

    results.append(
        evaluator.evaluate(
            example=example,
            sanitized_text=sanitized_text,
        )
    )

print(results[:3])
```

---

## Listing registered datasets

```python
from sanitization_bench import list_datasets

print(list_datasets())
```

Expected output:

```python
["qatd2k", "spia", "synthpai", "tab"]
```

The exact order depends on registry implementation.

---

## Loader parameters

```python
load_dataset(
    name,
    split="train",
    variant="",
    percentage=None,
    seed=42,
    cache_dir="./data",
    source="default",
    granularity=None,
    download=True,
    show_progress=True,
    keep_raw=True,
    validate=True,
)
```

| Parameter       | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| `name`          | Registered dataset name                                        |
| `split`         | Requested train, dev, test, or all split                       |
| `variant`       | Optional percentage alias or backward-compatible sampling mode |
| `percentage`    | Preferred numeric percentage of the selected split             |
| `seed`          | Deterministic splitting and sampling seed                      |
| `cache_dir`     | Raw dataset cache directory                                    |
| `source`        | Dataset-specific source selector                               |
| `granularity`   | Dataset-specific example grouping                              |
| `download`      | Download missing public files                                  |
| `show_progress` | Display progress bars                                          |
| `keep_raw`      | Preserve original records                                      |
| `validate`      | Validate normalized examples                                   |

---

## Raw sources

The adapters load public files from:

* TAB JSON files from `NorskRegnesentral/text-anonymization-benchmark`
* SynthPAI JSONL data from `eth-sri/SynthPAI`
* SPIA JSONL files from `maisonOP/spia`
* QATD-2k anonymous-clone CSV files from `Eedi/qatd-2k-anonymous-clone`

Downloaded files are cached under:

```text
cache_dir/<dataset-name>/
```

Raw source formats may differ between datasets. The loader converts them into the common Python dictionary schema before returning examples.

---

## Design principles

* Framework-independent Python objects
* Consistent top-level output fields
* Deterministic generated splits
* Deterministic percentage sampling
* Independent sampling of train, dev, and test
* Preservation of multi-hop privacy structure
* Preservation of raw source records when requested
* Support for document-, profile-, dialogue-, and message-level examples
* No mandatory deep-learning framework dependency
