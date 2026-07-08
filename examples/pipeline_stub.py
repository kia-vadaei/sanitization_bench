from sanitization_bench import load_dataset


class DummyDefense:
    def sanitize(self, text: str) -> str:
        return text


class DummyEvaluator:
    def evaluate(self, example, sanitized_text: str):
        return {
            "id": example["id"],
            "dataset": example["dataset"],
            "num_chars": len(example["text"]),
            "num_spans": len(example["spans"]),
            "num_attributes": len(example["attributes"]),
            "sanitized_num_chars": len(sanitized_text),
        }


dataset = load_dataset("synthpai", split="all", variant="10%", seed=42)
defense = DummyDefense()
evaluator = DummyEvaluator()

results = []
for example in dataset:
    sanitized = defense.sanitize(example["text"])
    results.append(evaluator.evaluate(example, sanitized))

print(results[:3])
