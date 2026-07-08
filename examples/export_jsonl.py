from sanitization_bench import load_dataset


dataset = load_dataset(
    name="spia",
    split="all",
    source="all",
    variant="custom",
    percentage=35,
    seed=42,
    cache_dir="./data",
)

path = dataset.to_jsonl("./normalized_spia_35pct.jsonl")
print(f"Saved {len(dataset)} examples to {path}")
