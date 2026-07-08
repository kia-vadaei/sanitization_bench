from sanitization_bench import load_dataset


configs = [
    dict(name="tab", split="train", variant="10%"),
    dict(name="synthpai", split="all", variant="10%", granularity="author"),
    dict(name="spia", split="all", variant="10%", source="all"),
    dict(name="qatd2k", split="train", variant="10%", granularity="dialogue"),
]

for cfg in configs:
    dataset = load_dataset(**cfg, seed=42, cache_dir="./data")
    print(dataset)
    print("first id:", dataset[0]["id"] if len(dataset) else None)
