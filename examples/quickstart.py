from sanitization_bench import load_dataset


def main() -> None:
    dataset = load_dataset(
        name="tab",
        split="train",
        variant="10%",
        seed=42,
        cache_dir="./data",
    )
    print(dataset)
    print(dataset.preview(1)[0].keys())


if __name__ == "__main__":
    main()
