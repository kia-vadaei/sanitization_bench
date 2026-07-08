from sanitization_bench.schema import base_example, validate_example


def test_base_example_validates():
    example = base_example(
        example_id="x",
        dataset="dummy",
        split="train",
        domain="test",
        text="hello",
    )
    validate_example(example)
