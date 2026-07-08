from sanitization_bench.sampling import deterministic_sample, resolve_percentage


def test_resolve_percentage():
    assert resolve_percentage("10%", None) == 10.0
    assert resolve_percentage("20pct", None) == 20.0
    assert resolve_percentage("full", None) == 100.0
    assert resolve_percentage("custom", 33) == 33.0


def test_deterministic_sample():
    examples = [{"id": str(i)} for i in range(100)]
    a = deterministic_sample(examples, variant="10%", seed=42)
    b = deterministic_sample(examples, variant="10%", seed=42)
    assert a == b
    assert len(a) == 10
