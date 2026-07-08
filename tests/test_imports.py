from sanitization_bench import list_datasets


def test_list_datasets():
    assert set(list_datasets()) == {"tab", "synthpai", "spia", "qatd2k"}
