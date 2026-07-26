from src.data.prepare_dataset import deduplicate


def test_deduplicate():
    rows = [
        {"input": "Same", "instruction": "x", "response": "a", "category": "git"},
        {"input": "same", "instruction": "x", "response": "b", "category": "git"},
    ]
    assert len(deduplicate(rows)) == 1
