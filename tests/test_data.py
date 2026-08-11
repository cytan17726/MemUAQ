import json

from memuaq.data import load_dataset, normalize_record


def test_normalize_dataset_record():
    row = normalize_record({"id": 1, "question": "q", "answer": "a", "answerable": True})
    assert row.answers == ["a"]
    assert row.answerable is True


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"id": 1, "question": "q", "answer": [], "answerable": False}) + "\n")
    assert load_dataset(path)[0].answerable is False

