from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetRecord:
    id: str | int
    question: str
    answers: list[str]
    answerable: bool
    question_type: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_answer(self) -> list[str]:
        return self.answers


def normalize_record(raw: dict[str, Any], *, location: str = "record") -> DatasetRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be an object")
    for key in ("id", "question"):
        if key not in raw:
            raise ValueError(f"{location} missing {key}")
    answers = raw.get("answers", raw.get("answer", raw.get("gold_answer", [])))
    if isinstance(answers, str):
        answers = [answers]
    if answers is None:
        answers = []
    if not isinstance(answers, list):
        raise ValueError(f"{location}.answers must be a list or string")
    if "answerable" in raw:
        answerable = bool(raw["answerable"])
    else:
        answerable = str(raw.get("question_type", "")).lower() not in {
            "unanswerable", "incomprehensible", "false_presuppositions",
            "underspecified", "safety-concern", "modality-limited",
        }
    known = {"id", "question", "answers", "answer", "gold_answer", "answerable", "question_type"}
    metadata = {key: value for key, value in raw.items() if key not in known}
    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"{location}.question is empty")
    return DatasetRecord(
        id=raw["id"],
        question=question,
        answers=[str(item) for item in answers],
        answerable=answerable,
        question_type=str(raw.get("question_type", "default")),
        metadata=metadata,
    )


def load_dataset(path: str | Path, *, fmt: str | None = None) -> list[DatasetRecord]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    format_name = (fmt or file_path.suffix.lstrip(".") or "json").lower()
    if format_name in {"jsonl", "ndjson"}:
        raw_items: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    raw_items.append(json.loads(line))
    else:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("items", []))
        if not isinstance(payload, list):
            raise ValueError(f"Dataset must be a list: {file_path}")
        raw_items = payload
    return [normalize_record(item, location=f"{file_path}:{i}") for i, item in enumerate(raw_items, 1)]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_dict(record: DatasetRecord) -> dict[str, Any]:
    return asdict(record)

