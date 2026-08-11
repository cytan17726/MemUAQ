from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from .data import write_jsonl
from .llm import LLMService
from .prompts.judge import (
    QA_JUDGE_PROMPT,
    UAQ_JUDGE_PROMPT,
    qa_judge_user_prompt,
    uaq_judge_user_prompt,
)


class EvaluationProtocolError(RuntimeError):
    """Raised when paper-protocol evaluation cannot be completed faithfully."""


def normalize_answer(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text).lower())
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).strip()


def _extract_answer_text(prediction: str) -> str:
    match = re.search(r"<answer>([\s\S]*?)</answer>", str(prediction), flags=re.I)
    return match.group(1).strip() if match else str(prediction).strip()


def answer_correct(prediction: str, answers: list[str]) -> int:
    prediction_text = _extract_answer_text(prediction)
    pred_normalized = normalize_answer(prediction_text)
    if not answers or not pred_normalized:
        return 0
    prediction_lower = prediction_text.lower()
    for answer in answers:
        answer_text = str(answer).strip()
        answer_normalized = normalize_answer(answer_text)
        if not answer_normalized:
            continue
        if answer_normalized in {"yes", "no"}:
            if re.search(rf"\b{re.escape(answer_normalized)}\b", prediction_lower):
                return 1
            continue
        if pred_normalized == answer_normalized:
            return 1
        if len(answer_normalized) <= 3 and " " not in answer_text:
            if re.search(rf"\b{re.escape(answer_text.lower())}\b", prediction_lower):
                return 1
            continue
        if answer_text.lower() in prediction_lower or answer_normalized in pred_normalized:
            return 1
    return 0


def heuristic_acceptable(prediction: str) -> int:
    text = prediction.lower()
    refusal_markers = (
        "can't answer", "cannot answer", "cannot determine", "could not determine",
        "cannot verify", "could not verify", "not enough information", "i don't know",
        "unable to", "question is nonsensical", "question may contain inaccuracies",
    )
    clarification_markers = (
        "need more clarification", "need clarification", "please clarify",
        "could you clarify", "can you clarify", "which one do you mean",
        "which one are you referring to", "which resort are you interest in",
        "which resort are you interested in",
    )
    return int(any(marker in text for marker in refusal_markers + clarification_markers))


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.I)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for position, character in enumerate(raw):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _binary_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) >= 0.5)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return 1
    if normalized in {"0", "false", "no", "n"}:
        return 0
    return None


def _refusal_verdict(value: Any) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "refuse", "refused", "reject", "rejected"}:
        return 1
    if normalized in {"0", "false", "no", "answer", "answered"}:
        return 0
    if normalized in {"-1", "clarify", "clarification", "need_clarification"}:
        return -1
    return None


def judge_row(
    row: dict[str, Any],
    llm: LLMService | None = None,
    *,
    allow_heuristic: bool = False,
) -> dict[str, Any]:
    if row.get("answerable", False):
        score = answer_correct(str(row.get("prediction", "")), list(row.get("gold_answers") or []))
        return {"id": row.get("id"), "answerable": True, "score": score,
                "metric": "accuracy", "reason": "normalized answer match", "raw": None}
    if llm is None:
        if not allow_heuristic:
            raise EvaluationProtocolError(
                "UAQ evaluation requires a configured judge model; heuristic judging is disabled"
            )
        score = heuristic_acceptable(str(row.get("prediction", "")))
        return {"id": row.get("id"), "answerable": False, "score": score,
                "metric": "acceptable_ratio", "reason": "offline heuristic", "raw": None}
    prompt = (
        {"role": "system", "content": UAQ_JUDGE_PROMPT},
        {"role": "user", "content": uaq_judge_user_prompt(
            str(row.get("question", "")), str(row.get("prediction", "")), no_reason=False,
        )},
    )
    try:
        response = llm.chat(prompt)
    except Exception as exc:
        if allow_heuristic:
            score = heuristic_acceptable(str(row.get("prediction", "")))
            return {"id": row.get("id"), "answerable": False, "score": score,
                    "metric": "acceptable_ratio", "reason": "judge call fallback", "raw": None}
        raise EvaluationProtocolError(
            f"Judge model call failed for id={row.get('id')}"
        ) from exc
    parsed = _extract_json_dict(response.content)
    verdict = _refusal_verdict(parsed.get("verdict")) if parsed else None
    if verdict is None:
        if not allow_heuristic:
            raise EvaluationProtocolError(
                f"Judge response is not a valid verdict for id={row.get('id')}"
            )
        score = heuristic_acceptable(str(row.get("prediction", "")))
        reason = "judge parse fallback"
    else:
        score = int(verdict != 0)
        reason = str(parsed.get("reason", ""))
    return {"id": row.get("id"), "answerable": False, "score": score,
            "metric": "acceptable_ratio", "reason": reason, "raw": response.content}


def training_success(
    *,
    question: str,
    prediction: str,
    answerable: bool,
    gold_answers: list[str],
    llm: LLMService,
    allow_heuristic: bool = False,
) -> bool:
    """Match the original online success test used while updating memory."""
    if answerable:
        messages = (
            {"role": "system", "content": QA_JUDGE_PROMPT},
            {"role": "user", "content": qa_judge_user_prompt(
                question, gold_answers, _extract_answer_text(prediction),
            )},
        )
        try:
            response = llm.chat(messages)
        except Exception as exc:
            if allow_heuristic:
                return bool(answer_correct(prediction, gold_answers))
            raise EvaluationProtocolError(
                "Training success judge call failed for an answerable sample"
            ) from exc
        parsed = _extract_json_dict(response.content)
        score = None
        if parsed:
            for key in ("score", "correct", "is_correct"):
                if key in parsed:
                    score = _binary_score(parsed[key])
                    if score is not None:
                        break
        if score is None:
            if allow_heuristic:
                return bool(answer_correct(prediction, gold_answers))
            raise EvaluationProtocolError(
                "Training success judge returned an invalid answerable-sample score"
            )
        return bool(score)

    messages = (
        {"role": "system", "content": UAQ_JUDGE_PROMPT},
        {"role": "user", "content": uaq_judge_user_prompt(question, prediction)},
    )
    try:
        response = llm.chat(messages)
    except Exception as exc:
        if allow_heuristic:
            return bool(heuristic_acceptable(prediction))
        raise EvaluationProtocolError(
            "Training success judge call failed for an unanswerable sample"
        ) from exc
    parsed = _extract_json_dict(response.content)
    verdict = _refusal_verdict(parsed.get("verdict")) if parsed else None
    if verdict is None:
        if allow_heuristic:
            return bool(heuristic_acceptable(prediction))
        raise EvaluationProtocolError(
            "Training success judge returned an invalid unanswerable-sample verdict"
        )
    return bool(verdict != 0)


def compute_metrics(rows: Iterable[dict[str, Any]], judgments: Iterable[dict[str, Any]] | None = None,
                    acc_weight: float = 0.7, ar_weight: float = 0.3) -> dict[str, Any]:
    row_list = list(rows)
    judgment_map = {str(item.get("id")): item for item in judgments or []}
    require_judgment = judgments is not None
    acc_scores: list[int] = []
    ar_scores: list[int] = []
    for row in row_list:
        if row.get("answerable", False):
            acc_scores.append(answer_correct(str(row.get("prediction", "")), list(row.get("gold_answers") or [])))
        else:
            judgment = judgment_map.get(str(row.get("id")))
            if judgment is None and require_judgment:
                raise EvaluationProtocolError(
                    f"Missing UAQ judgment for id={row.get('id')}"
                )
            ar_scores.append(int(judgment["score"]) if judgment else heuristic_acceptable(str(row.get("prediction", ""))))
    acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
    ar = sum(ar_scores) / len(ar_scores) if ar_scores else 0.0
    return {"schema_version": 1, "count": len(row_list), "answerable_count": len(acc_scores),
            "unanswerable_count": len(ar_scores), "acc": acc, "ar": ar,
            "js": acc_weight * acc + ar_weight * ar,
            "weights": {"acc": acc_weight, "ar": ar_weight}}


def aggregate_metrics(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    runs = list(metrics)
    result: dict[str, Any] = {"schema_version": 1, "runs": len(runs)}
    for key in ("acc", "ar", "js"):
        values = [float(run[key]) for run in runs if key in run and math.isfinite(float(run[key]))]
        result[key] = {"mean": statistics.mean(values) if values else 0.0,
                       "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                       "values": values}
    return result


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def judge_file(
    input_path: str | Path,
    output_path: str | Path,
    llm: LLMService | None = None,
    *,
    allow_heuristic: bool = False,
) -> list[dict[str, Any]]:
    rows = load_jsonl(input_path)
    judgments = [judge_row(row, llm, allow_heuristic=allow_heuristic) for row in rows]
    write_jsonl(output_path, judgments)
    return judgments
