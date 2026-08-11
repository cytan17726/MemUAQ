import pytest

from memuaq.config import ModelProfile
from memuaq.evaluation import (
    EvaluationProtocolError,
    answer_correct,
    aggregate_metrics,
    compute_metrics,
    judge_row,
    training_success,
)
from memuaq.llm import LLMService, ScriptedClient


class _FailingJudge:
    def chat(self, messages, **kwargs):
        raise RuntimeError("judge unavailable")


def test_acc_ar_js():
    rows = [
        {"id": "a", "answerable": True, "prediction": "Paris", "gold_answers": ["Paris"]},
        {"id": "u", "answerable": False, "prediction": "I cannot verify this.", "gold_answers": []},
    ]
    metrics = compute_metrics(rows)
    assert metrics["acc"] == 1.0
    assert metrics["ar"] == 1.0
    assert metrics["js"] == 1.0


def test_three_run_aggregation():
    summary = aggregate_metrics([{"acc": 0.5, "ar": 0.4, "js": 0.47}] * 3)
    assert summary["runs"] == 3
    assert summary["js"]["mean"] == 0.47
    assert summary["js"]["std"] == 0.0


def test_judge_accepts_paper_verdict_format():
    client = ScriptedClient(['{"verdict":"-1","reason":"needs clarification"}'])
    service = LLMService(
        client,
        ModelProfile(role="judge", provider="fake", model="offline"),
    )
    result = judge_row(
        {"id": "u", "question": "Which one?", "prediction": "Please clarify.", "answerable": False},
        service,
    )
    assert result["score"] == 1
    assert "Example 6" in client.calls[0][1]["content"]


def test_original_answer_match_accepts_gold_inside_response():
    assert answer_correct("Paris is the capital of France.", ["Paris"]) == 1


def test_training_success_uses_llm_judge_for_answerable_case():
    service = LLMService(
        ScriptedClient(['{"score":1}']),
        ModelProfile(role="agent", provider="fake", model="offline"),
    )
    assert training_success(
        question="What is the capital of France?",
        prediction="The answer is Paris.",
        answerable=True,
        gold_answers=["Paris"],
        llm=service,
    )


def test_training_success_falls_back_when_judge_call_fails():
    service = LLMService(
        _FailingJudge(),
        ModelProfile(role="judge", provider="fake", model="offline"),
    )
    assert training_success(
        question="capital?", prediction="Paris", answerable=True,
        gold_answers=["Paris"], llm=service, allow_heuristic=True,
    ) is True
    assert training_success(
        question="unknown?", prediction="I cannot determine this.", answerable=False,
        gold_answers=[], llm=service, allow_heuristic=True,
    ) is True
    judged = judge_row(
        {"id": "u", "question": "unknown?", "prediction": "I cannot determine this.",
         "answerable": False},
        service, allow_heuristic=True,
    )
    assert judged["score"] == 1


def test_real_judge_failure_does_not_fall_back_silently():
    service = LLMService(
        _FailingJudge(),
        ModelProfile(role="judge", provider="fake", model="offline"),
    )
    with pytest.raises(EvaluationProtocolError, match="Judge model call failed"):
        judge_row(
            {"id": "u", "question": "unknown?", "prediction": "I cannot determine this.",
             "answerable": False},
            service,
        )


def test_metrics_reject_missing_explicit_judgment():
    with pytest.raises(EvaluationProtocolError, match="Missing UAQ judgment"):
        compute_metrics(
            [{"id": "u", "answerable": False, "prediction": "unknown"}],
            [],
        )
