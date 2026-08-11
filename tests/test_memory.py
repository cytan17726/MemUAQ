import json

from memuaq.config import ModelProfile
import pytest

from memuaq.memory import Experience, MemoryProtocolError, create_memory
from memuaq.llm import LLMService, ScriptedClient


def test_all_public_memory_methods_build_and_retrieve(tmp_path):
    experience = Experience(id="1", question="Is a fictional premise true?", answer="No",
                            answerable=False, trajectory="Search and verify the premise.", success=True)
    for method in ("expel", "awm", "content_memory"):
        memory = create_memory(method, store_path=str(tmp_path / method), content_types=["workflow"])
        memory.build([experience])
        assert memory.retrieve("fictional premise", 2)


def test_memevolve_builds_structured_principle():
    service = LLMService(
        ScriptedClient([
            '[DESCRIPTION]:\nVerify the premise before answering.\n[STRUCTURE]:\n[["agent","verifies","premise"]]',
            "The agent encountered an uncertain premise and verified it before answering.",
        ]),
        ModelProfile(role="memory", provider="fake", model="offline"),
    )
    memory = create_memory("memevolve", llm=service, embedder=ScriptedClient())
    memory.build([Experience(id="1", question="fictional premise", answer="No",
                             answerable=False, trajectory="verify", success=True)])
    retrieved = memory.retrieve("fictional premise", 1)
    assert retrieved and "[Structure/Pattern]" in retrieved[0].content


def test_agentkb_requires_and_preserves_structured_package():
    payload = {
        "agent_planning": "Decompose the task, select the Wikipedia search tool, inspect evidence, and make an explicit evidence-based answerability decision.",
        "search_agent_planning": "Form a specific canonical query, prioritize the relevant encyclopedia page, inspect matching passages, and cross-check the requested relation.",
        "agent_experience": "Successful execution depended on verifying the premise before answering and avoiding unsupported completion when the evidence remained inconclusive.",
        "search_agent_experience": "Canonical entity names and focused lookup terms produced the most useful evidence, while broad repeated searches added little value.",
    }
    service = LLMService(
        ScriptedClient([json.dumps(payload), "fictional premise", "1. Verify the premise", "Use verified evidence."]),
        ModelProfile(role="memory", provider="fake", model="offline"),
    )
    memory = create_memory("agentkb", llm=service, embedder=ScriptedClient())
    memory.build([Experience(id="1", question="fictional premise", answer="No",
                             answerable=False, trajectory="verify", success=True)])
    retrieved = memory.retrieve("fictional premise", 1)
    assert retrieved and "AGENT-KB Student Guidance" in retrieved[0].content


def test_awm_skips_failed_experience():
    memory = create_memory("awm")
    memory.build([Experience(id="x", question="q", answer="", answerable=True, success=False)])
    assert memory.items == []


def test_hybrid_retrieval_records_score():
    memory = create_memory("expel", embedder=ScriptedClient())
    memory.build([Experience(id="x", question="capital city", answer="Paris",
                             answerable=True, trajectory="evidence", success=True)])
    items = memory.retrieve("capital city", 1)
    assert items and "retrieval_score" in items[0].metadata


def test_content_memory_keeps_outcome_specific_types_separate():
    service = LLMService(
        ScriptedClient([
            "Successful trajectory summary with reusable verification steps.",
            "Verify the premise with authoritative evidence before answering.",
            "Use a canonical entity name in the first search.\nCross-check the decisive claim before finishing.",
            "Failed trajectory summary showing unsupported assumptions.",
            "Avoid treating an unverified premise as an established fact.",
            "Do not repeat an uninformative broad search.\nAvoid answering when the premise remains unsupported.",
        ]),
        ModelProfile(role="memory", provider="fake", model="offline"),
    )
    memory = create_memory(
        "content_memory",
        llm=service,
        content_types=[
            "principle_success", "principle_failure",
            "insight_success", "insight_failure",
        ],
    )
    memory.add_experience(Experience(
        id="s", question="verified question", answer="yes",
        answerable=True, trajectory="verified evidence", success=True,
    ))
    memory.add_experience(Experience(
        id="f", question="unsupported question", answer="no",
        answerable=False, trajectory="unsupported premise", success=False,
    ))

    by_type = {item.memory_type: item for item in memory.items}
    assert set(by_type) == {
        "principle_success", "principle_failure",
        "insight_success", "insight_failure",
    }
    assert by_type["principle_success"].metadata["success"] is True
    assert by_type["principle_failure"].metadata["success"] is False
    assert by_type["insight_success"].metadata["success"] is True
    assert by_type["insight_failure"].metadata["success"] is False


def test_content_memory_skips_case_insensitive_exact_duplicates():
    memory = create_memory("content_memory", content_types=["raw"])
    experience = Experience(
        id="1", question="same question", answer="same answer",
        answerable=True, trajectory="the same sufficiently long trajectory", success=True,
    )
    memory.add_experience(experience)
    memory.add_experience(Experience(**{**experience.__dict__, "id": "2"}))
    assert len(memory.items) == 1


def test_content_memory_artifact_roundtrip_infers_types(tmp_path):
    memory = create_memory("content_memory", content_types=["workflow"])
    memory.add_experience(Experience(
        id="1", question="workflow question", answer="yes", answerable=True,
        trajectory="A sufficiently long successful trajectory.", success=True,
    ))
    artifact = tmp_path / "memory.json"
    memory.save(artifact)

    loaded = type(memory).load(artifact)
    assert loaded.content_types == ["workflow"]
    assert loaded.retrieve("workflow question", 1)


def test_content_memory_balanced_retrieval_honors_per_type_target():
    memory = create_memory(
        "content_memory",
        content_types=["workflow", "principle"],
        selection_topk_multiplier=1,
        balance_top_k_by_type=True,
        min_per_type=1,
    )
    memory.add_experience(Experience(
        id="1", question="balanced question", answer="yes", answerable=True,
        trajectory="A sufficiently long successful trajectory.", success=True,
    ))
    retrieved = memory.retrieve("balanced question", 2)
    assert len(retrieved) == 2
    assert {item.metadata["content_type"] for item in retrieved} == {"workflow", "principle"}


def test_content_memory_default_budget_is_two():
    memory = create_memory("content_memory", content_types=["workflow", "principle"])
    for index in range(3):
        memory.add_experience(Experience(
            id=str(index), question=f"budget question {index}", answer="yes",
            answerable=True, trajectory="A sufficiently long successful trajectory.", success=True,
        ))
    retrieved = memory.retrieve("budget question", 2)
    assert len(retrieved) == 2


def test_embedding_failure_does_not_fall_back_to_lexical():
    class FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("embedding unavailable")

    memory = create_memory(
        "content_memory", content_types=["raw"], embedder=FailingEmbedder(),
    )
    memory.add_experience(Experience(
        id="1", question="strict embedding", answer="yes", answerable=True,
        trajectory="A sufficiently long trajectory for strict embedding.", success=True,
    ))
    with pytest.raises(MemoryProtocolError, match="embedding request failed"):
        memory.retrieve("strict embedding")


def test_memory_llm_failure_does_not_fall_back_to_raw_trajectory():
    class FailingClient:
        def chat(self, messages, **kwargs):
            raise RuntimeError("memory model unavailable")

    service = LLMService(
        FailingClient(),
        ModelProfile(role="memory", provider="openai_compatible", model="test"),
    )
    memory = create_memory("content_memory", content_types=["workflow"], llm=service)
    with pytest.raises(MemoryProtocolError, match="Memory LLM call failed"):
        memory.add_experience(Experience(
            id="1", question="strict extraction", answer="yes", answerable=True,
            trajectory="A sufficiently long successful trajectory.", success=True,
        ))


def test_structured_memory_rejects_malformed_real_model_output():
    service = LLMService(
        ScriptedClient(["not valid json"]),
        ModelProfile(role="memory", provider="openai_compatible", model="test"),
    )
    memory = create_memory("awm", llm=service)
    with pytest.raises(MemoryProtocolError, match="invalid JSON"):
        memory.add_experience(Experience(
            id="1", question="strict workflow", answer="yes", answerable=True,
            trajectory="A sufficiently long successful trajectory.", success=True,
        ))
