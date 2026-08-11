from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .llm import LLMService
from .prompts.memory import content_extraction_prompt, extraction_prompt


class MemoryProtocolError(RuntimeError):
    """Raised when a configured memory protocol cannot be executed faithfully."""


@dataclass(frozen=True)
class Experience:
    id: str | int
    question: str
    answer: str
    answerable: bool
    trajectory: str | list[dict[str, Any]] = ""
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    id: str
    query: str
    content: str
    memory_type: str
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def lexical_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(1, len(a | b))


def _tfidf_scores(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        return [float(value) for value in cosine_similarity(
            vectorizer.transform([query]), matrix,
        ).flatten()]
    except (ImportError, ValueError):
        return [lexical_similarity(query, text) for text in texts]


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise MemoryProtocolError(
            f"Embedding dimension mismatch during similarity: {len(left)} != {len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _top_indices(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:max(1, k)]


def _chat_text(llm: LLMService | None, prompt: str) -> str:
    if llm is None:
        return ""
    try:
        content = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    except Exception as exc:
        raise MemoryProtocolError(
            f"Memory LLM call failed during: {prompt.splitlines()[0][:120]}"
        ) from exc
    if not content:
        raise MemoryProtocolError(
            f"Memory LLM returned empty content during: {prompt.splitlines()[0][:120]}"
        )
    return content


def _strict_llm_protocol(llm: LLMService | None) -> bool:
    """Return true for real endpoints; fake clients may use test fallbacks."""
    return llm is not None and getattr(llm.profile, "provider", "") != "fake"


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw).strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _trajectory_items(experience: Experience) -> list[dict[str, Any]]:
    raw: Any = experience.trajectory
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            raw = parsed if isinstance(parsed, list) else stripped
        except json.JSONDecodeError:
            raw = stripped
    if isinstance(raw, str):
        return [{"type": "trajectory", "content": raw}]
    items: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return items
    for step in raw:
        if not isinstance(step, dict):
            if str(step).strip():
                items.append({"type": "step", "content": str(step).strip()})
            continue
        if step.get("type") and step.get("content") is not None:
            content = str(step.get("content", "")).strip()
            if content:
                items.append({"type": str(step.get("type")), "content": content})
            continue
        for key, item_type in (("assistant", "assistant"), ("action", "action"),
                               ("observation", "observation")):
            content = str(step.get(key, "")).strip()
            if content:
                items.append({"type": item_type, "content": content})
    return items


def _format_trajectory(experience: Experience) -> str:
    items = _trajectory_items(experience)
    if not items:
        return f"Task: {experience.question}\nNo execution trajectory available."
    lines = [f"Task: {experience.question}", ""]
    for index, step in enumerate(items, 1):
        lines.append(f"Step {index} ({step.get('type', 'step')}): {step.get('content', '')}")
    if experience.answer.strip():
        lines.extend(("", f"Final Result: {experience.answer}"))
    return "\n".join(lines)


def _clean_insights(content: str, limit: int = 4) -> list[str]:
    insights: list[str] = []
    for line in str(content).splitlines():
        clean = line.strip().lstrip("•-*1234567890. ").strip()
        for prefix in ("Do:", "Avoid:", "Insight:", "Tip:", "Note:"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
                break
        if clean and len(clean) > 10:
            insights.append(clean)
    return insights[:limit]


class MemoryMethod:
    name = "base"

    def __init__(
        self,
        *,
        top_k: int = 2,
        store_path: str | None = None,
        llm: LLMService | None = None,
        embedder: Any | None = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        **_: Any,
    ):
        self.top_k = max(1, int(top_k))
        self.store_path = Path(store_path).expanduser() if store_path else None
        self.llm = llm
        self.embedder = embedder
        self.semantic_weight = float(semantic_weight)
        self.lexical_weight = float(lexical_weight)
        self.items: list[MemoryItem] = []

    def build(self, experiences: Iterable[Experience]) -> None:
        for experience in experiences:
            self.add_experience(experience)
        self.finalize()

    def finalize(self) -> None:
        self._ensure_embeddings()
        if self.store_path:
            self.save(self.store_path)

    def add_experience(self, experience: Experience) -> None:
        content = self.extract(experience)
        if content:
            self.items.append(self._new_item(experience, content, self.name))

    def _new_item(self, experience: Experience, content: str, memory_type: str,
                  metadata: dict[str, Any] | None = None) -> MemoryItem:
        base_metadata = {"success": experience.success, "answerable": experience.answerable}
        base_metadata.update(metadata or {})
        return MemoryItem(
            id=uuid.uuid4().hex,
            query=experience.question,
            content=content,
            memory_type=memory_type,
            source_ids=[str(experience.id)],
            metadata=base_metadata,
        )

    def extract(self, experience: Experience) -> str:
        return _format_trajectory(experience)

    def _embedding_text(self, item: MemoryItem) -> str:
        return f"{item.query}\n{item.content}"

    def _ensure_embeddings(self, items: list[MemoryItem] | None = None) -> None:
        selected = items if items is not None else self.items
        if not self.embedder or not selected:
            return
        missing = [item for item in selected if not item.metadata.get("embedding")]
        if not missing:
            return
        try:
            vectors = self.embedder.embed([self._embedding_text(item) for item in missing])
        except Exception as exc:
            raise MemoryProtocolError("Memory embedding request failed while indexing items") from exc
        if len(vectors) != len(missing) or any(not vector for vector in vectors):
            raise MemoryProtocolError(
                "Memory embedding response count or vector content does not match indexed items"
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise MemoryProtocolError("Memory embedding response contains inconsistent dimensions")
        if any(
            not all(math.isfinite(float(value)) for value in vector)
            or not any(float(value) != 0.0 for value in vector)
            for vector in vectors
        ):
            raise MemoryProtocolError("Memory embedding response contains a non-finite or zero vector")
        for item, vector in zip(missing, vectors):
            item.metadata["embedding"] = [float(value) for value in vector]

    def _embed_query(self, query: str) -> list[float] | None:
        if not self.embedder:
            return None
        try:
            vectors = self.embedder.embed([query])
        except Exception as exc:
            raise MemoryProtocolError("Memory embedding request failed for retrieval query") from exc
        if len(vectors) != 1 or not vectors[0]:
            raise MemoryProtocolError("Memory embedding query response is empty or malformed")
        vector = [float(value) for value in vectors[0]]
        if not all(math.isfinite(value) for value in vector) or not any(value != 0.0 for value in vector):
            raise MemoryProtocolError("Memory embedding query response contains a non-finite or zero vector")
        return vector

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        k = max(1, int(top_k or self.top_k))
        lexical_scores = _tfidf_scores(query, [self._embedding_text(item) for item in self.items])
        self._ensure_embeddings()
        query_embedding = self._embed_query(query) if self.items else None
        for index, item in enumerate(self.items):
            lexical = lexical_scores[index] if index < len(lexical_scores) else 0.0
            semantic = _cosine(query_embedding, item.metadata.get("embedding")) if query_embedding else lexical
            item.metadata["retrieval_score"] = self.semantic_weight * semantic + self.lexical_weight * lexical
        return sorted(
            self.items, key=lambda item: float(item.metadata.get("retrieval_score", 0.0)), reverse=True,
        )[:k]

    def render(self, items: list[MemoryItem]) -> str:
        return "\n\n".join(f"[{item.memory_type}] {item.content}" for item in items)

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path or self.store_path or "memory.json")
        if target.suffix.lower() != ".json":
            target = target / "memory.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([asdict(item) for item in self.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "MemoryMethod":
        instance = cls(**kwargs)
        target = Path(path)
        if target.is_dir():
            target = target / "memory.json"
        if target.is_file():
            payload = json.loads(target.read_text(encoding="utf-8"))
            instance.items = [MemoryItem(**item) for item in payload]
            # A standalone artifact carries its content type in each item. If
            # callers do not provide the original ContentMemory config, infer
            # the buckets so load/retrieve remains a complete public roundtrip.
            if isinstance(instance, ContentMemory) and "content_types" not in kwargs:
                inferred = list(dict.fromkeys(
                    instance._normalize_type(str(item.metadata.get("content_type", item.memory_type)))
                    for item in instance.items
                    if item.metadata.get("content_type", item.memory_type)
                ))
                if inferred:
                    instance.content_types = inferred
        return instance


class NoneMemory(MemoryMethod):
    name = "none"

    def add_experience(self, experience: Experience) -> None:
        return None

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        return []

    def render(self, items: list[MemoryItem]) -> str:
        return ""


class HumanHintMemory(NoneMemory):
    name = "human_hint"


class ExpelMemory(MemoryMethod):
    name = "expel"

    def _embedding_text(self, item: MemoryItem) -> str:
        return item.content

    def add_experience(self, experience: Experience) -> None:
        if experience.success is None:
            return
        trajectory = _format_trajectory(experience)
        insight_prompt = extraction_prompt(
            "insight", experience.question, experience.answer, trajectory, experience.success,
        )
        insights = _clean_insights(_chat_text(self.llm, insight_prompt))
        if not insights and _strict_llm_protocol(self.llm):
            raise MemoryProtocolError("ExpeL insight extraction returned no valid insights")
        for insight in insights:
            self.items.append(self._new_item(
                experience,
                insight,
                "expel_insight",
                {"kind": "insight", "insight_type": "success" if experience.success else "failure"},
            ))
        if experience.success is True:
            refined = _chat_text(self.llm, extraction_prompt(
                "success_trace", experience.question, experience.answer, trajectory, True,
            )) or trajectory
            self.items.append(self._new_item(
                experience, refined, "expel_success_trace", {"kind": "success"},
            ))

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        k = max(1, int(top_k or self.top_k))
        insights = [item for item in self.items if item.metadata.get("kind") == "insight"]
        successes = [item for item in self.items if item.metadata.get("kind") == "success"]
        rows: list[MemoryItem] = []

        insight_scores = _tfidf_scores(query, [item.content for item in insights])
        for index in _top_indices(insight_scores, k) if insight_scores else []:
            source = insights[index]
            score = insight_scores[index]
            label = "Failure" if source.metadata.get("insight_type") == "failure" else "Success"
            rows.append(MemoryItem(
                id=source.id, query=source.query,
                content=f"ExpeL {label} Insight: {source.content}",
                memory_type="expel", source_ids=list(source.source_ids),
                metadata={**source.metadata, "retrieval_score": score},
            ))

        success_lexical = _tfidf_scores(query, [item.content for item in successes])
        self._ensure_embeddings(successes)
        query_embedding = self._embed_query(query) if successes else None
        success_scores: list[float] = []
        for index, source in enumerate(successes):
            lexical = success_lexical[index] if index < len(success_lexical) else 0.0
            if query_embedding:
                score = 0.3 * lexical + 0.7 * _cosine(query_embedding, source.metadata.get("embedding"))
            else:
                score = lexical
            success_scores.append(score)
        for index in _top_indices(success_scores, k) if success_scores else []:
            source = successes[index]
            rows.append(MemoryItem(
                id=source.id, query=source.query,
                content=f"ExpeL Similar successful case for '{source.query}':\n{source.content}",
                memory_type="expel", source_ids=list(source.source_ids),
                metadata={**source.metadata, "retrieval_score": success_scores[index]},
            ))

        rows.sort(key=lambda item: float(item.metadata.get("retrieval_score", 0.0)), reverse=True)
        return rows[:max(k, min(len(rows), k * 2))]

    def render(self, items: list[MemoryItem]) -> str:
        if not items:
            return "ExpeL Memory:\n(none)"
        return "ExpeL Memory:\n" + "\n".join(f"- {item.content}" for item in items)


class MemEvolveMemory(MemoryMethod):
    name = "memevolve"
    _SPECIAL_TOKENS = ("<code>", "</code>", "Thought:", "Observation:", "MessageRole.",
                       "[DESCRIPTION]:", "[STRUCTURE]:")

    def __init__(self, *args: Any, prune_threshold: float = 0.2, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.prune_threshold = float(prune_threshold)
        self._last_provided: dict[str, list[str]] = {}

    def _embedding_text(self, item: MemoryItem) -> str:
        return item.content

    def _principle_prompt(self, trajectory: str, success: bool) -> str:
        kind = "Guiding" if success else "Cautionary"
        expert = "distill generalizable wisdom" if success else "find failure root causes"
        outcome = "SUCCESS" if success else "FAILURE"
        return f"""You are an expert in analyzing interaction logs to {expert}.
Analyze the {'successful' if success else 'failed'} trajectory and extract a {kind} Principle.

A principle has two parts:
1. One concise sentence.
2. Structured triplets as JSON list: [subject, predicate, object].

[Trajectory Log]:
{trajectory}

Final Outcome: {outcome}

Output format:
[DESCRIPTION]:
<one sentence>
[STRUCTURE]:
<json list>
"""

    @staticmethod
    def _parse_principle(response: str) -> tuple[str, list[list[str]]] | None:
        if not response:
            return None
        if "[DESCRIPTION]:" not in response or "[STRUCTURE]:" not in response:
            return None
        description_part, structure_part = response.split("[STRUCTURE]:", 1)
        description = description_part.replace("[DESCRIPTION]:", "").strip()
        try:
            structure = json.loads(structure_part.replace("```json", "").replace("```", "").strip())
        except json.JSONDecodeError:
            return None
        if not description or not isinstance(structure, list):
            return None
        return description, structure

    def _compact(self, trajectory: str) -> str:
        prompt = f"""You are a Technical Incident Reporter.
Compress this log into a Single Pure-Text Narrative (Max 150 words).

Guidelines:
1. Fluid Narrative: combine trigger, corrective action, and result.
2. Start with the Trouble.
3. Keep technical precision: error names, tools, key arguments.
4. Remove arithmetic and generic internal thoughts.

Input Log:
{trajectory}

Output:
"""
        return _chat_text(self.llm, prompt) or "[(Compaction failed: Empty LLM response)]"

    @staticmethod
    def _quality(item: MemoryItem) -> float:
        success = float(item.metadata.get("success_count", 0))
        usage = float(item.metadata.get("usage_count", 0))
        return (success + 1.0) / (usage + 2.0)

    def _feedback_and_prune(self, experience: Experience) -> None:
        ids = self._last_provided.pop(experience.question, [])
        for item in self.items:
            if item.id not in ids:
                continue
            item.metadata["usage_count"] = int(item.metadata.get("usage_count", 0)) + 1
            if experience.success is True:
                item.metadata["success_count"] = int(item.metadata.get("success_count", 0)) + 1
        self.items = [item for item in self.items if self._quality(item) >= self.prune_threshold]

    def add_experience(self, experience: Experience) -> None:
        if experience.success is None:
            return
        self._feedback_and_prune(experience)
        trajectory = _format_trajectory(experience)
        parsed = self._parse_principle(_chat_text(
            self.llm, self._principle_prompt(trajectory, experience.success is True),
        ))
        if not parsed or not parsed[0].strip():
            if _strict_llm_protocol(self.llm):
                raise MemoryProtocolError("MemEvolve principle extraction returned invalid structure")
            return
        description, structure = parsed
        compressed = self._compact(trajectory)
        source = {
            "original_query": experience.question,
            "compressed_trace": compressed,
            "metadata": {"is_correct": experience.success},
        }
        self._ensure_embeddings()
        query_vector = self._embed_query(description) if self.items else None
        if query_vector:
            scored = sorted(
                ((_cosine(query_vector, item.metadata.get("embedding")), item) for item in self.items),
                key=lambda pair: pair[0], reverse=True,
            )[:self.top_k]
            for score, item in scored:
                if score <= 0.8:
                    continue
                same = _chat_text(self.llm, f"""You are a semantic analysis expert.
Determine whether two principles express the same core advice.

Principle A: "{description}"
Principle B: "{item.content}"

Answer only "Yes" or "No".
""").lower()
                if "yes" not in same and "no" not in same and _strict_llm_protocol(self.llm):
                    raise MemoryProtocolError("MemEvolve semantic merge check returned neither Yes nor No")
                if "yes" in same:
                    item.metadata.setdefault("source_trajectories", []).append(source)
                    item.metadata.setdefault("source_count", 1)
                    item.metadata["source_count"] += 1
                    return
        self.items.append(self._new_item(
            experience,
            description,
            f"memevolve_{'guiding' if experience.success else 'cautionary'}",
            {
                "principle_type": "guiding" if experience.success else "cautionary",
                "structure": structure,
                "source_trajectories": [source],
                "source_count": 1,
                "usage_count": 0,
                "success_count": 0,
            },
        ))

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        k = max(1, int(top_k or self.top_k))
        self._ensure_embeddings()
        query_vector = self._embed_query(query) if self.items else None
        if query_vector:
            scores = [_cosine(query_vector, item.metadata.get("embedding")) for item in self.items]
        else:
            scores = _tfidf_scores(query, [item.content for item in self.items])
        selected: list[tuple[MemoryItem, float]] = []
        for index in _top_indices(scores, k) if scores else []:
            item = self.items[index]
            if any(token in item.content for token in self._SPECIAL_TOKENS):
                continue
            selected.append((item, scores[index]))
        if selected:
            self._last_provided[query] = [item.id for item, _ in selected]
        packaged: list[MemoryItem] = []
        for item, score in selected:
            parts = [
                f"--- Retrieved {str(item.metadata.get('principle_type', 'principle')).capitalize()} Principle "
                f"(Similarity: {score:.4f}, Quality Score: {self._quality(item):.4f}) ---",
                f"[Principle]\n{item.content}",
            ]
            structure = item.metadata.get("structure") or []
            if structure:
                parts.append("\n[Structure/Pattern]")
                for triplet in structure:
                    if isinstance(triplet, list) and len(triplet) == 3:
                        parts.append(f"- {triplet[0]} {triplet[1]} {triplet[2]}")
            positive, negative = [], []
            for source in reversed(item.metadata.get("source_trajectories") or []):
                trace = str(source.get("compressed_trace", ""))
                if (source.get("metadata") or {}).get("is_correct") is True and len(positive) < 1:
                    positive.append(trace)
                elif (source.get("metadata") or {}).get("is_correct") is False and len(negative) < 1:
                    negative.append(trace)
            if positive:
                parts.append("\n[Successful Examples]")
                parts.extend(f"Example {index}:\n{trace}" for index, trace in enumerate(reversed(positive), 1))
            if negative:
                parts.append("\n[Cautionary Examples]")
                parts.extend(f"Example {index}:\n{trace}" for index, trace in enumerate(reversed(negative), 1))
            parts.append("--- End of Retrieved Memory ---")
            packaged.append(MemoryItem(
                id=item.id, query=item.query, content="\n\n".join(parts), memory_type="memevolve",
                source_ids=list(item.source_ids), metadata={**item.metadata, "retrieval_score": score},
            ))
        return packaged

    def render(self, items: list[MemoryItem]) -> str:
        if not items:
            return "Evolver Principles:\n(none)"
        return "Evolver Principles:\n\n" + "\n\n".join(item.content for item in items)


class AWMMemory(MemoryMethod):
    name = "awm"

    def _embedding_text(self, item: MemoryItem) -> str:
        return item.content

    def add_experience(self, experience: Experience) -> None:
        if experience.success is not True:
            return
        trajectory = _format_trajectory(experience)
        prompt = f"""You are an expert analyst for tasks.
Your goal is to extract a generic, reusable workflow from the specific execution trajectory provided below.

Guidelines:
1. Abstraction: convert specific values into descriptive variable names when possible.
2. Invariance: keep logical steps and tool names invariant.
3. Format: output strictly valid JSON.

Output JSON Schema:
{{
  "workflow": "The concise text summary of the steps (under 200 words)"
}}

Task Query:
{experience.question}

Trajectory to analyze:
{trajectory}
"""
        parsed = _extract_json_object(_chat_text(self.llm, prompt))
        workflow = str((parsed or {}).get("workflow", "")).strip()
        if not workflow:
            if _strict_llm_protocol(self.llm):
                raise MemoryProtocolError("AWM workflow extraction returned invalid JSON")
            workflow = trajectory
        self.items.append(self._new_item(
            experience, f"Query: {experience.question}\nWorkflow: {workflow}", "awm",
        ))

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        k = max(1, int(top_k or self.top_k))
        self._ensure_embeddings()
        query_vector = self._embed_query(query) if self.items else None
        scores = (
            [_cosine(query_vector, item.metadata.get("embedding")) for item in self.items]
            if query_vector else _tfidf_scores(query, [item.content for item in self.items])
        )
        selected = []
        for index in _top_indices(scores, k) if scores else []:
            item = self.items[index]
            item.metadata["retrieval_score"] = scores[index]
            selected.append(item)
        return selected

    def render(self, items: list[MemoryItem]) -> str:
        if not items:
            return "AWM:\n(none)"
        return "Agent Workflow Memory:\n" + "\n".join(
            f"{index}. {item.content}" for index, item in enumerate(items, 1)
        )


class AgentKBMemory(MemoryMethod):
    name = "agentkb"
    _QUERY_PROMPT = """Extract key information from user query to construct efficient search terms for retrieving the most relevant results.

Requirements:
1. Analyze the user's question to identify core concepts, terminology, and keywords
2. Extract contextual information and constraints that may impact search quality
3. Break down complex questions into searchable components
4. Identify the domain, subject matter, and specific needs of the question

Output format:
<core concepts or topics of the question>

Ensure search terms are specific enough to retrieve relevant information while maintaining sufficient breadth to capture related cases.
Combine technical terminology with everyday expressions to optimize search effectiveness.

Here is the user query:
{query}"""

    def _embedding_text(self, item: MemoryItem) -> str:
        return item.query

    def add_experience(self, experience: Experience) -> None:
        if experience.success is not True:
            return
        trajectory = _format_trajectory(experience)
        prompt = f"""You are an expert AI agent trainer analyzing a successful task execution to extract high-quality memory patterns for future similar tasks.

TASK ANALYSIS:
Question: {experience.question}

Execution Trajectory:
{trajectory}

Final Result: {experience.answer or "Task completed successfully"}

MEMORY EXTRACTION INSTRUCTIONS:
Extract structured memory components that capture the strategic thinking and methodological approaches used in this successful execution. Focus on actionable insights, specific techniques, and reusable patterns.

Please provide detailed analysis in the following JSON format:

{{
    "agent_planning": "Detailed strategic planning approach with numbered steps, decision-making criteria, tool selection rationale, and problem decomposition strategy",
    "search_agent_planning": "Comprehensive search strategy including query formulation techniques, source prioritization methods, information extraction approaches, and result validation processes",
    "agent_experience": "Key lessons learned, successful methodologies, best practices discovered, error avoidance strategies, and general principles that can guide future similar tasks",
    "search_agent_experience": "Search-specific insights including effective query patterns, reliable source types, information validation techniques, and data processing approaches"
}}

QUALITY REQUIREMENTS:
1. Each field must contain substantial, specific content (minimum 2-3 detailed sentences)
2. Focus on ACTIONABLE strategies and CONCRETE methodologies, not generic descriptions
3. Include specific decision points, tool choices, and reasoning patterns
4. Emphasize successful techniques that led to task completion
5. Extract transferable knowledge that applies to similar problem types
6. Use professional, instructional language as if training another agent
7. Include specific examples or patterns where applicable

Return ONLY the JSON object with no additional text or explanations."""
        parsed = _extract_json_object(_chat_text(self.llm, prompt))
        fields = ("agent_planning", "search_agent_planning", "agent_experience", "search_agent_experience")
        if not parsed or any(len(str(parsed.get(key, "")).strip()) < 50 for key in fields):
            if _strict_llm_protocol(self.llm):
                raise MemoryProtocolError("AgentKB extraction returned an invalid workflow package")
            return
        metadata = {key: str(parsed[key]).strip() for key in fields}
        self.items.append(self._new_item(
            experience, json.dumps(metadata, ensure_ascii=False), "agentkb", metadata,
        ))

    def _hybrid_results(self, query: str, k: int) -> list[tuple[MemoryItem, float]]:
        lexical = _tfidf_scores(query, [item.query for item in self.items])
        self._ensure_embeddings()
        query_vector = self._embed_query(query) if self.items else None
        semantic = (
            [_cosine(query_vector, item.metadata.get("embedding")) for item in self.items]
            if query_vector else lexical
        )
        scores: dict[int, float] = {}
        for index in _top_indices(lexical, k * 2) if lexical else []:
            scores[index] = scores.get(index, 0.0) + 0.5 * lexical[index]
        for index in _top_indices(semantic, k * 2) if semantic else []:
            scores[index] = scores.get(index, 0.0) + 0.5 * semantic[index]
        return [
            (self.items[index], score)
            for index, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]
        ]

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        k = max(1, int(top_k or self.top_k))
        refined = _chat_text(self.llm, self._QUERY_PROMPT.format(query=query)) or query
        results = self._hybrid_results(refined, k)
        if not results:
            return []

        planning_blocks: list[str] = []
        experience_blocks: list[str] = []
        for index, (item, score) in enumerate(results, 1):
            plan = str(item.metadata.get("agent_planning", ""))
            search_plan = str(item.metadata.get("search_agent_planning", ""))
            planning_blocks.append(
                f"Similar task:\n{item.query}\nSuggestions:\n{' '.join(x for x in (plan, search_plan) if x)}"
            )
            experience_blocks.append(
                f"Source {index} (Score: {score:.3f}):\nQuery: {item.query}\n"
                f"Agent Experience: {item.metadata.get('agent_experience', '')}\n"
                f"Search Experience: {item.metadata.get('search_agent_experience', '')}"
            )
        student_prompt = f"""Analyze similar tasks and past experiences to generate concise, actionable suggestions for improving the current plan. Based on the patterns identified in relevant tasks and insights from the knowledge base, provide specific recommendations.

**Key Requirements:**
1. Focus exclusively on technical/behavioral improvements derived from similar task patterns and experience.
2. Provide root-cause solutions and implementation strategies based on past successes.
3. Provide 2-3 specific suggestions only.
4. Format output strictly as:
   1. [Specific suggestion 1]
   2. [Specific suggestion 2]
   ...
5. Use gentle, suggestive language rather than directive commands.
No headings, explanations, or markdown.

**Current Task:** {query}

**You can refer to similar tasks, plans, and corresponding experience to provide your suggestions:**
{chr(10).join(planning_blocks)}"""
        teacher_prompt = f"""You are an experienced AI agent teacher synthesizing multiple experience entries to provide unified operational guidance.

Current Task: {query}

Retrieved Experience Entries ({len(results)} sources):
{chr(10).join(experience_blocks)}

Based on ALL the matched experience above, synthesize cohesive, unified operational guidance for the agent. Your guidance should:

1. Integrate techniques and methods from all sources
2. Combine common pitfalls and best practices across sources
3. Provide specific, actionable execution tips

Requirements:
- Be specific and comprehensive (2-3 sentences)
- Focus on detailed operations and practical techniques
- Present a unified perspective synthesizing all sources
- Provide concrete, actionable suggestions
- Help refine and improve the current approach based on collective experience
- Use gentle, suggestive language rather than directive commands.
Provide only the synthesized guidance text with no additional explanations or source references."""
        student = _chat_text(self.llm, student_prompt) or " ".join(planning_blocks)
        teacher = _chat_text(self.llm, teacher_prompt) or " ".join(experience_blocks)
        content = f"AGENT-KB Student Guidance:\n{student}\n\nAGENT-KB Teacher Guidance:\n{teacher}"
        return [MemoryItem(
            id="agentkb:" + ":".join(item.id for item, _ in results),
            query=query,
            content=content,
            memory_type="agentkb",
            source_ids=[source_id for item, _ in results for source_id in item.source_ids],
            metadata={
                "source_memory_ids": [item.id for item, _ in results],
                "retrieval_score": sum(score for _, score in results) / len(results),
                "refined_query": refined,
            },
        )]

    def render(self, items: list[MemoryItem]) -> str:
        if not items:
            return "Agent-KB:\n(none)"
        return "Agent-KB Guidance:\n\n" + "\n\n".join(item.content for item in items)


class ContentMemory(MemoryMethod):
    name = "content_memory"

    def __init__(
        self,
        content_types: list[str] | None = None,
        *,
        selection_topk_multiplier: float = 1.0,
        balance_top_k_by_type: bool = True,
        min_per_type: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.content_types = list(dict.fromkeys(
            self._normalize_type(item) for item in (content_types or ["compact_trajectory"])
        ))
        self.selection_topk_multiplier = max(1.0, float(selection_topk_multiplier))
        self.balance_top_k_by_type = bool(balance_top_k_by_type)
        self.min_per_type = max(1, int(min_per_type))

    @staticmethod
    def _normalize_type(content_type: str) -> str:
        return {
            "summary": "compact_trajectory",
            "workflow_memory": "workflow",
        }.get(str(content_type).lower(), str(content_type).lower())

    def _embedding_text(self, item: MemoryItem) -> str:
        return item.content

    def _append_content(self, experience: Experience, content_type: str, content: str,
                        group_id: str) -> None:
        normalized = content.strip()
        if len(normalized) < 10:
            return
        if any(
            self._normalize_type(str(
                item.metadata.get("content_type", item.memory_type)
            )) == content_type
            and item.content.lower() == normalized.lower()
            for item in self.items
        ):
            return
        self.items.append(self._new_item(
            experience,
            normalized,
            content_type,
            {"content_type": content_type, "memory_group_id": group_id},
        ))

    def add_experience(self, experience: Experience) -> None:
        trajectory = _format_trajectory(experience)
        summary = _chat_text(self.llm, content_extraction_prompt(
            "compact_trajectory", experience.question, experience.answer, trajectory, experience.success,
        )) or trajectory
        if len(summary) > 1200:
            summary = summary[:1200].rstrip() + " ..."
        group_id = uuid.uuid4().hex
        for content_type in self.content_types:
            if content_type == "compact_trajectory":
                self._append_content(experience, content_type, summary, group_id)
            elif content_type in {"insight", "insight_success", "insight_failure"}:
                if content_type == "insight_success" and experience.success is not True:
                    continue
                if content_type == "insight_failure" and experience.success is not False:
                    continue
                prompt = content_extraction_prompt(
                    "insight", experience.question, experience.answer, trajectory, experience.success,
                )
                insights = _clean_insights(_chat_text(self.llm, prompt))
                if not insights and _strict_llm_protocol(self.llm):
                    raise MemoryProtocolError(
                        f"Content-memory {content_type} extraction returned no valid insights"
                    )
                for insight in insights:
                    self._append_content(experience, content_type, insight, group_id)
            elif content_type in {"principle", "principle_success", "principle_failure"}:
                if experience.success is None:
                    continue
                if content_type == "principle_success" and experience.success is not True:
                    continue
                if content_type == "principle_failure" and experience.success is not False:
                    continue
                prompt = content_extraction_prompt(
                    "principle", experience.question, experience.answer, trajectory, experience.success,
                )
                principle = _chat_text(self.llm, prompt)
                if not principle:
                    head = summary.splitlines()[0] if summary.splitlines() else summary
                    principle = f"{'Do' if experience.success is True else 'Avoid'}: {head}"
                self._append_content(experience, content_type, principle, group_id)
            elif content_type == "summary_success" and experience.success is True:
                self._append_content(experience, content_type, summary, group_id)
            elif content_type == "summary_failure" and experience.success is False:
                self._append_content(experience, content_type, summary, group_id)
            elif content_type == "raw":
                self._append_content(experience, content_type, trajectory, group_id)
            elif content_type == "success_trace" and experience.success is True:
                prompt = content_extraction_prompt(
                    "success_trace", experience.question, experience.answer, trajectory, True,
                )
                self._append_content(experience, content_type, _chat_text(self.llm, prompt) or trajectory, group_id)
            elif content_type == "workflow" and experience.success is True:
                prompt = content_extraction_prompt(
                    "workflow", experience.question, experience.answer, trajectory, True,
                )
                workflow = _chat_text(self.llm, prompt) or trajectory
                self._append_content(
                    experience, content_type,
                    f"Query: {experience.question}\nWorkflow:\n{workflow}", group_id,
                )

    def _format_content(self, item: MemoryItem) -> str:
        content_type = self._normalize_type(str(item.metadata.get("content_type", item.memory_type)))
        if content_type == "compact_trajectory":
            return f"Summary: {item.content}"
        if content_type == "insight":
            label = "Success" if item.metadata.get("success") is True else "Failure"
            return f"{label} Insight: {item.content}"
        if content_type == "insight_success":
            return f"Success Insight: {item.content}"
        if content_type == "insight_failure":
            return f"Failure Insight: {item.content}"
        if content_type in {"principle", "principle_success", "principle_failure"}:
            if content_type == "principle_success":
                return f"Guiding Principle from successful trajectories: {item.content}"
            if content_type == "principle_failure":
                return f"Cautionary Principle from failed trajectories: {item.content}"
            label = "Guiding" if item.metadata.get("success") is True else "Cautionary"
            return f"{label} Principle: {item.content}"
        if content_type == "summary_success":
            return f"Summary from successful trajectories: {item.content}"
        if content_type == "summary_failure":
            return f"Summary from failed trajectories: {item.content}"
        if content_type == "raw":
            return f"Raw Trace: {item.content}"
        if content_type == "success_trace":
            return f"Similar successful case for '{item.query}':\n{item.content}"
        if content_type == "workflow":
            return f"Workflow Memory:\n{item.content}"
        return item.content

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        k = max(1, int(top_k or self.top_k))
        self._ensure_embeddings()
        query_vector = self._embed_query(query) if self.items else None
        candidates: list[tuple[MemoryItem, float]] = []
        for content_type in self.content_types:
            bucket = [
                item for item in self.items
                if self._normalize_type(str(item.metadata.get("content_type", item.memory_type)))
                == content_type
            ]
            lexical = _tfidf_scores(query, [item.content for item in bucket])
            scores = []
            for index, item in enumerate(bucket):
                lexical_score = lexical[index] if index < len(lexical) else 0.0
                semantic_score = (
                    _cosine(query_vector, item.metadata.get("embedding"))
                    if query_vector else lexical_score
                )
                scores.append(self.lexical_weight * lexical_score + self.semantic_weight * semantic_score)
            candidates.extend((bucket[index], scores[index]) for index in _top_indices(scores, k) if scores)
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        cap = max(k, int(math.ceil(k * self.selection_topk_multiplier)))
        if self.balance_top_k_by_type and len(self.content_types) > 1:
            queues = {
                content_type: [
                    pair for pair in candidates
                    if self._normalize_type(str(
                        pair[0].metadata.get("content_type", pair[0].memory_type)
                    )) == content_type
                ]
                for content_type in self.content_types
            }
            chosen: list[tuple[MemoryItem, float]] = []
            per_type_target = max(
                self.min_per_type,
                (cap + len(self.content_types) - 1) // len(self.content_types),
            )
            # Round-robin selection gives each configured type an opportunity
            # while keeping the *total* number of returned items within cap.
            # The previous implementation could exceed cap when top_k=1 and
            # more than one content type was configured.
            for _ in range(per_type_target):
                for content_type in self.content_types:
                    if len(chosen) >= cap:
                        break
                    if queues[content_type]:
                        chosen.append(queues[content_type].pop(0))
                if len(chosen) >= cap:
                    break
            if len(chosen) < cap:
                selected_ids = {(item.memory_type, item.id) for item, _ in chosen}
                for pair in candidates:
                    key = (pair[0].memory_type, pair[0].id)
                    if key in selected_ids:
                        continue
                    chosen.append(pair)
                    selected_ids.add(key)
                    if len(chosen) >= cap:
                        break
            candidates = chosen
        else:
            candidates = candidates[:cap]
        return [MemoryItem(
            id=item.id, query=item.query, content=self._format_content(item),
            memory_type=item.memory_type, source_ids=list(item.source_ids),
            metadata={**item.metadata, "retrieval_score": score},
        ) for item, score in candidates]

    def render(self, items: list[MemoryItem]) -> str:
        if not items:
            return "TABER Ablation Memory:\n(none)"
        return "TABER Ablation Memory Guidance:\n" + "\n".join(
            f"- {item.content}" for item in items
        )


MEMORY_REGISTRY = {
    "none": NoneMemory,
    "human_hint": HumanHintMemory,
    "expel": ExpelMemory,
    "memevolve": MemEvolveMemory,
    "evolver": MemEvolveMemory,
    "awm": AWMMemory,
    "agentkb": AgentKBMemory,
    "content_memory": ContentMemory,
}


def create_memory(
    method: str,
    *,
    top_k: int = 2,
    store_path: str | None = None,
    content_types: list[str] | None = None,
    llm: LLMService | None = None,
    embedder: Any | None = None,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
    selection_topk_multiplier: float = 1.0,
    balance_top_k_by_type: bool = True,
    min_per_type: int = 1,
) -> MemoryMethod:
    key = method.lower().replace("agent_kb", "agentkb")
    cls = MEMORY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown memory method: {method}")
    kwargs: dict[str, Any] = {
        "top_k": top_k,
        "store_path": store_path,
        "llm": llm,
        "embedder": embedder,
        "semantic_weight": semantic_weight,
        "lexical_weight": lexical_weight,
    }
    if cls is ContentMemory:
        kwargs.update({
            "content_types": content_types,
            "selection_topk_multiplier": selection_topk_multiplier,
            "balance_top_k_by_type": balance_top_k_by_type,
            "min_per_type": min_per_type,
        })
    return cls(**kwargs)
