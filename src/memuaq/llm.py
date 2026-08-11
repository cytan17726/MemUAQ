from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import ModelProfile


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: Any = None


class ChatClient(Protocol):
    def chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> ChatResponse: ...

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]: ...


class OpenAICompatibleClient:
    """Thin adapter around the OpenAI SDK and compatible endpoints."""

    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install memuaq with the OpenAI dependency first") from exc
            kwargs: dict[str, Any] = {"api_key": self.profile.resolved_api_key()}
            if self.profile.base_url:
                kwargs["base_url"] = self.profile.base_url
            kwargs["timeout"] = self.profile.timeout_seconds
            kwargs["max_retries"] = self.profile.max_retries
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        params = {
            "model": self.profile.model,
            "messages": list(messages),
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
        }
        params.update(kwargs)
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        message = choice.message
        calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            calls.append({
                "id": getattr(call, "id", ""),
                "name": getattr(function, "name", "") if function else "",
                "arguments": getattr(function, "arguments", "") if function else "",
            })
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
        return ChatResponse(
            content=str(getattr(message, "content", "") or ""),
            tool_calls=calls,
            usage=usage_dict,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        model = kwargs.pop("model", self.profile.model)
        response = self.client.embeddings.create(model=model, input=list(texts), **kwargs)
        return [list(item.embedding) for item in response.data]


class LLMService:
    def __init__(self, client: ChatClient, profile: ModelProfile):
        self.client = client
        self.profile = profile

    def chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        return self.client.chat(messages, **kwargs)

    def complete_json(self, messages: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        response = self.chat(messages, response_format={"type": "json_object"})
        import json
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


class ScriptedClient:
    """Deterministic offline client used by tests and the smoke recipe."""

    def __init__(self, responses: Sequence[str] | None = None):
        self.responses = list(responses or ["Thought 1: finish safely.\nAction 1: Finish[I cannot answer with the available evidence.]"])
        self.index = 0
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls.append([dict(message) for message in messages])
        content = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return ChatResponse(content=content, usage={"prompt_tokens": 0, "completion_tokens": 0})

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        return [[float(len(text)), float(sum(map(ord, text)) % 997)] for text in texts]
