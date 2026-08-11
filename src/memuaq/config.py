from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a MemUAQ configuration is invalid."""


_ENV_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def read_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ConfigError(f"Configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a mapping: {file_path}")
    return _expand(payload)


def _check_keys(payload: dict[str, Any], allowed: set[str], source: Path) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys {unknown} in {source}")


@dataclass(frozen=True)
class ModelProfile:
    role: str
    provider: str = "openai_compatible"
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 5
    concurrency: int = 1
    embedding_dimension: int | None = None

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    path: str
    train_path: str | None = None
    test_path: str | None = None
    format: str = "json"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    models: dict[str, str]
    dataset: str
    agent: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def load_model_profile(path: str | Path, *, role: str | None = None) -> ModelProfile:
    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    _check_keys(raw, {"schema_version", "role", "provider", "model", "base_url",
                      "api_key", "api_key_env", "temperature", "max_tokens",
                      "timeout_seconds", "max_retries", "concurrency",
                      "embedding_dimension"}, source)
    selected_role = role or str(raw.get("role", "agent"))
    required = ("model",)
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ConfigError(f"Missing model fields {missing} in {source}")
    if float(raw.get("temperature", 0.0)) < 0 or int(raw.get("max_tokens", 2048)) <= 0:
        raise ConfigError(f"Invalid generation settings in {source}")
    base_url = raw.get("base_url")
    if isinstance(base_url, str) and base_url.startswith("${"):
        base_url = _expand(base_url)
    return ModelProfile(
        role=selected_role,
        provider=str(raw.get("provider", "openai_compatible")),
        model=str(raw["model"]),
        base_url=str(base_url) if base_url else None,
        api_key=str(raw["api_key"]) if raw.get("api_key") else None,
        api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 2048)),
        timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
        max_retries=int(raw.get("max_retries", 5)),
        concurrency=max(1, int(raw.get("concurrency", 1))),
        embedding_dimension=(
            int(raw["embedding_dimension"]) if raw.get("embedding_dimension") else None
        ),
    )


def load_dataset_profile(path: str | Path) -> DatasetProfile:
    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    _check_keys(raw, {"schema_version", "name", "path", "train_path", "test_path", "format"}, source)
    if not raw.get("name"):
        raise ConfigError(f"Missing dataset name in {source}")
    def resolve(value: Any) -> str | None:
        if not value:
            return None
        candidate = Path(str(value)).expanduser()
        return str((source.parent / candidate).resolve() if not candidate.is_absolute() else candidate)
    main_path = resolve(raw.get("path") or raw.get("test_path"))
    if not main_path:
        raise ConfigError(f"Dataset path or test_path is required in {source}")
    return DatasetProfile(name=str(raw["name"]), path=main_path,
                          train_path=resolve(raw.get("train_path")),
                          test_path=resolve(raw.get("test_path")),
                          format=str(raw.get("format", "json")))


def _resolve_ref(reference: str, source: Path) -> Path:
    candidate = Path(reference).expanduser()
    return (source.parent / candidate).resolve() if not candidate.is_absolute() else candidate


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = read_yaml(source)
    _check_keys(raw, {"schema_version", "name", "models", "dataset", "agent", "environment",
                      "memory", "evaluation", "run"}, source)
    if not raw.get("name"):
        raise ConfigError(f"Missing name in {source}")
    models = raw.get("models")
    if not isinstance(models, dict) or "agent" not in models:
        raise ConfigError(f"models.agent is required in {source}")
    dataset = raw.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ConfigError(f"dataset is required in {source}")
    for role_name, reference in models.items():
        resolved = _resolve_ref(str(reference), source)
        if not resolved.is_file():
            raise ConfigError(f"models.{role_name} not found: {resolved}")
    dataset_path = _resolve_ref(dataset, source)
    if not dataset_path.is_file():
        raise ConfigError(f"dataset profile not found: {dataset_path}")
    memory = dict(raw.get("memory") or {})
    method = str(memory.get("method", "none")).lower().replace("agent_kb", "agentkb")
    if method not in {"none", "human_hint", "expel", "memevolve", "evolver", "awm", "agentkb", "content_memory"}:
        raise ConfigError(f"Unknown memory.method {method!r} in {source}")
    if method not in {"none", "human_hint"} and "embedding" not in models:
        raise ConfigError(
            f"models.embedding is required for memory.method={method!r} in {source}"
        )
    if int(memory.get("top_k", 2)) <= 0:
        raise ConfigError(f"memory.top_k must be positive in {source}")
    train_limit = memory.get("train_limit", 200)
    if train_limit is not None and int(train_limit) <= 0:
        raise ConfigError(f"memory.train_limit must be positive in {source}")
    # Sec.5 uses a fixed total top-2 retrieval budget.  The reproduced
    # baseline adapters retain their method-specific wider candidate pool,
    # while the unified content-memory pipeline defaults to one final top-2
    # pool and balanced type exposure for pairwise settings.
    selection_multiplier_default = 1.0 if method == "content_memory" else 2.0
    memory.setdefault("selection_topk_multiplier", selection_multiplier_default)
    if method == "content_memory":
        memory.setdefault("balance_top_k_by_type", True)
    if float(memory.get("selection_topk_multiplier", selection_multiplier_default)) < 1:
        raise ConfigError(f"memory.selection_topk_multiplier must be >= 1 in {source}")
    if int(memory.get("min_per_type", 1)) <= 0:
        raise ConfigError(f"memory.min_per_type must be positive in {source}")
    evaluation = dict(raw.get("evaluation") or {})
    test_limit = evaluation.get("test_limit", 400)
    if test_limit is not None and int(test_limit) <= 0:
        raise ConfigError(f"evaluation.test_limit must be positive in {source}")
    source_dataset = memory.get("source_dataset")
    if source_dataset:
        resolved_source_dataset = _resolve_ref(str(source_dataset), source)
        if not resolved_source_dataset.is_file():
            raise ConfigError(f"memory.source_dataset not found: {resolved_source_dataset}")
        memory["source_dataset"] = str(resolved_source_dataset)
    channel = str(memory.get("channel", "combined")).lower()
    if channel not in {"combined", "guidance_only", "trajectory_only"}:
        raise ConfigError(f"Unknown memory.channel {channel!r} in {source}")
    content_types = memory.get("content_types")
    if content_types is not None:
        if not isinstance(content_types, list) or not content_types:
            raise ConfigError(f"memory.content_types must be a non-empty list in {source}")
        allowed_content_types = {
            "compact_trajectory", "insight", "insight_success", "insight_failure",
            "principle", "principle_success", "principle_failure", "success_trace", "workflow",
            "summary", "summary_success", "summary_failure", "workflow_memory", "raw",
        }
        unknown_content_types = sorted(
            set(str(item).lower() for item in content_types) - allowed_content_types
        )
        if unknown_content_types:
            raise ConfigError(
                f"Unknown memory.content_types {unknown_content_types} in {source}"
            )
    source_name = str(memory.get("experience_source", "mixed")).lower()
    if source_name not in {"mixed", "both", "success_only", "failure_only", "success", "failure"}:
        raise ConfigError(f"Unknown memory.experience_source {source_name!r} in {source}")
    return ExperimentConfig(
        name=str(raw["name"]),
        models={str(key): str(_resolve_ref(value, source)) for key, value in models.items()},
        dataset=str(_resolve_ref(dataset, source)),
        agent=dict(raw.get("agent") or {}),
        environment=dict(raw.get("environment") or {}),
        memory=memory,
        evaluation=evaluation,
        run=dict(raw.get("run") or {}),
        source_path=source,
    )


def redact(value: Any) -> Any:
    """Return a serializable copy with credential-looking fields removed."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in {"api_key", "authorization", "token", "password"}:
                result[key] = "[REDACTED]" if item else item
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return copy.deepcopy(value)
