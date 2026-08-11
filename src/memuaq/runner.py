from __future__ import annotations

import json
import hashlib
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .agent import WikiReactAgent
from .config import ConfigError, ExperimentConfig, load_dataset_profile, load_experiment_config, load_model_profile, read_yaml, redact
from .data import DatasetRecord, load_dataset, write_jsonl
from .environment import WikipediaClient, WikiReactEnvironment
from .evaluation import aggregate_metrics, compute_metrics, judge_file, load_jsonl, training_success
from .llm import LLMService, OpenAICompatibleClient, ScriptedClient
from .memory import Experience, create_memory
from .prompts import HUMAN_HINT, SYSTEM_PROMPT, UAQ_JUDGE_PROMPT


class ExperimentRunError(RuntimeError):
    """Raised when agent/environment execution fails within an experiment."""


def _prompt_hashes() -> dict[str, str]:
    return {
        "agent": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "human_hint": hashlib.sha256(HUMAN_HINT.encode()).hexdigest()[:16],
        "uaq_judge": hashlib.sha256(UAQ_JUDGE_PROMPT.encode()).hexdigest()[:16],
    }


def _model_service(profile_path: str, role: str) -> LLMService:
    profile = load_model_profile(profile_path, role=role)
    if profile.provider != "fake" and not profile.resolved_api_key():
        raise ConfigError(f"No API key configured for {role} model profile: {profile_path}")
    client = ScriptedClient() if profile.provider == "fake" else OpenAICompatibleClient(profile)
    return LLMService(client, profile)


def _allow_heuristic_protocol(config: ExperimentConfig, role: str) -> bool:
    """Permit fallbacks only for explicitly offline/fake test configurations."""
    if bool(config.evaluation.get("allow_heuristic_judge", False)):
        return True
    profile_path = config.models.get(role)
    if not profile_path:
        return False
    return load_model_profile(profile_path, role=role).provider == "fake"


def _project_root(config: ExperimentConfig) -> Path:
    start = config.source_path.parent if config.source_path else Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _artifact_path(config: ExperimentConfig, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (_project_root(config) / path).resolve())


def _resolve_output(config: ExperimentConfig, explicit: str | None = None) -> Path:
    root = Path(explicit or config.run.get("output_root", "outputs"))
    if not root.is_absolute():
        root = _project_root(config) / root
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    target = root / config.name / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _dataset(config: ExperimentConfig, split: str) -> list[DatasetRecord]:
    profile = load_dataset_profile(config.dataset)
    if split == "train":
        if not profile.train_path:
            raise ConfigError(
                f"Dataset profile {config.dataset} has no train_path; "
                "memory construction requires an explicit training split."
            )
        path = profile.train_path
    elif split == "test":
        path = profile.test_path or profile.path
    else:
        path = profile.path
    return load_dataset(path, fmt=profile.format)


def _dataset_from_profile(config: ExperimentConfig, profile_path: str, split: str) -> list[DatasetRecord]:
    profile = load_dataset_profile(_artifact_path(config, profile_path) or profile_path)
    if split == "train":
        if not profile.train_path:
            raise ConfigError(
                f"Dataset profile {profile_path} has no train_path; "
                "memory construction requires an explicit training split."
            )
        path = profile.train_path
    elif split == "test":
        path = profile.test_path or profile.path
    else:
        path = profile.path
    return load_dataset(path, fmt=profile.format)


def _environment(config: ExperimentConfig) -> WikiReactEnvironment:
    env = config.environment
    cache = env.get("cache") or {}
    cache_path = (
        _artifact_path(config, cache.get("path"))
        if cache.get("enabled") and cache.get("path") else None
    )
    client = WikipediaClient(language=str(env.get("language", "en")), cache_path=cache_path,
                             timeout=float(env.get("timeout_seconds", 30)))
    return WikiReactEnvironment(client, max_steps=int(config.agent.get("max_steps", 10)))


def _build_agent(config: ExperimentConfig, memory, *, role: str = "agent") -> WikiReactAgent:
    llm = _model_service(config.models[role], role)
    agent = WikiReactAgent(llm, _environment(config), memory,
                          max_steps=int(config.agent.get("max_steps", 10)),
                          human_hint=(
                              bool(config.agent.get("human_hint", False))
                              or memory.name == "human_hint"
                          ))
    # The original taber agents pass their own LLM service to memory stores;
    # extraction, refinement, and online success judgments therefore use the
    # same model profile as the acting agent.
    memory.llm = llm
    return agent


def _new_memory(config: ExperimentConfig, method: str, store_path: str | None):
    memory_llm = _model_service(config.models.get("memory", config.models["agent"]), "memory")
    embedder = None
    if "embedding" in config.models:
        embedding_profile = load_model_profile(config.models["embedding"], role="embedding")
        embedder = ScriptedClient() if embedding_profile.provider == "fake" else OpenAICompatibleClient(embedding_profile)
    return create_memory(method, top_k=int(config.memory.get("top_k", 2)), store_path=store_path,
                         content_types=config.memory.get("content_types"), llm=memory_llm,
                         embedder=embedder,
                         semantic_weight=float(config.memory.get("semantic_weight", 0.7)),
                         lexical_weight=float(config.memory.get("lexical_weight", 0.3)),
                         selection_topk_multiplier=float(config.memory.get("selection_topk_multiplier", 1.0)),
                         balance_top_k_by_type=bool(config.memory.get("balance_top_k_by_type", True)),
                         min_per_type=int(config.memory.get("min_per_type", 1)))


def _experience_trajectory(record: DatasetRecord, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert MemUAQ's trace schema to the explicit trajectory used by taber."""
    items: list[dict[str, Any]] = [{"type": "question", "content": record.question}]
    guidance = str(result.get("memory_guidance", "")).strip()
    if guidance:
        items.append({"type": "memory_begin_guidance", "content": guidance})
    for step in result.get("trace", []) if isinstance(result.get("trace"), list) else []:
        if not isinstance(step, dict):
            continue
        assistant = str(step.get("assistant", "")).strip()
        if assistant:
            thought = re.search(
                r"(?ims)^\s*Thought(?:\s+\d+)?\s*:\s*(.*?)"
                r"(?=^\s*Action(?:\s+\d+)?\s*:|\Z)",
                assistant,
            )
            if thought:
                items.append({"type": "thought", "content": thought.group(1).strip()})
            else:
                items.append({"type": "assistant", "content": assistant})
        action = str(step.get("action", "")).strip()
        if action:
            items.append({"type": "action", "content": action})
        observation = str(step.get("observation", "")).strip()
        if observation:
            items.append({"type": "observation", "content": observation})
    prediction = str(result.get("prediction", "")).strip()
    if prediction:
        items.append({"type": "final_answer", "content": prediction})
    return items


def _balanced_memory_items(items: list[Any], source: str) -> list[Any]:
    """Apply the original post-build success/failure split per memory type.

    The upstream balance utility operates on each type-specific store after the
    normal online update pass.  Keeping that ordering matters: later training
    examples must still be able to retrieve memories created by earlier ones.
    """
    normalized = {"success": "success_only", "failure": "failure_only", "both": "mixed"}.get(
        str(source).lower(), str(source).lower(),
    )
    if normalized not in {"success_only", "failure_only", "mixed"}:
        return items

    by_type: dict[str, list[Any]] = {}
    for item in items:
        by_type.setdefault(str(getattr(item, "memory_type", "")), []).append(item)

    selected: list[Any] = []
    for bucket in by_type.values():
        successes = [item for item in bucket if item.metadata.get("success") is True]
        failures = [item for item in bucket if item.metadata.get("success") is False]
        budget = min(len(successes), len(failures))
        if normalized == "success_only":
            selected.extend(successes[:budget])
        elif normalized == "failure_only":
            selected.extend(failures[:budget])
        else:
            selected.extend(successes[: budget // 2])
            selected.extend(failures[: budget - budget // 2])
    return selected


def _write_effective_config(config: ExperimentConfig, output: Path) -> None:
    raw = read_yaml(config.source_path) if config.source_path else {}
    raw["resolved_models"] = {
        role: redact(read_yaml(path)) for role, path in config.models.items()
    }
    raw["resolved_dataset"] = redact(read_yaml(config.dataset))
    output.joinpath("effective_config.yaml").write_text(
        yaml.safe_dump(redact(raw), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def build_memory(config_path: str, output_root: str | None = None) -> Path:
    config = load_experiment_config(config_path)
    source_dataset = config.memory.get("source_dataset")
    build_dataset_profile = load_dataset_profile(str(source_dataset) if source_dataset else config.dataset)
    records = (
        _dataset_from_profile(config, str(source_dataset), "train")
        if source_dataset else _dataset(config, "train")
    )
    output = _resolve_output(config, output_root)
    _write_effective_config(config, output)
    method = str(config.memory.get("method", "none"))
    store_path = _artifact_path(config, config.memory.get("store_path")) or str(output / "memory.json")
    memory = _new_memory(config, method, store_path)
    agent = _build_agent(config, memory)
    trajectories: list[dict[str, Any]] = []
    source = str(config.memory.get("experience_source", "mixed")).lower()
    source = {"success": "success_only", "failure": "failure_only", "both": "mixed"}.get(source, source)
    balance_success_failure = bool(config.memory.get("balance_success_failure"))
    limit = config.memory.get("train_limit", 200)
    for record in records[: int(limit) if limit else None]:
        result = agent.run(record)
        trajectories.append(result)
        if result.get("error"):
            write_jsonl(output / "trajectories.jsonl", trajectories)
            write_jsonl(output / "errors.jsonl", [result])
            raise ExperimentRunError(
                f"Memory build aborted for id={record.id}: {result['error']}"
            )
        success = training_success(
            question=record.question,
            prediction=str(result["prediction"]),
            answerable=record.answerable,
            gold_answers=record.answers,
            llm=agent.llm,
            allow_heuristic=_allow_heuristic_protocol(config, "agent"),
        )
        experience = Experience(
            id=record.id,
            question=record.question,
            answer=result["prediction"],
            answerable=record.answerable,
            trajectory=_experience_trajectory(record, result),
            success=success,
        )
        if balance_success_failure or (
            source == "mixed" or (source == "success_only" and success)
            or (source == "failure_only" and not success)
        ):
            memory.add_experience(experience)
    # Original update_memory.py adds one trajectory immediately after each
    # sample.  Balanced source experiments then split the completed stores;
    # they do not rerun extraction on a filtered sample list.
    memory.finalize()
    if balance_success_failure:
        memory.items = _balanced_memory_items(memory.items, source)
        memory.finalize()
    memory.save(store_path)
    output.joinpath("metadata.json").write_text(json.dumps({
        "schema_version": 1, "run_id": output.name, "experiment": config.name,
        "stage": "memory_build", "seed": config.run.get("seed", 1),
        "dataset": build_dataset_profile.name,
        "memory_method": method, "agent_model": load_model_profile(config.models["agent"]).model,
        "prompt_hashes": _prompt_hashes(),
    }, indent=2), encoding="utf-8")
    dataset_name = build_dataset_profile.name
    agent_model = load_model_profile(config.models["agent"]).model
    for result in trajectories:
        result.update({"run_id": output.name, "experiment": config.name,
                       "seed": config.run.get("seed", 1), "stage": "memory_build",
                       "dataset": dataset_name, "agent_model": agent_model})
    write_jsonl(output / "trajectories.jsonl", trajectories)
    errors = [row for row in trajectories if row.get("error")]
    if errors:
        write_jsonl(output / "errors.jsonl", errors)
    output.joinpath("memories.jsonl").write_text("\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in memory.items) + "\n", encoding="utf-8")
    return output


def evaluate(config_path: str, output_root: str | None = None, memory_path: str | None = None) -> Path:
    config = load_experiment_config(config_path)
    records = _dataset(config, "test")
    test_limit = config.evaluation.get("test_limit", 400)
    records = records[: int(test_limit) if test_limit else None]
    output = _resolve_output(config, output_root)
    _write_effective_config(config, output)
    method = str(config.memory.get("method", "none"))
    store_path = (
        _artifact_path(config, memory_path)
        if memory_path else _artifact_path(config, config.memory.get("store_path"))
    )
    if method not in {"none", "human_hint"} and not store_path:
        raise ConfigError(
            "A memory path is required for evaluate when memory.method is not none/human_hint"
        )
    memory = _new_memory(config, method, store_path)
    if store_path:
        target = Path(store_path)
        target = target / "memory.json" if target.suffix.lower() != ".json" else target
        if not target.is_file():
            raise ConfigError(f"Memory artifact not found: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        from .memory import MemoryItem
        memory.items = [MemoryItem(**item) for item in payload]
    agent = _build_agent(config, memory)
    fixed_by_id: dict[str, list[dict[str, Any]]] = {}
    fixed_path = _artifact_path(config, config.memory.get("fixed_trajectory_path"))
    if fixed_path:
        for item in load_jsonl(fixed_path):
            trace = item.get("trace")
            if trace is None and isinstance(item.get("fixed_trajectory"), dict):
                # Accept the upstream wiki_react_fixed artifact shape as an
                # input adapter; MemUAQ's own traces remain the canonical form.
                trace = item["fixed_trajectory"].get("replay_steps")
            if not isinstance(trace, list):
                raise ConfigError(
                    f"Fixed trajectory trace must be a list for id={item.get('id')}"
                )
            fixed_by_id[str(item.get("id"))] = trace
    channel = str(config.memory.get("channel", "combined")).lower()
    if channel in {"trajectory_only", "guidance_only"}:
        if not fixed_path:
            raise ConfigError(f"memory.fixed_trajectory_path is required for channel={channel}")
        missing_ids = [str(record.id) for record in records if str(record.id) not in fixed_by_id]
        if missing_ids:
            raise ConfigError(
                f"Fixed trajectories missing {len(missing_ids)} evaluation records; "
                f"first missing id={missing_ids[0]}"
            )
        empty_ids = [sample_id for sample_id, trace in fixed_by_id.items() if not trace]
        if empty_ids:
            raise ConfigError(f"Fixed trajectory is empty for id={empty_ids[0]}")
    results = [
        agent.run(
            record,
            guidance_enabled=channel != "trajectory_only",
            fixed_trajectory=(
                fixed_by_id[str(record.id)]
                if channel in {"trajectory_only", "guidance_only"} else None
            ),
        )
        for record in records
    ]
    dataset_name = load_dataset_profile(config.dataset).name
    agent_model = load_model_profile(config.models["agent"]).model
    for result in results:
        result.update({"run_id": output.name, "experiment": config.name,
                       "seed": config.run.get("seed", 1), "stage": "evaluation",
                       "dataset": dataset_name, "agent_model": agent_model})
    write_jsonl(output / "trajectories.jsonl", results)
    errors = [row for row in results if row.get("error")]
    if errors:
        write_jsonl(output / "errors.jsonl", errors)
        raise ExperimentRunError(
            f"Evaluation aborted with {len(errors)} agent/environment errors; "
            f"first id={errors[0].get('id')}: {errors[0].get('error')}"
        )
    output.joinpath("metadata.json").write_text(json.dumps({"schema_version": 1,
        "run_id": output.name, "experiment": config.name, "stage": "evaluation",
        "seed": config.run.get("seed", 1), "dataset": dataset_name,
        "agent_model": agent_model, "memory_method": method,
        "prompt_hashes": _prompt_hashes(),
        "cache_enabled": bool((config.environment.get("cache") or {}).get("enabled", False))}, indent=2), encoding="utf-8")
    return output


def judge(config_path: str, input_path: str, output_path: str | None = None) -> Path:
    config = load_experiment_config(config_path)
    target = Path(output_path or Path(input_path).with_name("judgments.jsonl"))
    allow_heuristic = bool(config.evaluation.get("allow_heuristic_judge", False))
    if "judge" in config.models:
        allow_heuristic = allow_heuristic or _allow_heuristic_protocol(config, "judge")
    if "judge" not in config.models and not allow_heuristic:
        raise ConfigError(
            "models.judge is required for paper-protocol evaluation; "
            "evaluation.allow_heuristic_judge is reserved for offline smoke tests"
        )
    llm = _model_service(config.models["judge"], "judge") if "judge" in config.models else None
    judge_file(input_path, target, llm, allow_heuristic=allow_heuristic)
    rows = load_jsonl(input_path)
    judgments = load_jsonl(target)
    metrics = compute_metrics(rows, judgments,
                              acc_weight=float(config.evaluation.get("acc_weight", 0.7)),
                              ar_weight=float(config.evaluation.get("ar_weight", 0.3)))
    target.with_name("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return target


def aggregate(paths: list[str], output_path: str) -> Path:
    metrics = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(aggregate_metrics(metrics), indent=2), encoding="utf-8")
    return target
