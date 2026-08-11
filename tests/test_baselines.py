import json
from pathlib import Path

import pytest

from memuaq.cli import main
from memuaq.config import ConfigError
from memuaq.data import DatasetRecord
from memuaq.runner import _experience_trajectory, build_memory, evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_memory_trajectory_preserves_multiline_thought():
    record = DatasetRecord(
        id="1", question="q", answers=["a"], answerable=True,
    )
    trajectory = _experience_trajectory(record, {
        "prediction": "a",
        "trace": [{
            "assistant": "Thought 1: first line\nsecond line\nAction 1: Finish[a]",
            "action": "Finish[a]",
            "observation": "Finished.",
        }],
    })
    assert {"type": "thought", "content": "first line\nsecond line"} in trajectory


@pytest.mark.parametrize("method", ["expel", "memevolve", "awm", "agentkb", "content_memory"])
def test_baseline_end_to_end_offline(method, tmp_path):
    config = tmp_path / f"{method}.yaml"
    content = "  content_types: [workflow]\n" if method == "content_memory" else ""
    config.write_text(
        f"""schema_version: 1
name: test_{method}
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
agent: {{type: wiki_react, max_steps: 2}}
environment: {{type: wikipedia, language: en, cache: {{enabled: false}}}}
memory:
  method: {method}
  top_k: 2
{content}run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    built = build_memory(str(config), str(tmp_path / "runs"))
    evaluated = evaluate(str(config), str(tmp_path / "runs"), str(built / "memory.json"))
    assert (built / "memory.json").is_file()
    assert (evaluated / "trajectories.jsonl").is_file()
    assert not (evaluated / "metrics.json").exists()


def test_memory_build_updates_online_before_next_training_sample(tmp_path):
    config = tmp_path / "online.yaml"
    config.write_text(
        f"""schema_version: 1
name: online_expel
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
agent: {{type: wiki_react, max_steps: 2}}
environment: {{type: wikipedia, language: en, cache: {{enabled: false}}}}
memory: {{method: expel, top_k: 2}}
run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    built = build_memory(str(config), str(tmp_path / "runs"))
    rows = [json.loads(line) for line in (built / "trajectories.jsonl").read_text().splitlines()]
    assert rows[1]["memory_guidance"]


def test_balanced_source_split_happens_after_online_updates(tmp_path):
    config = tmp_path / "balanced.yaml"
    config.write_text(
        f"""schema_version: 1
name: balanced_content
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
agent: {{type: wiki_react, max_steps: 2}}
environment: {{type: wikipedia, language: en, cache: {{enabled: false}}}}
memory:
  method: content_memory
  content_types: [raw]
  experience_source: mixed
  balance_success_failure: true
  top_k: 2
run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    built = build_memory(str(config), str(tmp_path / "runs"))
    trajectories = [
        json.loads(line) for line in (built / "trajectories.jsonl").read_text().splitlines()
    ]
    memories = json.loads((built / "memory.json").read_text())

    assert trajectories[1]["memory_guidance"]
    # The original splitter uses min(success, failure) as the total mixed
    # budget, assigning the odd remainder to failures.
    assert len(memories) == 1
    assert memories[0]["metadata"]["success"] is False


def test_cross_dataset_build_uses_source_dataset_profile(tmp_path):
    source_profile = tmp_path / "source.yaml"
    source_profile.write_text(
        f"name: source_smoke\ntrain_path: {ROOT / 'tests/fixtures/smoke.json'}\n"
        f"path: {ROOT / 'tests/fixtures/smoke.json'}\nformat: json\n",
        encoding="utf-8",
    )
    target_profile = tmp_path / "target.yaml"
    target_profile.write_text(
        f"name: target_smoke\npath: {ROOT / 'tests/fixtures/smoke.json'}\nformat: json\n",
        encoding="utf-8",
    )
    config = tmp_path / "cross_dataset.yaml"
    config.write_text(
        f"""schema_version: 1
name: cross_dataset
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {target_profile}
agent: {{type: wiki_react, max_steps: 1}}
environment: {{type: wikipedia, language: en, cache: {{enabled: false}}}}
memory:
  method: expel
  source_dataset: {source_profile}
run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    built = build_memory(str(config), str(tmp_path / "runs"))
    metadata = json.loads((built / "metadata.json").read_text())
    trajectories = [
        json.loads(line) for line in (built / "trajectories.jsonl").read_text().splitlines()
    ]

    assert metadata["dataset"] == "source_smoke"
    assert trajectories and {row["dataset"] for row in trajectories} == {"source_smoke"}


def test_memory_build_rejects_test_only_dataset(tmp_path):
    profile = tmp_path / "test_only.yaml"
    profile.write_text(
        f"name: test_only\npath: {ROOT / 'tests/fixtures/smoke.json'}\nformat: json\n",
        encoding="utf-8",
    )
    config = tmp_path / "test_only_memory.yaml"
    config.write_text(
        f"""schema_version: 1
name: test_only_memory
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {profile}
memory: {{method: expel}}
run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="no train_path"):
        build_memory(str(config), str(tmp_path / "runs"))


def test_cross_model_reuse_requires_existing_artifact(tmp_path):
    config = tmp_path / "cross_model.yaml"
    config.write_text(
        f"""schema_version: 1
name: cross_model
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
agent: {{type: wiki_react, max_steps: 1}}
environment: {{type: wikipedia, language: en, cache: {{enabled: false}}}}
memory:
  method: expel
  store_path: outputs/missing-source-memory.json
  reuse_existing: true
evaluation: {{allow_heuristic_judge: true}}
run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Reusable memory artifact not found"):
        main(["run", "--config", str(config), "--output-root", str(tmp_path / "runs")])


def test_evaluate_accepts_upstream_fixed_trajectory_shape(tmp_path):
    fixed = tmp_path / "fixed.jsonl"
    fixed.write_text(
        "\n".join(json.dumps({
            "id": sample_id,
            "fixed_trajectory": {
                "replay_steps": [{"action": "Search[page]", "observation": "evidence"}],
                "replay_messages": [],
                "finalization_prompt": "Give the final answer.",
            },
        }) for sample_id in ("uaq-1", "abq-1")) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "fixed.yaml"
    config.write_text(
        f"""schema_version: 1
name: upstream_fixed
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
agent: {{type: wiki_react, max_steps: 2}}
environment: {{type: wikipedia, language: en, cache: {{enabled: false}}}}
memory:
  method: none
  channel: guidance_only
  fixed_trajectory_path: {fixed}
run: {{seed: 1, output_root: outputs}}
""",
        encoding="utf-8",
    )
    evaluated = evaluate(str(config), str(tmp_path / "runs"))
    rows = [json.loads(line) for line in (evaluated / "trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {row["stop_reason"] for row in rows} == {"fixed_trajectory"}
