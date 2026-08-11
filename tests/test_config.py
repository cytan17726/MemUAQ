from pathlib import Path

import pytest

from memuaq.config import ConfigError, load_experiment_config, load_model_profile, redact


ROOT = Path(__file__).resolve().parents[1]


def test_direct_api_key_is_supported(tmp_path):
    profile = tmp_path / "model.yaml"
    profile.write_text("role: agent\nmodel: demo\napi_key: direct-key\napi_key_env: OTHER_KEY\n", encoding="utf-8")
    loaded = load_model_profile(profile)
    assert loaded.resolved_api_key() == "direct-key"


def test_smoke_config_resolves_references():
    config = load_experiment_config(ROOT / "configs/experiments/smoke.yaml")
    assert config.name == "smoke"
    assert Path(config.models["agent"]).is_file()
    assert Path(config.dataset).is_file()


def test_redact_hides_keys():
    assert redact({"api_key": "secret", "model": "x"}) == {"api_key": "[REDACTED]", "model": "x"}


def test_unknown_model_key_is_rejected(tmp_path):
    profile = tmp_path / "model.yaml"
    profile.write_text("model: demo\nunknown_setting: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_model_profile(profile)


def test_unknown_channel_is_rejected(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        f"""schema_version: 1
name: bad
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
  embedding: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
memory: {{method: expel, channel: invalid}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown memory.channel"):
        load_experiment_config(config)


def test_memory_experiment_requires_embedding_profile(tmp_path):
    config = tmp_path / "missing_embedding.yaml"
    config.write_text(
        f"""schema_version: 1
name: missing_embedding
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
memory: {{method: content_memory, content_types: [workflow]}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="models.embedding is required"):
        load_experiment_config(config)
