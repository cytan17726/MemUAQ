from pathlib import Path

import pytest

from memuaq.cli import main
from memuaq.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]


def test_offline_smoke(tmp_path, capsys):
    main(["run", "--config", str(ROOT / "configs/experiments/smoke.yaml"),
          "--output-root", str(tmp_path)])
    result = Path(capsys.readouterr().out.strip())
    assert result.name == "judgments.jsonl"
    assert result.is_file()
    assert result.with_name("metrics.json").is_file()


def test_run_requires_judge_before_experiment_calls(tmp_path):
    config = tmp_path / "missing_judge.yaml"
    config.write_text(
        f"""schema_version: 1
name: missing_judge
models:
  agent: {ROOT / 'configs/models/fake.yaml'}
dataset: {ROOT / 'configs/datasets/smoke.yaml'}
memory: {{method: none}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="models.judge is required by memuaq run"):
        main(["run", "--config", str(config), "--output-root", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()
