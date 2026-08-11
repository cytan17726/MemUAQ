# MemUAQ

MemUAQ provides the code for the paper *What Makes Agent Memory Useful for
Reliable Unanswerable Question Handling?*

The release organizes the experiments around a common Wikipedia-based agent
pipeline. It includes the four memory baselines studied in the paper and the
unified memory-content analyses.

## Requirements and installation

MemUAQ requires Python 3.10 or newer. The Python dependencies are installed by
the package itself:

```bash
cd memuaq
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

An experiment using the real pipeline needs:

- network access to the Wikipedia API (the default environment is online);
- an OpenAI-compatible chat-completion API for the agent and judge models;
- an OpenAI-compatible embedding API for memory-based experiments.

The agent, judge, and embedding models may use different API platforms. A
user-owned cache can be enabled in an experiment's `environment.cache` section,
but no Wikipedia cache is bundled with this release.

To verify the installation without calling an external model or Wikipedia,
run the deterministic smoke test:

```bash
bash scripts/smoke_test.sh
```

## Datasets

This repository includes the fixed subsets used by the released experiment
configurations:

| Dataset | Memory-building split | Evaluation split |
| --- | ---: | ---: |
| KUQ | `data/kuq_train.json` (200) | `data/kuq_test.json` (400) |
| UAQFact | `data/uaqfact_train.json` (200) | `data/uaqfact_test.json` (400) |
| RefuNQ | not provided for memory construction | `data/refunq_test.json` (400) |

The files use a normalized JSON schema with `id`, `question`, `answers`,
`answerable`, and `question_type`. KUQ and UAQFact training records also keep
the `reason` field used by the original data. The loader accepts JSON arrays
and JSONL files, as well as the legacy `answer` and `gold_answer` field names.

The subsets are derived from the datasets introduced in the following work:

- *Knowledge of Knowledge: Exploring Known-Unknowns Uncertainty with Large
  Language Models* [\[paper\]](https://aclanthology.org/2024.findings-acl.383/)
- *UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable
  Questions* [\[paper\]](https://aclanthology.org/2025.findings-acl.85/)
- *Examining LLMs' Uncertainty Expression Towards Questions Outside Parametric
  Knowledge* [\[paper\]](https://arxiv.org/abs/2311.09731v2)

The provided KUQ and UAQFact training subsets contain 200 records each, and
each evaluation subset contains 400 records. The corresponding default limits
can be changed with `memory.train_limit` and `evaluation.test_limit`. RefuNQ is
evaluation-only in this release and must not be used as
`memory.source_dataset`.

## Model configuration

Model profiles are YAML files under `configs/models/`. Put the settings for
each model in its corresponding profile and reference that file from an
experiment configuration. In particular, the credential for a model can be
written directly in the same profile as its model and endpoint:

```yaml
role: agent
provider: openai_compatible
model: Qwen3-235B-A22B-Instruct-2507
base_url: https://your-platform.example/v1
api_key: your-platform-key
temperature: 0
max_tokens: 4096
```

The repository includes example profiles for the paper's agent models,
`gpt-5.4-mini` as the judge placeholder, and `Qwen3-Embedding-8B` as the
embedding model. Replace the endpoint, key, and model settings with the
configuration available to you.

Real runs require a `judge` profile to compute Acceptable Ratio (AR). Memory
methods require an `embedding` profile; the no-memory and Human Hint baselines
do not.

## Running an experiment

### Overall workflow

The one-command interface runs the three stages used by the release:

```text
memory build  ->  agent evaluation  ->  judge and metrics
```

For example:

```bash
bash scripts/run_experiment.sh configs/experiments/main/expel.example.yaml
```

The command builds memory from the configured training split, evaluates the
configured test split in the online Wikipedia environment, asks the judge
model to score UAQ responses, and writes the final `metrics.json`. The stages
can also be run separately when an existing artifact is being reused:

```bash
python -m memuaq memory build --config CONFIG
python -m memuaq evaluate --config CONFIG --memory-path PATH_TO_MEMORY_JSON
python -m memuaq judge --config CONFIG \
  --input PATH_TO_TRAJECTORIES_JSONL
```

`evaluate` writes interaction trajectories only. `judge` writes
`judgments.jsonl` and the final `metrics.json` next to the input trajectories.
The metrics contain Accuracy (Acc), Acceptable Ratio (AR), and Joint Score
(JS). Acc follows the normalized string-matching protocol used by the
original code; AR is produced by the configured judge model.

Each stage creates a directory under `outputs/<experiment-name>/<run-id>/`.
It contains the effective configuration, run metadata, trajectories, and,
where applicable, memory artifacts, judgments, metrics, and an `errors.jsonl`
file. Generated outputs are ignored by git.

Copy an example YAML under `configs/experiments/` and change its model,
dataset, memory, environment, or run fields to define a related experiment.
The shell scripts are convenience wrappers; the same commands can be called
directly with `python -m memuaq`.

### Existing memory methods

This part corresponds to the experiments in Section 4 of the paper.

#### Main in-distribution comparison

Run the no-memory and Human Hint controls together with the four memory
methods:

```text
configs/experiments/main/no_memory.example.yaml
configs/experiments/main/human_hint.example.yaml
configs/experiments/main/expel.example.yaml
configs/experiments/main/memevolve.example.yaml
configs/experiments/main/awm.example.yaml
configs/experiments/main/agentkb.example.yaml
```

Run any configuration with:

```bash
bash scripts/run_experiment.sh configs/experiments/main/expel.example.yaml
```

`expel_with_human_hint.example.yaml` is available for the corresponding
combined control. Change `models.agent` and the dataset profile when running a
different paper model or split.

#### Cross-dataset and cross-model transfer

`configs/experiments/transfer/cross_dataset.example.yaml` evaluates memory
constructed from KUQ on UAQFact. To use another source/target pair, change
`memory.source_dataset` and `dataset`; keep a training split in the source
profile and a test split in the target profile. RefuNQ can be selected as the
target evaluation dataset, but not as the source for memory construction.

`configs/experiments/transfer/cross_model.example.yaml` demonstrates reusing a
memory artifact with a different agent model. First build the artifact with
the source model, then set the target experiment's `memory.store_path` to that
artifact and keep `memory.reuse_existing: true`.

#### Functional channels

The configurations under `configs/experiments/functional_channels/` compare
decision guidance, trajectory shaping, and their combination:

```text
guidance_only.example.yaml
trajectory_only.example.yaml
combined.example.yaml
```

The two single-channel settings require a reference trajectory file. Run a
reference evaluation first, then set `memory.fixed_trajectory_path` to its
`trajectories.jsonl` before running the channel configuration. This keeps the
retrieval setup and the accumulated training experience unchanged while
changing which memory channel is exposed at evaluation time.

### Unified memory-content analysis

This part corresponds to the experiments in Section 5 of the paper. These
analyses use `memory.method: content_memory` so that content representations
can be compared under one memory pipeline. The total retrieval budget is two
memory items per query.

#### Single content types

Start with `configs/experiments/content/single.example.yaml`. Set
`memory.content_types` to one of:

```yaml
[compact_trajectory]
[insight]
[principle]
[success_trace]
[workflow]
```

Run the copied configuration once for each content type.

#### Pairwise content composition

Use `configs/experiments/content/pairwise.example.yaml` and set
`memory.content_types` to the pair to compare, for example:

```yaml
content_types: [workflow, principle]
```

When both types have candidates, the default balanced selector exposes one
item from each type while keeping the total budget at two.

#### Successful and failed experience

Use `configs/experiments/experience_source/success_failure.example.yaml` and
change `memory.experience_source` to `success_only`, `failure_only`, or
`mixed`. Set `memory.balance_success_failure: true` for the quantity-balanced
variant. Combine this setting with `memory.content_types` to test how the
usefulness of successful and failed experiences depends on the representation.

## Aggregating externally repeated runs

Repeated runs are intentionally not orchestrated by MemUAQ. If you run the
same configuration several times, aggregate the resulting metrics explicitly:

```bash
bash scripts/aggregate_results.sh --output summary.json \
  RUN1/metrics.json RUN2/metrics.json RUN3/metrics.json
```

## Citation

TBD.

## Acknowledgements

The method adapters and experiment organization were informed by the following
papers and public repositories:

- *ExpeL: LLM Agents Are Experiential Learners* [\[paper\]](https://arxiv.org/abs/2308.10144v3) [\[repository\]](https://github.com/LeapLabTHU/ExpeL)
- *Agent Workflow Memory* [\[paper\]](https://arxiv.org/abs/2409.07429) [\[repository\]](https://github.com/zorazrw/agent-workflow-memory)
- *Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving* [\[paper\]](https://arxiv.org/abs/2507.06229) [\[repository\]](https://github.com/OPPO-PersonalAI/Agent-KB)
- *MemEvolve: Meta-Evolution of Agent Memory Systems* [\[paper\]](https://arxiv.org/abs/2512.18746v1) [\[repository\]](https://github.com/bingreeky/MemEvolve)

This public codebase was reorganized and documented with assistance from
OpenAI Codex.

## License

MemUAQ is released under the Apache License 2.0; see [LICENSE](LICENSE).
