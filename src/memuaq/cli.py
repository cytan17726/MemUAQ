from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, load_experiment_config
from .runner import _artifact_path, aggregate, build_memory, evaluate, judge


def _config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--output-root", default=None, help="Override output root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memuaq", description="MemUAQ experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    memory = commands.add_parser("memory", help="Memory operations")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_build = memory_commands.add_parser("build", help="Build memory from a training split")
    _config_arg(memory_build)

    evaluate_parser = commands.add_parser("evaluate", help="Run agent evaluation")
    _config_arg(evaluate_parser)
    evaluate_parser.add_argument("--memory-path", default=None)

    judge_parser = commands.add_parser("judge", help="Judge existing trajectories")
    judge_parser.add_argument("--config", required=True)
    judge_parser.add_argument("--input", required=True)
    judge_parser.add_argument("--output", default=None)

    aggregate_parser = commands.add_parser("aggregate", help="Aggregate metrics from runs")
    aggregate_parser.add_argument("--metrics", nargs="+", required=True)
    aggregate_parser.add_argument("--output", required=True)

    run_parser = commands.add_parser("run", help="Run build, evaluate, and judge stages")
    _config_arg(run_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "memory":
        output = build_memory(args.config, args.output_root)
    elif args.command == "evaluate":
        output = evaluate(args.config, args.output_root, args.memory_path)
    elif args.command == "judge":
        output = judge(args.config, args.input, args.output)
    elif args.command == "aggregate":
        output = aggregate(args.metrics, args.output)
    elif args.command == "run":
        config = load_experiment_config(args.config)
        if (
            "judge" not in config.models
            and not bool(config.evaluation.get("allow_heuristic_judge", False))
        ):
            raise ConfigError(
                "models.judge is required by memuaq run before any experiment calls; "
                "use separate evaluate/judge stages or an offline smoke configuration"
            )
        method = str(config.memory.get("method", "none"))
        memory_path = config.memory.get("store_path")
        if method not in {"none", "human_hint"}:
            if config.memory.get("reuse_existing"):
                resolved = _artifact_path(config, str(memory_path)) if memory_path else None
                if not resolved:
                    raise ConfigError("memory.store_path is required when memory.reuse_existing is true")
                artifact = Path(resolved)
                artifact = artifact / "memory.json" if artifact.suffix.lower() != ".json" else artifact
                if not artifact.is_file():
                    raise ConfigError(f"Reusable memory artifact not found: {artifact}")
            else:
                build_output = build_memory(args.config, args.output_root)
                memory_path = memory_path or str(build_output / "memory.json")
        eval_output = evaluate(args.config, args.output_root, memory_path)
        output = judge(args.config, str(eval_output / "trajectories.jsonl"))
    else:  # pragma: no cover
        raise SystemExit(2)
    print(Path(output).resolve())
