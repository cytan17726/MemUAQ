from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from .data import DatasetRecord
from .environment import WikiReactEnvironment
from .llm import LLMService
from .memory import MemoryItem, MemoryMethod
from .prompts.agent import HUMAN_HINT, SYSTEM_PROMPT


_ACTION_RE = re.compile(
    r"^\s*Action(?:\s+\d+)?\s*:\s*(Search|Lookup|Finish)\[([^\n]*)\]",
    re.I | re.M,
)


def parse_action(text: str) -> str:
    match = _ACTION_RE.search(text)
    if match:
        return f"{match.group(1).title()}[{match.group(2).strip()}]"
    for name in ("Search", "Lookup", "Finish"):
        direct = re.search(rf"{name}\[(.*?)\]", text, re.I | re.S)
        if direct:
            return f"{name}[{direct.group(1).strip()}]"
    return ""


class WikiReactAgent:
    def __init__(self, llm: LLMService, environment: WikiReactEnvironment,
                 memory: MemoryMethod, max_steps: int = 10, human_hint: bool = False):
        self.llm = llm
        self.environment = environment
        self.memory = memory
        self.max_steps = max_steps
        self.human_hint = human_hint

    def run(self, record: DatasetRecord, *, guidance_enabled: bool = True,
            fixed_trajectory: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.environment.reset()
        retrieved: list[MemoryItem] = self.memory.retrieve(record.question) if guidance_enabled else []
        guidance = (
            self.memory.render(retrieved)
            if guidance_enabled and self.memory.name != "human_hint" else ""
        )
        system_prompt = SYSTEM_PROMPT
        if self.human_hint:
            system_prompt += "\n" + HUMAN_HINT
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if guidance:
            messages.append({
                "role": "user",
                "content": f"[Memory System Guidance - BEGIN]\n{guidance}",
            })
        messages.append({
            "role": "user",
            "content": (
                f"Question: {record.question}\n"
                "Start solving now. Remember to output one Thought line and one Action line."
            ),
        })
        steps: list[dict[str, Any]] = []
        usage: list[dict[str, Any]] = []
        run_error: str | None = None

        if fixed_trajectory is not None:
            replay_steps: list[dict[str, Any]] = []
            for source_step in fixed_trajectory:
                if not isinstance(source_step, dict):
                    continue
                step = dict(source_step)
                action = str(step.get("action", "")).strip()
                if not action.lower().startswith(("search[", "lookup[", "finish[")):
                    action = parse_action(action) or action
                step["action"] = action
                if action.lower().startswith("finish["):
                    break
                replay_steps.append(step)
            steps = replay_steps
            for index, step in enumerate(replay_steps, start=1):
                action = str(step.get("action", "")).strip()
                observation = str(step.get("observation", "")).strip()
                if action:
                    messages.append({"role": "assistant", "content": f"Action {index}: {action}"})
                if observation:
                    messages.append({"role": "user", "content": f"Observation {index}: {observation}"})

            final_index = len(replay_steps) + 1
            final = self.llm.chat(messages + [{
                "role": "user",
                "content": (
                    "Based on the fixed evidence above, give the final answer now.\n"
                    f"Output exactly:\nThought {final_index}: <brief reasoning>\n"
                    f"Action {final_index}: Finish[answer]"
                ),
            }])
            action = parse_action(final.content)
            answer = action[7:-1].strip() if action.lower().startswith("finish[") else final.content.strip()
            return {
                "schema_version": 1, "id": record.id, "question": record.question,
                "gold_answers": record.answers, "answerable": record.answerable,
                "question_type": record.question_type, "prediction": answer,
                "memory_method": self.memory.name,
                "retrieved_memories": [asdict(item) for item in retrieved],
                "memory_guidance": guidance, "trace": steps,
                "stop_reason": "fixed_trajectory", "usage": [final.usage] if final.usage else [],
                "error": None,
            }

        for index in range(1, self.max_steps + 1):
            try:
                response = self.llm.chat(messages)
            except Exception as exc:  # noqa: BLE001
                run_error = f"llm_error:{type(exc).__name__}:{exc}"
                break
            if response.usage:
                usage.append(response.usage)
            action = parse_action(response.content)
            if not action:
                action = f"Finish[{response.content.strip()}]"
            try:
                observation = self.environment.step(action)
            except Exception as exc:  # noqa: BLE001
                observation = f"Environment error: {type(exc).__name__}: {exc}"
                run_error = f"environment_error:{type(exc).__name__}:{exc}"
            step = {"step": index, "assistant": response.content, "action": action,
                    "observation": observation, "usage": response.usage}
            steps.append(step)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": f"Observation {index}: {observation}"})
            if self.environment.terminated:
                break

        answer = self.environment.answer
        if not answer and not run_error:
            step_index = len(steps) + 1
            current_page = getattr(self.environment.current_page, "title", "")
            current_page_line = f"Current page: {current_page}\n" if current_page else ""
            final = self.llm.chat(messages + [{
                "role": "user",
                "content": (
                    f"You have reached the maximum number of steps ({self.max_steps}).\n"
                    "You must give your final reply now based on the evidence collected so far.\n"
                    f"{current_page_line}Question: {record.question}\n"
                    f"Output exactly:\nThought {step_index}: <brief reasoning>\n"
                    f"Action {step_index}: Finish[answer]"
                ),
            }])
            if final.usage:
                usage.append(final.usage)
            action = parse_action(final.content)
            if action.lower().startswith("finish["):
                answer = action[7:-1]
            else:
                answer = final.content.strip()
                action = f"Finish[{answer}]"
            observation = self.environment.step(action)
            steps.append({
                "step": step_index,
                "assistant": final.content,
                "action": action,
                "observation": observation,
                "usage": final.usage,
                "forced_finalization": True,
            })

        stop_reason = "error" if run_error else ("finish" if answer else "max_steps")

        return {
            "schema_version": 1,
            "id": record.id,
            "question": record.question,
            "gold_answers": record.answers,
            "answerable": record.answerable,
            "question_type": record.question_type,
            "prediction": answer,
            "memory_method": self.memory.name,
            "retrieved_memories": [asdict(item) for item in retrieved],
            "memory_guidance": guidance,
            "trace": steps,
            "stop_reason": stop_reason,
            "usage": usage,
            "error": run_error,
        }
