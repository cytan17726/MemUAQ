from pathlib import Path

from memuaq.agent import WikiReactAgent, parse_action
from memuaq.config import ModelProfile
from memuaq.data import DatasetRecord
from memuaq.environment import WikiPage, WikiReactEnvironment, WikipediaClient
from memuaq.llm import LLMService, ScriptedClient
from memuaq.memory import create_memory
from memuaq.prompts.agent import HUMAN_HINT, SYSTEM_PROMPT
from memuaq.config import load_experiment_config
from memuaq.runner import _build_agent

ROOT = Path(__file__).resolve().parents[1]


def test_no_memory_prompt_is_not_human_hint_prompt():
    assert "verify answerability" not in SYSTEM_PROMPT.lower()
    assert "abstain" in HUMAN_HINT.lower()


def test_human_hint_memory_selects_system_hint():
    config = load_experiment_config(ROOT / "configs/experiments/main/human_hint.example.yaml")
    agent = _build_agent(config, create_memory("human_hint"))
    assert agent.human_hint is True


def test_parse_action_uses_the_first_action_line():
    text = "Thought 1: search\nAction 1: Search[foo]\nAction 2: Finish[bar]"
    assert parse_action(text) == "Search[foo]"


def test_memory_with_human_hint_and_fixed_trajectory():
    client = ScriptedClient(["Thought 1: use evidence.\nAction 1: Finish[No answer]"])
    service = LLMService(client, ModelProfile(role="agent", provider="fake", model="offline"))
    agent = WikiReactAgent(service, WikiReactEnvironment(WikipediaClient()),
                           create_memory("expel"), max_steps=2, human_hint=True)
    record = DatasetRecord(id="1", question="q", answers=[], answerable=False)
    result = agent.run(record, fixed_trajectory=[{"observation": "No reliable evidence"}])
    assert result["stop_reason"] == "fixed_trajectory"
    assert "BE TRUSTWORTHY" in client.calls[0][0]["content"]
    assert "Fixed interaction evidence" not in "\n".join(
        str(message.get("content", "")) for message in client.calls[0]
    )


def test_fixed_trajectory_normalizes_upstream_action_prefix():
    client = ScriptedClient(["Thought 2: use the replayed evidence.\nAction 2: Finish[answer]"])
    service = LLMService(client, ModelProfile(role="agent", provider="fake", model="offline"))
    agent = WikiReactAgent(service, WikiReactEnvironment(WikipediaClient()),
                           create_memory("none"), max_steps=2)
    record = DatasetRecord(id="1", question="q", answers=["answer"], answerable=True)
    result = agent.run(record, fixed_trajectory=[
        {"action": "Action 1: Search[page]", "observation": "evidence"},
        {"action": "Action 2: Finish[old answer]", "observation": "Finished."},
    ])
    assert result["trace"][0]["action"] == "Search[page]"
    assert result["prediction"] == "answer"


def test_lookup_returns_context_for_first_match():
    class StubClient:
        def page(self, title):
            return WikiPage(title="Page", summary="", content="alpha KEY omega; beta KEY delta")

        def search(self, query):
            return []

    environment = WikiReactEnvironment(StubClient(), lookup_window=12)
    environment.reset()
    environment.current_page = StubClient().page("Page")
    result = environment.lookup("KEY")
    assert "alpha KEY omega" in result
    assert "beta KEY delta" not in result
