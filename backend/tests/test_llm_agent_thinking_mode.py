import types
import sys

import pytest


def _install_fake_litellm(monkeypatch, captured_kwargs, content="ok"):
    class _FakeMsg:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMsg(content)

    class _FakeUsage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResponse(content)

    fake_litellm = types.SimpleNamespace(completion=fake_completion, drop_params=False)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)


def test_thinking_mode_adds_reasoning_effort_for_gpt5(monkeypatch):
    from agents.llm_agent import LLMAgent

    captured = {}
    _install_fake_litellm(monkeypatch, captured)

    agent = LLMAgent(
        player_id="player_1",
        model="gpt-5.2",
        enable_retrieval=False,
        thinking_mode=True,
        thinking_effort="medium",
    )
    out = agent._call_llm([{"role": "user", "content": "hi"}])

    assert out == "ok"
    assert captured["model"] == "gpt-5.2"
    assert captured["reasoning_effort"] == "medium"


def test_thinking_mode_not_sent_when_disabled(monkeypatch):
    from agents.llm_agent import LLMAgent

    captured = {}
    _install_fake_litellm(monkeypatch, captured)

    agent = LLMAgent(
        player_id="player_1",
        model="gpt-5.2",
        enable_retrieval=False,
        thinking_mode=False,
        thinking_effort="medium",
    )
    agent._call_llm([{"role": "user", "content": "hi"}])

    assert "reasoning" not in captured
    assert "reasoning_effort" not in captured


def test_thinking_effort_validation():
    from agents.llm_agent import LLMAgent

    with pytest.raises(ValueError):
        LLMAgent(
            player_id="player_1",
            model="gpt-5.2",
            enable_retrieval=False,
            thinking_mode=True,
            thinking_effort="banana",
        )


