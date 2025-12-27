import types
import sys


def test_parse_failure_retries_with_gpt52_no_thinking(monkeypatch):
    """
    If the first model response is unparsable, LLMAgent should retry once using gpt-5.2 with thinking disabled.
    """
    from agents.llm_agent import LLMAgent
    from engine import GameState, Player, Action

    calls = []

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
        calls.append(kwargs)
        # Simulate: when thinking param is present, model returns bad JSON; when absent, returns good JSON.
        if kwargs.get("reasoning_effort") is not None:
            return _FakeResponse("NOT JSON")
        return _FakeResponse('{"reasoning":"ok","action_type":"end_turn","action_payload":null}')

    fake_litellm = types.SimpleNamespace(completion=fake_completion, drop_params=False)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    state = GameState(
        game_id="g1",
        players=[Player(id="player_1", name="P1"), Player(id="player_2", name="P2")],
        phase="playing",
    )
    legal_actions_list = [(Action.END_TURN, None)]

    agent = LLMAgent(
        player_id="player_1",
        model="gpt-5.2",
        enable_retrieval=False,
        thinking_mode=True,
        thinking_effort="medium",
    )

    action, payload, reasoning, raw = agent.choose_action(state, legal_actions_list)

    assert action == Action.END_TURN
    assert payload is None
    assert reasoning == "ok"
    assert len(calls) == 2
    # First call uses thinking (reasoning_effort present)
    assert calls[0]["model"] == "gpt-5.2"
    assert calls[0].get("reasoning_effort") == "medium"
    # Retry call forces gpt-5.2 with thinking disabled (reasoning_effort omitted)
    assert calls[1]["model"] == "gpt-5.2"
    assert "reasoning_effort" not in calls[1]


