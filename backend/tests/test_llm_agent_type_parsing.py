from api.routes import _parse_llm_agent_spec


def test_parse_llm_legacy():
    assert _parse_llm_agent_spec("llm") == {"model": None, "thinking_mode": None, "thinking_effort": None}


def test_parse_llm_model_only_disables_thinking():
    assert _parse_llm_agent_spec("llm:gpt-5.2") == {
        "model": "gpt-5.2",
        "thinking_mode": False,
        "thinking_effort": "medium",
    }


def test_parse_llm_thinking_colon_form():
    assert _parse_llm_agent_spec("llm:gpt-5.2:thinking:medium") == {
        "model": "gpt-5.2",
        "thinking_mode": True,
        "thinking_effort": "medium",
    }


def test_parse_llm_thinking_dash_form():
    assert _parse_llm_agent_spec("llm:gpt-5.2:thinking-high") == {
        "model": "gpt-5.2",
        "thinking_mode": True,
        "thinking_effort": "high",
    }


def test_parse_non_llm_returns_none():
    assert _parse_llm_agent_spec("random") is None


