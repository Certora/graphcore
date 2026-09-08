import pytest

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from graphcore.graph import (
    MAX_CONSECUTIVE_NO_TOOL_TURNS,
    NoToolCallsError,
    _consecutive_no_tool_turns,
    _scolding_node,
)


def _no_tool_turn() -> AIMessage:
    return AIMessage(content="I am done, thanks.")


def _tool_turn() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "result", "args": {}, "id": "call-1"}],
    )


def _scolding() -> HumanMessage:
    return HumanMessage(content="call a tool", display_tag="scolding")


def test_counts_only_the_tail():
    messages: list[AnyMessage] = [_tool_turn(), ToolMessage(content="ok", tool_call_id="call-1"), _no_tool_turn()]
    assert _consecutive_no_tool_turns(messages) == 1


def test_scoldings_do_not_break_the_run():
    messages: list[AnyMessage] = [_no_tool_turn(), _scolding(), _no_tool_turn(), _scolding(), _no_tool_turn()]
    assert _consecutive_no_tool_turns(messages) == 3


def test_a_tool_call_resets_the_count():
    messages: list[AnyMessage] = [
        _no_tool_turn(), _scolding(),
        _tool_turn(), ToolMessage(content="ok", tool_call_id="call-1"),
        _no_tool_turn(),
    ]
    assert _consecutive_no_tool_turns(messages) == 1


def test_scolds_while_under_the_limit():
    messages: list[AnyMessage] = [_no_tool_turn()]
    out = _scolding_node({"messages": messages})
    assert len(out["messages"]) == 1
    assert getattr(out["messages"][0], "display_tag", None) == "scolding"


def _run_of_no_tool_turns(n: int) -> list[AnyMessage]:
    """n no-tool turns, each scolded except the last — the state the node sees."""
    messages: list[AnyMessage] = []
    for _ in range(n - 1):
        messages += [_no_tool_turn(), _scolding()]
    messages.append(_no_tool_turn())
    return messages


def test_last_turn_before_the_limit_is_still_scolded():
    messages = _run_of_no_tool_turns(MAX_CONSECUTIVE_NO_TOOL_TURNS - 1)
    out = _scolding_node({"messages": messages})
    assert getattr(out["messages"][0], "display_tag", None) == "scolding"


def test_raises_once_scolding_stops_working():
    messages = _run_of_no_tool_turns(MAX_CONSECUTIVE_NO_TOOL_TURNS)
    with pytest.raises(NoToolCallsError):
        _scolding_node({"messages": messages})
