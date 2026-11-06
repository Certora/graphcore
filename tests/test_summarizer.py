from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph import MessagesState

from graphcore.graph import SUMMARY_ATTEMPTS, _get_summarizer_pure
from graphcore.summary import SummaryConfig


class _Config(SummaryConfig[Any]):
    def get_summarization_prompt(self, state: Any) -> str:
        return "Summarize."

    def get_resume_prompt(self, state: Any, summary: str) -> str:
        return f"Resume from: {summary}"


def _history() -> list[AnyMessage]:
    return [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "memory", "args": {}, "id": "call-1"}]),
        ToolMessage(content="notes", tool_call_id="call-1"),
    ]


def _summarize(replies: list[AIMessage]) -> tuple[list[list[AnyMessage]], dict[str, Any]]:
    """Run the summarizer against ``replies`` in order; the requests it made, and its result."""
    pure = _get_summarizer_pure("system", "initial", MessagesState, _Config())
    thunk = pure({"messages": _history()})
    requests = [next(thunk)]
    try:
        for reply in replies:
            requests.append(thunk.send(reply))
    except StopIteration as e:
        return requests, e.value
    raise AssertionError("the summarizer asked for more replies than the test supplied")


def _tool_call_only() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "memory", "args": {}, "id": "call-2"}],
        response_metadata={"stop_reason": "tool_use"},
    )


def test_the_request_asks_for_plain_text():
    requests, _ = _summarize([AIMessage(content="the summary")])
    last = requests[0][-1]
    assert isinstance(last, HumanMessage)
    assert isinstance(last.content, str)
    assert last.content.startswith("Summarize.")
    assert "Do not call any tool" in last.content


def test_a_text_reply_replaces_the_history():
    _, result = _summarize([AIMessage(content="the summary")])
    messages = result["messages"]
    assert isinstance(messages[0], RemoveMessage)
    assert messages[-1].content == "Resume from: the summary"


def test_an_empty_reply_is_retried():
    requests, result = _summarize([_tool_call_only(), AIMessage(content="the summary")])
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert result["messages"][-1].content == "Resume from: the summary"


def test_no_text_after_every_attempt_keeps_the_history(caplog):
    replies = [_tool_call_only() for _ in range(SUMMARY_ATTEMPTS)]
    _, result = _summarize(replies)
    assert result == {}
    assert "stop_reason=tool_use" in caplog.text
    assert "tool calls=['memory']" in caplog.text
