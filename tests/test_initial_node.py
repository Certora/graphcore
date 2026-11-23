import operator
from typing import Annotated

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, MessagesState, StateGraph

from graphcore.graph import FlowInput, initial_node


class _Input(FlowInput):
    history: Annotated[list[str], operator.add]
    label: str


class _State(MessagesState):
    history: Annotated[list[str], operator.add]
    label: str


def _graph():
    llm = RunnableLambda(lambda _messages: AIMessage(content="ok"))
    builder = StateGraph(state_schema=_State, input_schema=_Input)
    builder.add_node("initial", initial_node(_Input, _State, "sys", "go", llm), input_schema=_Input)
    builder.add_edge(START, "initial")
    builder.add_edge("initial", END)
    return builder.compile()


def test_an_input_field_with_an_appending_reducer_is_written_once():
    """A caller seeding an author with the edit ids it had already applied got each id twice,
    because the initial node wrote the input fields again after the graph had written them."""
    out = _graph().invoke({"input": ["start"], "history": ["e1", "e2"], "label": "x"})
    assert out["history"] == ["e1", "e2"]


def test_input_fields_still_reach_the_state():
    out = _graph().invoke({"input": ["start"], "history": [], "label": "x"})
    assert out["label"] == "x"
    assert [type(m).__name__ for m in out["messages"]] == ["SystemMessage", "HumanMessage", "AIMessage"]
