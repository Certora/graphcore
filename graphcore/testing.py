from pydantic import BaseModel
from typing_extensions import TypedDict

import uuid

from typing import (
    Any, Mapping, TYPE_CHECKING, cast,
    Generic, TypeVar, Callable
)

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph._internal._typing import StateLike


# ---------------------------------------------------------------------------
# Typed tool call harness
# ---------------------------------------------------------------------------


if TYPE_CHECKING:
    class ToolId[T: Mapping[str, Any] | BaseModel](str):
        ...
else:
    class ToolId(str):
        def __new__(cls, s: str) -> str:
            return s

        def __class_getitem__(cls, _item: Any) -> type:
            return cls


class ToolCallDict(TypedDict):
    name: str
    args: dict[str, Any]


def tool_call[T: Mapping[str, Any] | BaseModel](name: ToolId[T], args: T) -> ToolCallDict:
    args_dict = args.model_dump() if isinstance(args, BaseModel) else dict(args)
    return {"name": name, "args": args_dict}

def tool_call_raw(name: str, **args) -> ToolCallDict:
    return {"name": name, "args": args}

STATE_TYPE = TypeVar("STATE_TYPE", bound = MessagesState)

CTXT_LIKE = TypeVar("CTXT_LIKE", bound=StateLike)

class _ContextRecord(Generic[CTXT_LIKE]):
    def __init__(self, ty: type[CTXT_LIKE], ctxt: CTXT_LIKE):
        self.ty = ty
        self.ctxt = ctxt


class ToolCallPair(TypedDict):
    tool_call: ToolCall
    resp: str

class Scenario(Generic[STATE_TYPE]):
    def __init__(
        self,
        state_type: type[STATE_TYPE],
        *tools: BaseTool
    ):
        self.tools = list(tools)
        self.state_type = state_type
        self.ctxt : _ContextRecord | None = None

    def with_context(
        self,
        init_context: StateLike
    ) -> "Scenario[STATE_TYPE]":
        res = Scenario(self.state_type, *self.tools)
        res.ctxt = _ContextRecord(type(init_context), init_context)
        return res
    

    @classmethod
    def last_single_tool_mapper(
        cls,
        last_tool: str
    ) -> Callable[[STATE_TYPE], str]:
        def cb(
            st: STATE_TYPE
        ) -> str:
            r = cls.get_last_tool_result(st)
            assert last_tool in r and len(r[last_tool]) == 1
            return r[last_tool][0]["resp"].strip()
        return cb
    
    @classmethod
    def last_single_tool(
        cls, last_tool: str, st: STATE_TYPE
    ) -> str:
        return cls.last_single_tool_mapper(last_tool)(st)

    @classmethod
    def get_last_tool_result(
        cls,
        t: STATE_TYPE
    ) -> dict[str, list[ToolCallPair]]:
        m = t["messages"]
        assert len(m) > 1
        ind_it = -1
        assert isinstance(m[ind_it], AIMessage)
        ind_it -= 1
        curr = m[ind_it]
        tool_resps : list[ToolMessage] = []
        while not isinstance(curr, AIMessage):
            assert isinstance(curr, ToolMessage)
            tool_resps.append(curr)
            ind_it -= 1
            assert -ind_it <= len(m), "underflow looking for AI message"
            curr = m[ind_it]
        assert len(curr.tool_calls) > 0, "Prior ai message had no tool calls"
        assert all([
            isinstance(tm.content, str) for tm in tool_resps
        ]), "Weird tool response"
        id_to_resp = {
            tm.tool_call_id: cast(str, tm.content) for tm in tool_resps
        }
        to_ret : dict[str, list[ToolCallPair]]= {}
        for i in curr.tool_calls:
            if i["name"] not in to_ret:
                to_ret[i["name"]] = []
            assert i["id"] is not None, "un-ided call"
            assert i["id"] in id_to_resp, "tool call not serviced"
            to_ret[i["name"]].append({
                "resp": id_to_resp[i["id"]].strip(),
                "tool_call": i
            })
        return to_ret
    

    def init(
        self,
        **init_kwargs
    ) -> "InitializedScenario[STATE_TYPE]":
        return InitializedScenario(self.state_type, self.tools, init_kwargs, self.ctxt)

STAGED_TYPE = TypeVar("STAGED_TYPE")

class StagedGraphExecution(Generic[STATE_TYPE, STAGED_TYPE]):
    def __init__(self,
                 transformer: Callable[[STATE_TYPE], STAGED_TYPE],
                 graph: "InitializedScenario[STATE_TYPE]"
                 ):
        self._tr = transformer
        self.graph = graph
    
    async def run(self) -> STAGED_TYPE:
        res = await self.graph.run()
        return self._tr(res)

class InitializedScenario(Generic[STATE_TYPE]):
    def __init__(
        self,
        state_type: type[STATE_TYPE],
        tools: list[BaseTool],
        init_kwargs: dict[str, Any],
        ctxt: _ContextRecord | None
    ):
        self.state_type = state_type
        self.tools = tools
        self.init_kwargs = init_kwargs
        self.tool_turns : list[list[ToolCallDict]] = []
        self.context_record = ctxt

    def _copy(self) -> "InitializedScenario[STATE_TYPE]":
        to_ret = InitializedScenario(self.state_type, self.tools, self.init_kwargs, self.context_record)
        to_ret.tool_turns = self.tool_turns.copy()
        return to_ret

    def turn(self, *tool_calls: ToolCallDict) -> "InitializedScenario[STATE_TYPE]":
        to_ret = self._copy()
        to_ret.tool_turns.append(list(tool_calls))
        return to_ret
    
    def turns(self, *tool_calls: ToolCallDict) -> "InitializedScenario[STATE_TYPE]":
        to_ret = self._copy()
        to_ret.tool_turns.extend([
            [tc] for tc in tool_calls
        ])
        return to_ret

    def with_context(
        self,
        ctxt: StateLike
    ) -> "InitializedScenario[STATE_TYPE]":
        to_ret = self._copy()
        to_ret.context_record = _ContextRecord(
            type(ctxt), ctxt
        )
        return to_ret
    
    def map(
        self,
        cb: Callable[[STATE_TYPE], STAGED_TYPE]
    ) -> StagedGraphExecution[STATE_TYPE, STAGED_TYPE]:
        return StagedGraphExecution(cb, self)
    
    async def run_last_single_tool(
        self, last_tool: str
    ) -> str:
        return await self.map(Scenario.last_single_tool_mapper(last_tool)).run()
    
    async def map_run(self, cb: Callable[[STATE_TYPE], STAGED_TYPE]) -> STAGED_TYPE:
        return await self.map(cb).run()
    
    async def run(self) -> STATE_TYPE:
        responses: list[BaseMessage] = []
        for turn in self.tool_turns:
            tcs : list[ToolCall] = [
                {
                    "id": uuid.uuid4().hex,
                    "name": tc["name"],
                    "args": tc["args"],
                    "type": "tool_call"
                }
                for tc in turn
            ]
            responses.append(AIMessage(content="", tool_calls=tcs))
        responses.append(AIMessage(content="Done."))

        tool_node = ToolNode(self.tools, handle_tool_errors=False)
        llm = FakeMessagesListChatModel(responses=responses)

        async def agent(state: STATE_TYPE) -> dict[str, list[BaseMessage]]:
            return {"messages": [await llm.ainvoke(state["messages"])]}

        def should_continue(state: dict) -> str:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return "__end__"

        context_type = self.context_record.ty if self.context_record else None
        graph = StateGraph(self.state_type, context_type)
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")

        state_in : STATE_TYPE = cast(STATE_TYPE, {
            "messages": [HumanMessage(content="Go.")],
            **self.init_kwargs,
        })

        res = await graph.compile().ainvoke(
            state_in, 
            context=self.context_record.ctxt if self.context_record else None
        )
        assert isinstance(res, dict)
        return cast(STATE_TYPE, res)
