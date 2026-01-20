#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional, List, TypedDict, Annotated, Literal, TypeVar, Type, Protocol, cast, Any, Tuple, NotRequired, Iterable, Callable, Generator, Awaitable, Coroutine
from langchain_core.messages import ToolMessage, AnyMessage, SystemMessage, HumanMessage, BaseMessage, AIMessage, RemoveMessage
from langchain_core.tools import InjectedToolCallId, BaseTool
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, MessagesState
from langgraph._internal._typing import StateLike
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from .utils import cached_invoke, acached_invoke
from .summary import SummaryConfig

"""
This provides the framework for building applications which loop with an LLM,
using tools to refine the LLM output.
"""

class WithToolCallId(BaseModel):
    """
    A common schema used for tools which need explicit access to the tool_id (usually
    for calling tool_output
    """

    tool_call_id: Annotated[str, InjectedToolCallId]

def tool_output(tool_call_id: str, res: dict) -> Command:
    """
    Create a LangGraph Command for final tool outputs that update workflow state.

    Used by "completion" tools to set final
    results in the workflow state. The workflow's conditional edge will detect
    these state updates and route to completion.

    Args:
        tool_call_id: The ID of the tool call being responded to
        res: Dictionary containing the final workflow results to merge into state

    Returns:
        Command that updates state with final results and a success message
    """
    return Command(update={
        **res,
        "messages": [ToolMessage(
            tool_call_id=tool_call_id,
            content="Success"
        )]
    })

def tool_return(
    tool_call_id: str,
    content: str
) -> Command:
    """
    Create a LangGraph Command for tool responses that need to continue processing.

    Used by tools that want to return a result and continue the workflow by routing
    back to the tool_result node for LLM processing.

    Usually this is unnecessary, and you can just return a "str" or dictionary from the
    tool body.

    Args:
        tool_call_id: The ID of the tool call being responded to
        content: The response content from the tool execution

    Returns:
        Command that updates messages and continues workflow
    """
    return Command(
        update={
            "messages": [ToolMessage(tool_call_id=tool_call_id, content=content)]
        }
    )

def tool_state_update(
    tool_call_id: str,
    content: str,
    **state_diff
) -> Command:
    update = {
        "messages": [
            ToolMessage(tool_call_id=tool_call_id, content=content)
        ],
        **state_diff
    }
    return Command(update=update)

class FlowInput(TypedDict):
    """
    Upper bound on any type used as an input to a workflow.
    """

    """
    Any contents to be placed *before* the initial prompt but
    *after* the system prompt.
    """
    front_matter: NotRequired[list[HumanMessage]]
    """
    Any workflow specific data to add *after* the initial prompt.
    """
    input: list[str | dict]

# InputState <: FlowInput
InputState = TypeVar('InputState', bound=FlowInput)

InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)

class NodeFunction(Protocol[InputType, OutputType]):
    """Protocol defining the signature for LangGraph input functions. All instantiations of InputType
    are expected to be a type bounded above by FlowInput, although this is not expressible."""
    def __call__(self, state: InputType) -> OutputType:
        ...

class AsyncNodeFunction(Protocol[InputType, OutputType]):
    """Protocol defining the signature for LangGraph input functions. All instantiations of InputType
    are expected to be a type bounded above by FlowInput, although this is not expressible."""
    def __call__(self, state: InputType) -> Coroutine[Any, Any, OutputType]:
        ...

type ChatNodeFunction[InputType] = NodeFunction[InputType, dict[str, list[BaseMessage]]]
type AsyncChatNodeFunction[InputType] = AsyncNodeFunction[InputType, dict[str, list[BaseMessage]]]
type AnyChatNodeFunction[StateT] = AsyncChatNodeFunction[StateT] | ChatNodeFunction[StateT]
type AnyNodeFunction[InputType, OutputType] = AsyncNodeFunction[InputType, OutputType] | NodeFunction[InputType, OutputType]


ResT = TypeVar("ResT")

type PureFunctionGenerator[ResT] = Generator[list[AnyMessage], BaseMessage, ResT]
type PureFunction[StateT, ResT] = Callable[[StateT], PureFunctionGenerator[ResT]]
type SyncLLM = Callable[[list[AnyMessage]], BaseMessage]
type AsyncLLM = Callable[[list[AnyMessage]], Awaitable[BaseMessage]]
type LLM = Runnable[LanguageModelInput, BaseMessage]


# TypeVars for generic typing
StateT = TypeVar('StateT', bound=MessagesState)
OutputT = TypeVar('OutputT', bound=StateLike)
ContextT = TypeVar("ContextT", bound=StateLike)

def _async_llm(
    llm: LLM
) -> AsyncLLM:
    async def impl(
        s: list[AnyMessage]
    ) -> BaseMessage:
        res = await acached_invoke(llm, s)
        return res
    return impl

def _sync_llm(
    llm: LLM
) -> SyncLLM:
    return lambda m: cached_invoke(llm, m)

IN = TypeVar("IN")
OUT = TypeVar("OUT")

def _stitch_sync_impl(
    pure: PureFunction[IN, OUT],
    llm_impl: SyncLLM
) -> NodeFunction[IN, OUT]:
    def impl(
        state: IN
    ) -> OUT:
        thunk = pure(state)
        d = next(thunk)
        res = llm_impl(d)
        try:
            thunk.send(res)
            assert False, "did not terminate"
        except StopIteration as e:
            return e.value
    return impl

def _stitch_async_impl(
    pure: PureFunction[IN, OUT],
    llm_impl: AsyncLLM
) -> AsyncNodeFunction[IN, OUT]:
    async def impl(
        state: IN
    ) -> OUT:
        thunk = pure(state)
        d = next(thunk)
        res = await llm_impl(d)
        try:
            thunk.send(res)
            assert False, "did not terminate"
        except StopIteration as e:
            return e.value
    return impl

def _pure_tool_generator(
    t: type[StateT]
) -> PureFunction[StateT, dict[str, list[BaseMessage]]]:
    def impl(
        state: MessagesState
    ) -> PureFunctionGenerator[dict[str, list[BaseMessage]]]:
        res = yield state["messages"]
        return {"messages": [res]}
    return impl

def tool_result_generator(
    t: type[StateT],
    llm: Runnable[LanguageModelInput, BaseMessage],
) -> ChatNodeFunction[StateT]:
    return _stitch_sync_impl(
        _pure_tool_generator(t),
        _sync_llm(llm)
    )

def async_tool_result_generator(
    t: type[StateT],
    llm: LLM
) -> AsyncChatNodeFunction[StateT]:
    return _stitch_async_impl(
        _pure_tool_generator(t),
        _async_llm(llm)
    )

class _ResultFact(Protocol):
    def __call__(
        self,
        t: type[StateT],
        llm: LLM
    ) -> AnyChatNodeFunction[StateT]:
        ...

def _get_summarizer_pure(
    system_prompt: str,
    initial_prompt: str,
    state_type: type[StateT],
    context: SummaryConfig[StateT]
) -> PureFunction:
    def to_return(state: StateT) -> PureFunctionGenerator:
        config = context
        assert config is not None

        summary_prompt = config.get_summarization_prompt(state)

        messages = state["messages"].copy()
        assert len(messages) >= config.max_messages

        try:
            msg = yield(messages + [HumanMessage(content=summary_prompt)])
            assert isinstance(msg, AIMessage)
            summary = msg.text()
            resume_message = config.get_resume_prompt(state, summary)
            config.on_summary(state, summary, resume_message)
            return {
                "messages": [
                    RemoveMessage(id="__remove_all__"),
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=initial_prompt),
                    HumanMessage(content=resume_message),
                ]
            }
        except Exception:
            return {}
    return to_return

I = TypeVar("I", bound=FlowInput)
O = TypeVar("O")

def _get_initial_pure(
    t: Type[I],
    output_state: Type[O],
    sys_prompt: str,
    initial_prompt: str,
) -> PureFunction[I, O]:
    def impl(
        state: I
    ) -> PureFunctionGenerator[O]:
        """
        Any fields in state that are not specified by the FlowInput bound (i.e.,
        are not front_matter or input) are automatically added to the main state.
        """
        initial_messages : List[AnyMessage] = [
            SystemMessage(
                sys_prompt
            )
        ]
        front = state.get("front_matter", None)
        if front is not None:
            app = cast(List[HumanMessage], front)
            initial_messages.extend(app)

        prompt_and_input_message: List[str | dict] = [initial_prompt]
        prompt_and_input_message.extend(state["input"])

        initial_messages.append(
            HumanMessage(
                content=prompt_and_input_message
            )
        )
        # The format of the initial message is:
        # [system_prompt, front_matter?, initial_prompt, input]
        # front_matter is for variable inputs that provide context to the initial prompt
        # (e.g., reference material)
        # this cast is fine because we don't write into initial_messages
        res = yield initial_messages

        initial_messages.append(
            cast(AnyMessage, res)
        )
        to_ret : dict[str, Any] = {"messages": initial_messages}
        for (k, v) in state.items():
            if k == "front_matter" or k == "input":
                continue
            to_ret[k] = v
        return cast(O, to_ret)
    return impl
        

def get_summarizer(
    llm: LLM,
    system_prompt: str,
    initial_prompt: str,
    state_type: type[StateT],
    context: SummaryConfig[StateT]
) -> ChatNodeFunction[StateT]:
    return _stitch_sync_impl(
        pure=_get_summarizer_pure(system_prompt, initial_prompt, state_type, context),
        llm_impl=_sync_llm(llm)
    )

def get_async_summarizer(
    llm: LLM,
    system_prompt: str,
    initial_prompt: str,
    state_type: type[StateT],
    context: SummaryConfig[StateT]
) -> AsyncChatNodeFunction[StateT]:
    return _stitch_async_impl(
        _get_summarizer_pure(system_prompt, initial_prompt, state_type, context),
        _async_llm(llm)
    )

class _SummarizerFact(Protocol):
    def __call__(
        self,
        llm: LLM,
        system_prompt: str,
        initial_prompt: str,
        state_type: type[StateT],
        context: SummaryConfig[StateT]
    ) -> AnyChatNodeFunction[StateT]:
        ...

def initial_node(
    t: Type[InputState],
    output_state: Type[StateT],
    sys_prompt: str,
    initial_prompt: str,
    llm: LLM
) -> NodeFunction[InputState, StateT]:
    return _stitch_sync_impl(
        _get_initial_pure(t, output_state, sys_prompt, initial_prompt),
        _sync_llm(llm)
    )

def async_initial_node(
    t: Type[InputState],
    output_state: Type[StateT],
    sys_prompt: str,
    initial_prompt: str,
    llm: LLM
) -> AsyncNodeFunction[InputState, StateT]:
    return _stitch_async_impl(
        _get_initial_pure(t, output_state, sys_prompt, initial_prompt),
        _async_llm(llm)
    )

class _InitialFact(Protocol):
    def __call__(
        self,
        t: Type[InputState],
        output_state: Type[StateT],
        sys_prompt: str,
        initial_prompt: str,
        llm: LLM
    ) -> AnyNodeFunction[InputState, StateT]:
        ...

INITIAL_NODE = "initial"
TOOLS_NODE = "tools"
TOOL_RESULT_NODE = "tool_result"
SUMMARIZE_NODE = "summarize"

BoundLLM = LLM

SplitTool = tuple[dict[str, Any], BaseTool]

def build_workflow(
    state_class: Type[StateT],
    input_type: Type[InputState],
    tools_list: Iterable[BaseTool | SplitTool],
    sys_prompt: str,
    initial_prompt: str,
    output_key: str,
    unbound_llm: BaseChatModel,
    output_schema: Optional[Type[OutputT]] = None,
    context_schema: Optional[Type[ContextT]] = None,
    summary_config: SummaryConfig[StateT] | None = None
) -> Tuple[StateGraph[StateT, ContextT, InputState, OutputT], LLM]:
    return _build_workflow(
        state_class,
        input_type,
        tools_list,
        sys_prompt,
        initial_prompt,
        output_key,
        unbound_llm,
        output_schema,
        context_schema,
        summary_config,
        tool_result_generator,
        initial_node,
        get_summarizer
    )


def build_async_workflow(
    state_class: Type[StateT],
    input_type: Type[InputState],
    tools_list: Iterable[BaseTool | SplitTool],
    sys_prompt: str,
    initial_prompt: str,
    output_key: str,
    unbound_llm: BaseChatModel,
    output_schema: Optional[Type[OutputT]] = None,
    context_schema: Optional[Type[ContextT]] = None,
    summary_config: SummaryConfig[StateT] | None = None
) -> Tuple[StateGraph[StateT, ContextT, InputState, OutputT], LLM]:
    return _build_workflow(
        state_class,
        input_type,
        tools_list,
        sys_prompt,
        initial_prompt,
        output_key,
        unbound_llm,
        output_schema,
        context_schema,
        summary_config,
        async_tool_result_generator,
        async_initial_node,
        get_async_summarizer
    )

def _build_workflow(
    state_class: Type[StateT],
    input_type: Type[InputState],
    tools_list: Iterable[BaseTool | SplitTool],
    sys_prompt: str,
    initial_prompt: str,
    output_key: str,
    unbound_llm: BaseChatModel,
    output_schema: Optional[Type[OutputT]],
    context_schema: Optional[Type[ContextT]],
    summary_config: SummaryConfig[StateT] | None,
    result_fact: _ResultFact,
    init_fact: _InitialFact,
    summary_fact: _SummarizerFact
) -> Tuple[StateGraph[StateT, ContextT, InputState, OutputT], LLM]:
    """
    Build a standard workflow with initial node -> tools -> tool_result pattern.
    More specifically, the initial_prompt should instruct the LLM to use any of
    the tools available. The results of these tools are then returned back to the LLM,
    which will continue to call tools. One of these tools should be a distinguished "output"
    tool; this tool should be called when the llm's work is complete, and set
    the `output_key` field of the running state to be non-None. The graph created
    by this function will detect this field becoming non-null, and break the loop with the LLM,
    indicating the computation is complete.

    Args:
        state_class: The type of the "main" state, bounded by `MessagesState`
        input_type: The type of the "input" state, bounded by `FlowInput`. The `input` field of \
            this type should be used for task specific inputs
        tools_list: A list of tools that the LLM can call during iteration.
        sys_prompt: The system prompt sent to start the conversation
        initial_prompt: The static prompt sent describing the task and how to use the tools
        output_key: The designated "output" tool should set this key to be non-None in the current state. \
            When this happens, the computation is considered complete, and the graph exits
        unbound_llm: The llm to use for the computation and looping
        output_schema: (Optional) if non-none, describes the output format of the computation.
        context_schema: (Optional) if non-none, the type of contexts passed through the computation
        summary_config: if non-none, the parameters and prompts for history summarization.

    Returns:
        The state graph compiled to execute the workflow above, and the llm wth the tools bound
        and configured.
    """

    def should_end(state: StateT) -> Literal["__end__", "tool_result"]:
        """Check if workflow should end based on output key being defined."""
        assert isinstance(state, dict)
        if state.get(output_key, None) is not None:
            return "__end__"
        return "tool_result"
    
    tool_schemas : list[BaseTool | dict] = []
    tool_impls : list[BaseTool] = []

    supports_memory = isinstance(unbound_llm, ChatAnthropic) and \
        (beta_attr := getattr(unbound_llm, "betas", [])) is not None and \
            "context-management-2025-06-27" in beta_attr

    for t in tools_list:
        if isinstance(t, tuple):
            tool_schemas.append(t[0])
            tool_impls.append(t[1])
        elif t.name == "memory" and supports_memory:
            tool_schemas.append({
                "type": "memory_20250818",
                "name": "memory"
            })
            tool_impls.append(t)
        else:
            tool_schemas.append(t)
            tool_impls.append(t)

    llm = unbound_llm.bind_tools(tool_schemas)

    # Create initial node and tool node with curried LLM
    init_node = init_fact(input_type, state_class, sys_prompt=sys_prompt, initial_prompt=initial_prompt, llm=llm)
    tool_node = ToolNode(tool_impls, handle_tool_errors=False)
    tool_result_node = result_fact(state_class, llm)

    # Build the graph with fixed input schema, no context
    builder = StateGraph(
        state_schema=state_class,
        input_schema=input_type,
        output_schema=output_schema,
        context_schema=context_schema
    )

    builder.set_entry_point(INITIAL_NODE)
    builder.add_node(INITIAL_NODE, init_node, input_schema=input_type)
    builder.add_node(TOOLS_NODE, tool_node)
    builder.add_node(TOOL_RESULT_NODE, tool_result_node)
    builder.add_edge(INITIAL_NODE, TOOLS_NODE)
    builder.add_edge(TOOL_RESULT_NODE, TOOLS_NODE)

    if summary_config is not None:
        def routing(state: StateT) -> Literal["summarize", "tool_result", "__end__"]:
            if state.get(output_key, None) is not None:
                return "__end__"
            elif len(state["messages"]) > summary_config.max_messages:
                return "summarize"
            else:
                return "tool_result"

        summarizer = summary_fact(
            unbound_llm, sys_prompt, initial_prompt, state_class, summary_config
        )
        builder.add_node(SUMMARIZE_NODE, summarizer)
        builder.add_edge(SUMMARIZE_NODE, TOOL_RESULT_NODE)
        builder.add_conditional_edges(TOOLS_NODE, routing)
    else:
        builder.add_conditional_edges(TOOLS_NODE, should_end)
    return builder, llm
