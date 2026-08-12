"""Serialized execution for tools that rewrite a plain state channel.

``ToolNode`` runs every tool call of an assistant turn concurrently and emits one
update per call. That is right for tools that only append messages or write
reducer-backed channels, and wrong for a tool that *rewrites* a plain
(``LastValue``) channel: two such calls land in one superstep, produce two writes
to that channel, and LangGraph rejects the batch with ``InvalidUpdateError`` —
taking the whole graph down mid-run.

The model-level lever, ``parallel_tool_calls=False``, is all-or-nothing: it is a
property of the request, not of a tool, so it serializes every tool to protect
one. Serializing at execution time instead confines the cost to the tools that
need it:

* calls to undeclared tools run concurrently through the stock ``ToolNode``, and
  their updates pass through **verbatim**, so a batch of reducer-backed writes
  still arrives as one write per call;
* calls to declared tools run one at a time, each against the state the previous
  one produced, so tool *N* observes tool *N-1*'s writes through
  ``InjectedState``;
* the declared group's state writes collapse into a single update, so the
  rewritten channel takes one write per superstep and needs no reducer.

Sequencing is also the semantics a model assumes when it emits two edits to one
buffer in a single turn: the second is written against the result of the first.
A declared tool that then finds a stale anchor fails the way it always does — a
rejection message it can act on — instead of aborting the graph.

A reducer over the channel is the other way to admit concurrent writes, and it
cannot replace this: it runs after the tools have already answered, so an update
it drops or rewrites is invisible to the model, and validation that belongs in
the tool (parsing the result, rejecting it with errors) cannot follow the write
into a channel function whose only failure mode is raising.

Declare a tool with :func:`serialize_writes`; :func:`make_tool_node` picks the
declarations up off the tool list and wraps only when something declared.
"""

import asyncio
from typing import Any, Iterable, NamedTuple, Sequence

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages.tool import ToolCall
from langgraph._internal._runnable import RunnableCallable
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

#: Metadata key by which a tool declares that it must not share a superstep with
#: any other declaring tool. The value is the set of state keys it may write
#: besides ``messages``.
SERIALIZED_WRITES = "graphcore.serialized_writes"

#: Matches ``ToolNode``'s default ``messages_key``.
_MESSAGES_KEY = "messages"


def serialize_writes[T: BaseTool](tool: T, *state_keys: str) -> T:
    """Declare *tool* as serialized, naming the plain state keys it rewrites.

    All declaring tools of a graph form ONE group: they are serialized against
    each other, not just against themselves, because they contend for the same
    channels.

    Name only plain (``LastValue``) keys. A reducer-backed key must NOT be named —
    the group's declared writes are merged last-wins, which would drop all but the
    final write to a key whose reducer was meant to combine them. Naming nothing
    is not the safe default either: an undeclared key keeps the stock
    one-write-per-call behaviour, so a plain one written by two calls of the group
    still collides.
    """
    if not state_keys:
        raise ValueError(
            f"tool {tool.name!r} declared serialization without naming any state key"
        )
    tool.metadata = {**(tool.metadata or {}), SERIALIZED_WRITES: frozenset(state_keys)}
    return tool


def make_tool_node(tools: Sequence[BaseTool], **kwargs: Any) -> ToolNode | RunnableCallable:
    """A ``ToolNode`` over *tools*, wrapped to serialize the declared ones.

    Returns the stock node untouched when nothing declares — one declaring tool
    is enough to wrap, since the collision this prevents is most often a tool
    batched against *itself* (two edits to one buffer in one turn).
    """
    node = ToolNode(list(tools), **kwargs)
    names, keys = _declarations(tools)
    if not names:
        return node
    if kwargs.get("messages_key", _MESSAGES_KEY) != _MESSAGES_KEY:
        # The wrapper reads and rebuilds the message channel by name; supporting a
        # renamed one means threading the key through, which no caller has wanted.
        raise NotImplementedError("serialized tools require the default messages_key")
    return _serializing_node(node, names, keys)


def _declarations(tools: Iterable[BaseTool]) -> tuple[frozenset[str], frozenset[str]]:
    names: set[str] = set()
    keys: set[str] = set()
    for t in tools:
        declared = (t.metadata or {}).get(SERIALIZED_WRITES)
        if declared is None:
            continue
        names.add(t.name)
        keys.update(declared)
    return frozenset(names), frozenset(keys)


def _last_ai_message(messages: Sequence[AnyMessage]) -> tuple[int, AIMessage] | None:
    """The message ``ToolNode`` would take the batch from, and where it sits — its
    ``_parse_input`` reads the last ``AIMessage``, not ``messages[-1]``."""
    for i in reversed(range(len(messages))):
        m = messages[i]
        if isinstance(m, AIMessage):
            return i, m
    return None


def _with_calls(state: dict, at: int, ai: AIMessage, calls: Sequence[ToolCall]) -> dict:
    """*state* with the ``AIMessage`` at *at* narrowed to *calls*.

    ``ToolNode`` takes the batch off the last ``AIMessage`` of the state handed to
    it and resolves ``InjectedState`` against that same dict. Narrowing the
    message is therefore how a sub-batch is dispatched, and swapping the dict is
    how a serialized tool is shown its predecessor's writes.
    """
    messages = list(state[_MESSAGES_KEY])
    messages[at] = ai.model_copy(update={"tool_calls": list(calls)})
    return {**state, _MESSAGES_KEY: messages}


def _sole_update(out: Any) -> dict[str, Any]:
    """The single state update a one-call ``ToolNode`` invocation produced.

    The output is ``{"messages": [...]}`` when the tool returned a message, or a
    one-item list of ``Command`` when it returned one.
    """
    items = out if isinstance(out, list) else [out]
    if len(items) != 1:
        raise ValueError(f"expected one update from a single tool call, got {len(items)}")
    item = items[0]
    if isinstance(item, Command):
        if item.graph is Command.PARENT:
            raise ValueError("a serialized tool may not return a parent-graph Command")
        update = item.update
    else:
        update = item
    if not isinstance(update, dict):
        raise TypeError(f"unexpected ToolNode output {type(update).__name__}")
    return update


def _call_index(cmd: Command, order: dict[str | None, int]) -> int:
    """Where *cmd*'s answer sits in the model's original call order, so the
    ToolMessages are appended in the order they were requested rather than
    grouped by how they were scheduled.

    Sorting is what's available: the outputs cannot simply be interleaved by
    walking the original call list, because ``ToolNode._combine_tool_outputs``
    collapses a whole parallel batch into a single ``{"messages": [...]}`` dict
    whenever none of its tools returned a ``Command``, leaving fewer outputs than
    calls to interleave.
    """
    update = cmd.update if isinstance(cmd.update, dict) else {}
    for m in update.get(_MESSAGES_KEY, []):
        if isinstance(m, ToolMessage) and m.tool_call_id in order:
            return order[m.tool_call_id]
    return len(order)


class _Split(NamedTuple):
    """One batch, divided into the calls that must run alone and the rest."""
    at: int
    """Index of the ``AIMessage`` carrying the batch."""
    ai: AIMessage
    par: list[ToolCall]
    seq: list[ToolCall]
    order: dict[str | None, int]
    """Tool call id to its position in the model's original batch."""


def _serializing_node(
    tool_node: ToolNode, exclusive: frozenset[str], declared: frozenset[str]
) -> RunnableCallable:
    def plan(state: Any) -> _Split | None:
        """The split, or ``None`` when there is nothing to serialize and the
        stock node should handle the batch untouched."""
        if not isinstance(state, dict) or _MESSAGES_KEY not in state:
            return None
        found = _last_ai_message(state[_MESSAGES_KEY])
        if found is None:
            return None
        at, ai = found
        calls = list(ai.tool_calls)
        seq = [c for c in calls if c["name"] in exclusive]
        if len(seq) < 2:
            # A lone declared call contends with nothing, so it keeps the stock
            # node's output shape rather than being rebuilt into an equivalent one.
            return None
        par = [c for c in calls if c["name"] not in exclusive]
        return _Split(at, ai, par, seq, {c["id"]: i for i, c in enumerate(calls)})

    def fold(local: dict, out: Any, merged: dict, answers: list[dict]) -> dict:
        """Absorb one serialized call's result and hand back the state the next
        call should run against.

        Only declared keys collapse into the group's single write. Anything else
        the tool wrote — a reducer-backed channel like a validation stamp — rides
        on that call's own update, keeping the one-write-per-call shape its reducer
        expects, and is left out of *local* so it reaches the next tool exactly as
        an unserialized batch would.
        """
        update = dict(_sole_update(out))
        answer = {_MESSAGES_KEY: update.pop(_MESSAGES_KEY, [])}
        owned = {k: v for k, v in update.items() if k in declared}
        answer.update((k, v) for k, v in update.items() if k not in declared)
        answers.append(answer)
        merged.update(owned)
        return {**local, **owned}

    def result(
        par_out: Any, merged: dict, answers: list[dict], order: dict[str | None, int]
    ) -> list[Command]:
        items = par_out if isinstance(par_out, list) else [] if par_out is None else [par_out]
        # Everything is wrapped as a Command: LangGraph only treats a list return
        # as several updates when at least one element is one (see
        # ``StateGraph._get_updates``), and wrapping keeps the parallel group's
        # one-write-per-call shape that reducer-backed channels depend on.
        out = [i if isinstance(i, Command) else Command(update=i) for i in items]
        # Each serialized call answers in its own update so the batch's messages can
        # be ordered as the model requested them; the group's collapsed write of the
        # declared keys rides on the last of them, keeping those to one per step.
        out.extend(Command(update=a) for a in answers[:-1])
        out.append(Command(update={**answers[-1], **merged}))
        out.sort(key=lambda c: _call_index(c, order))
        return out

    def func(input: Any, config: RunnableConfig) -> Any:
        s = plan(input)
        if s is None:
            return tool_node.invoke(input, config)
        # Runs the groups back to back; `afunc` overlaps them.
        par_out = tool_node.invoke(_with_calls(input, s.at, s.ai, s.par), config) if s.par else None
        local, merged, answers = input, {}, []
        for c in s.seq:
            out = tool_node.invoke(_with_calls(local, s.at, s.ai, [c]), config)
            local = fold(local, out, merged, answers)
        return result(par_out, merged, answers, s.order)

    async def afunc(input: Any, config: RunnableConfig) -> Any:
        s = plan(input)
        if s is None:
            return await tool_node.ainvoke(input, config)

        async def run_parallel() -> Any:
            if not s.par:
                return None
            return await tool_node.ainvoke(_with_calls(input, s.at, s.ai, s.par), config)

        async def run_serial() -> tuple[dict, list[dict]]:
            local, merged, answers = input, {}, []
            for c in s.seq:
                out = await tool_node.ainvoke(_with_calls(local, s.at, s.ai, [c]), config)
                local = fold(local, out, merged, answers)
            return merged, answers

        # Overlapped: the chain must not wait behind a slow undeclared tool.
        par_out, (merged, answers) = await asyncio.gather(run_parallel(), run_serial())
        return result(par_out, merged, answers, s.order)

    return RunnableCallable(func, afunc, name="tools", trace=False)


__all__ = ["SERIALIZED_WRITES", "serialize_writes", "make_tool_node"]
