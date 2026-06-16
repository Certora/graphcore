"""
Async tests for the graphcore memory tool infrastructure (Postgres-only).

Mirrors the sync test_memory.py suite but exercises :class:`AsyncPostgresBackend`
and :func:`async_memory_tool`.
"""

from __future__ import annotations

from typing import Awaitable

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from graphcore.tools.memory import (
    MemoryBackendError,
    MemoryToolImpl,
    async_memory_tool,
)

from .conftest import AnyAsyncBackend, AsyncBackendFactory

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


async def run_async_agent(
    backend: MemoryToolImpl[Awaitable[str]], llm_responses: list[BaseMessage]
) -> list[BaseMessage]:
    """
    Build a minimal async ReAct-style agent wired to the async memory tool,
    drive it with ``FakeMessagesListChatModel``, and return the full message list.
    """
    tool = async_memory_tool(backend)
    tool_node = ToolNode([tool], handle_tool_errors=False)
    llm = FakeMessagesListChatModel(responses=llm_responses)

    async def agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
        return {"messages": [await llm.ainvoke(state["messages"])]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    result = await graph.compile().ainvoke(
        {"messages": [HumanMessage(content="Do the task.")]}
    )
    return result["messages"]


# ---------------------------------------------------------------------------
# Backend-direct: basic CRUD
# ---------------------------------------------------------------------------


class TestAsyncBasicCRUD:
    async def test_create_and_read(self, async_backend: AnyAsyncBackend) -> None:
        result = await async_backend.create("/memories/hello.txt", "hello world\n")
        assert "created" in result.lower()
        assert await async_backend.read_file("/memories/hello.txt") == "hello world\n"

    async def test_view_file_numbered(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "alpha\nbeta\ngamma\n")
        viewed = await async_backend.view("/memories/f.txt", None)
        assert "alpha" in viewed
        assert "beta" in viewed
        assert "gamma" in viewed

    async def test_view_file_range(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "line1\nline2\nline3\nline4\n")
        viewed = await async_backend.view("/memories/f.txt", (2, 3))
        assert "line2" in viewed
        assert "line3" in viewed
        assert "line1" not in viewed
        assert "line4" not in viewed

    async def test_view_nonexistent_returns_error(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        result = await async_backend.view("/memories/nope.txt", None)
        assert "ERROR" in result

    async def test_overwrite_file(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "v1\n")
        await async_backend.create("/memories/f.txt", "v2\n")
        assert await async_backend.read_file("/memories/f.txt") == "v2\n"

    async def test_str_replace(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "hello world\n")
        await async_backend.str_replace("/memories/f.txt", "hello", "goodbye")
        assert await async_backend.read_file("/memories/f.txt") == "goodbye world\n"

    async def test_str_replace_not_found_returns_error(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/f.txt", "hello world\n")
        result = await async_backend.str_replace("/memories/f.txt", "foobar", "baz")
        assert "ERROR" in result

    async def test_str_replace_ambiguous_returns_error(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/f.txt", "hello hello\n")
        result = await async_backend.str_replace("/memories/f.txt", "hello", "world")
        assert "ERROR" in result

    async def test_insert_mid_file(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "line1\nline3\n")
        await async_backend.insert("/memories/f.txt", 1, "line2")
        assert (
            await async_backend.read_file("/memories/f.txt") == "line1\nline2\nline3\n"
        )

    async def test_insert_at_start(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "line2\n")
        await async_backend.insert("/memories/f.txt", 0, "line1")
        assert await async_backend.read_file("/memories/f.txt") == "line1\nline2\n"

    async def test_insert_at_end(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "line1\n")
        await async_backend.insert("/memories/f.txt", 1, "line2")
        assert await async_backend.read_file("/memories/f.txt") == "line1\nline2\n"

    async def test_insert_out_of_bounds_returns_error(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/f.txt", "line1\n")
        result = await async_backend.insert("/memories/f.txt", 99, "extra")
        assert "ERROR" in result

    async def test_delete_file(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/f.txt", "data")
        await async_backend.delete("/memories/f.txt")
        assert await async_backend.read_file("/memories/f.txt") is None


# ---------------------------------------------------------------------------
# Backend-direct: directories
# ---------------------------------------------------------------------------


class TestAsyncDirectories:
    async def test_auto_mkdir_on_create(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/a/b/c/deep.txt", "deep")
        assert await async_backend.read_file("/memories/a/b/c/deep.txt") == "deep"

    async def test_view_directory_lists_children(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/file1.txt", "a")
        await async_backend.create("/memories/file2.txt", "b")
        listing = await async_backend.view("/memories", None)
        assert "file1.txt" in listing
        assert "file2.txt" in listing

    async def test_view_directory_shows_subdirs(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/subdir/file.txt", "x")
        listing = await async_backend.view("/memories", None)
        assert "subdir" in listing

    async def test_delete_directory_cascades_to_children(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/dir/file1.txt", "a")
        await async_backend.create("/memories/dir/nested/file2.txt", "b")
        await async_backend.delete("/memories/dir")
        assert await async_backend.read_file("/memories/dir/file1.txt") is None
        assert await async_backend.read_file("/memories/dir/nested/file2.txt") is None


# ---------------------------------------------------------------------------
# Backend-direct: rename
# ---------------------------------------------------------------------------


class TestAsyncRename:
    async def test_rename_file(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/old.txt", "content\n")
        result = await async_backend.rename("/memories/old.txt", "/memories/new.txt")
        assert "Renamed" in result
        assert await async_backend.read_file("/memories/old.txt") is None
        assert await async_backend.read_file("/memories/new.txt") == "content\n"

    async def test_rename_file_to_new_subdir(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/flat.txt", "data\n")
        await async_backend.rename("/memories/flat.txt", "/memories/sub/flat.txt")
        assert await async_backend.read_file("/memories/flat.txt") is None
        assert await async_backend.read_file("/memories/sub/flat.txt") == "data\n"

    async def test_rename_directory_updates_all_descendants(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/old_dir/file.txt", "top\n")
        await async_backend.create("/memories/old_dir/child/nested.txt", "nested\n")
        await async_backend.rename("/memories/old_dir", "/memories/new_dir")
        assert await async_backend.read_file("/memories/old_dir/file.txt") is None
        assert (
            await async_backend.read_file("/memories/old_dir/child/nested.txt") is None
        )
        assert await async_backend.read_file("/memories/new_dir/file.txt") == "top\n"
        assert (
            await async_backend.read_file("/memories/new_dir/child/nested.txt")
            == "nested\n"
        )

    async def test_rename_nonexistent_returns_error(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        result = await async_backend.rename("/memories/ghost.txt", "/memories/new.txt")
        assert "ERROR" in result

    async def test_rename_destination_exists_returns_error(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/a.txt", "a")
        await async_backend.create("/memories/b.txt", "b")
        result = await async_backend.rename("/memories/a.txt", "/memories/b.txt")
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# Backend-direct: path validation
# ---------------------------------------------------------------------------


class TestAsyncPathValidation:
    """The backend layer raises ``MemoryBackendError`` for path-validation
    failures; the memory *tool* wrapper is the layer that catches and
    transforms those into LLM-facing ``Error: ...`` strings (see
    ``memory_tool`` / ``async_memory_tool``). These tests exercise the
    backend directly and therefore assert the raise behaviour."""

    async def test_create_outside_memories_raises(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        with pytest.raises(MemoryBackendError, match="/memories"):
            await async_backend.create("/tmp/evil.txt", "bad")

    async def test_view_outside_memories_raises(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        with pytest.raises(MemoryBackendError):
            await async_backend.view("/etc/passwd", None)

    async def test_rename_source_outside_memories_raises(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        with pytest.raises(MemoryBackendError):
            await async_backend.rename("/etc/passwd", "/memories/stolen.txt")


# ---------------------------------------------------------------------------
# Namespace copying (init_from)
# ---------------------------------------------------------------------------


class TestAsyncNamespaceCopy:
    async def test_copy_flat_files(self, async_factory: AsyncBackendFactory) -> None:
        src = await async_factory.make("src")
        await src.create("/memories/note.txt", "important\n")
        await src.create("/memories/other.txt", "also important\n")

        dst = await async_factory.make("dst", init_from="src")
        assert await dst.read_file("/memories/note.txt") == "important\n"
        assert await dst.read_file("/memories/other.txt") == "also important\n"

    async def test_copy_includes_nested_directories(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        src = await async_factory.make("src")
        await src.create("/memories/docs/readme.txt", "readme\n")
        await src.create("/memories/docs/sub/detail.txt", "detail\n")

        dst = await async_factory.make("dst", init_from="src")
        assert await dst.read_file("/memories/docs/readme.txt") == "readme\n"
        assert await dst.read_file("/memories/docs/sub/detail.txt") == "detail\n"

    async def test_copy_is_independent(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        src = await async_factory.make("src")
        await src.create("/memories/shared.txt", "original\n")

        dst = await async_factory.make("dst", init_from="src")
        await dst.str_replace("/memories/shared.txt", "original", "modified")

        assert await src.read_file("/memories/shared.txt") == "original\n"
        assert await dst.read_file("/memories/shared.txt") == "modified\n"

    async def test_copy_skipped_when_namespace_not_empty(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        src = await async_factory.make("src")
        await src.create("/memories/src_only.txt", "from src\n")

        dst = await async_factory.make("dst")
        await dst.create("/memories/preexisting.txt", "already here\n")

        # Re-open the same namespace with init_from — should be a no-op
        dst2 = await async_factory.make("dst", init_from="src")
        assert await dst2.read_file("/memories/preexisting.txt") == "already here\n"
        assert await dst2.read_file("/memories/src_only.txt") is None

    async def test_transitive_copy_flat(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        ns1 = await async_factory.make("ns1")
        await ns1.create("/memories/data.txt", "original\n")

        ns2 = await async_factory.make("ns2", init_from="ns1")
        assert await ns2.read_file("/memories/data.txt") == "original\n"

        ns3 = await async_factory.make("ns3", init_from="ns2")
        assert await ns3.read_file("/memories/data.txt") == "original\n"

    async def test_transitive_copy_with_directories(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        ns1 = await async_factory.make("ns1")
        await ns1.create("/memories/category/item.txt", "item\n")
        await ns1.create("/memories/category/sub/deep.txt", "deep\n")

        ns2 = await async_factory.make("ns2", init_from="ns1")
        ns3 = await async_factory.make("ns3", init_from="ns2")

        assert await ns3.read_file("/memories/category/item.txt") == "item\n"
        assert await ns3.read_file("/memories/category/sub/deep.txt") == "deep\n"

    async def test_transitive_copy_independence(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        """Mutating ns2 (copy of ns1) must not affect ns3 (copy of ns2 taken before mutation)."""
        ns1 = await async_factory.make("ns1")
        await ns1.create("/memories/file.txt", "v1\n")

        ns2 = await async_factory.make("ns2", init_from="ns1")
        ns3 = await async_factory.make("ns3", init_from="ns2")

        await ns2.str_replace("/memories/file.txt", "v1", "v2")

        assert await ns1.read_file("/memories/file.txt") == "v1\n"
        assert await ns3.read_file("/memories/file.txt") == "v1\n"
        assert await ns2.read_file("/memories/file.txt") == "v2\n"


# ---------------------------------------------------------------------------
# Tool integration (FakeMessagesListChatModel + async LangGraph agent)
# ---------------------------------------------------------------------------


class TestAsyncMemoryToolIntegration:
    async def test_agent_creates_file(self, async_backend: AnyAsyncBackend) -> None:
        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "c1",
                        "memory",
                        {
                            "command": "create",
                            "path": "/memories/agent_note.txt",
                            "file_text": "agent wrote this\n",
                        },
                    )
                ],
            ),
            AIMessage(content="Done."),
        ]
        await run_async_agent(async_backend, responses)
        assert (
            await async_backend.read_file("/memories/agent_note.txt")
            == "agent wrote this\n"
        )

    async def test_agent_view_file_contents_returned_as_tool_message(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        await async_backend.create("/memories/preexisting.txt", "line one\nline two\n")

        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "v1",
                        "memory",
                        {"command": "view", "path": "/memories/preexisting.txt"},
                    )
                ],
            ),
            AIMessage(content="I read it."),
        ]
        messages = await run_async_agent(async_backend, responses)

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "line one" in tool_msgs[0].content
        assert "line two" in tool_msgs[0].content

    async def test_agent_create_then_rename(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "c1",
                        "memory",
                        {
                            "command": "create",
                            "path": "/memories/draft.txt",
                            "file_text": "draft content\n",
                        },
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "r1",
                        "memory",
                        {
                            "command": "rename",
                            "old_path": "/memories/draft.txt",
                            "new_path": "/memories/final.txt",
                        },
                    )
                ],
            ),
            AIMessage(content="All done."),
        ]
        await run_async_agent(async_backend, responses)
        assert await async_backend.read_file("/memories/draft.txt") is None
        assert (
            await async_backend.read_file("/memories/final.txt") == "draft content\n"
        )

    async def test_agent_str_replace(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/doc.txt", "The quick brown fox\n")
        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "s1",
                        "memory",
                        {
                            "command": "str_replace",
                            "path": "/memories/doc.txt",
                            "old_str": "brown fox",
                            "new_str": "lazy dog",
                        },
                    )
                ],
            ),
            AIMessage(content="Replaced."),
        ]
        await run_async_agent(async_backend, responses)
        assert (
            await async_backend.read_file("/memories/doc.txt")
            == "The quick lazy dog\n"
        )

    async def test_agent_reads_from_copied_namespace(
        self, async_factory: AsyncBackendFactory
    ) -> None:
        src = await async_factory.make("src")
        await src.create("/memories/knowledge.txt", "important knowledge\n")

        dst = await async_factory.make("dst", init_from="src")

        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "v1",
                        "memory",
                        {"command": "view", "path": "/memories/knowledge.txt"},
                    )
                ],
            ),
            AIMessage(content="Got it."),
        ]
        messages = await run_async_agent(dst, responses)

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "important knowledge" in tool_msgs[0].content

    async def test_agent_multiple_tool_calls_per_turn(
        self, async_backend: AnyAsyncBackend
    ) -> None:
        """The agent can issue multiple tool calls in a single AIMessage."""
        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "c1",
                        "memory",
                        {
                            "command": "create",
                            "path": "/memories/a.txt",
                            "file_text": "alpha\n",
                        },
                    ),
                    _tool_call(
                        "c2",
                        "memory",
                        {
                            "command": "create",
                            "path": "/memories/b.txt",
                            "file_text": "beta\n",
                        },
                    ),
                ],
            ),
            AIMessage(content="Created both."),
        ]
        await run_async_agent(async_backend, responses)
        assert await async_backend.read_file("/memories/a.txt") == "alpha\n"
        assert await async_backend.read_file("/memories/b.txt") == "beta\n"

    async def test_agent_list_directory(self, async_backend: AnyAsyncBackend) -> None:
        await async_backend.create("/memories/x.txt", "x")
        await async_backend.create("/memories/y.txt", "y")

        responses: list[BaseMessage] = [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "v1", "memory", {"command": "view", "path": "/memories"}
                    )
                ],
            ),
            AIMessage(content="I see the files."),
        ]
        messages = await run_async_agent(async_backend, responses)

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "x.txt" in tool_msgs[0].content
        assert "y.txt" in tool_msgs[0].content
