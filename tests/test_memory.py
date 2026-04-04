"""
Tests for the graphcore memory tool infrastructure.

Parameterised over SQLite and PostgreSQL (via testcontainers) backends.

Covers:
  - Direct backend operations (CRUD, directories, path validation)
  - Namespace copying (init_from) and transitive copying
  - Tool-level integration using FakeMessagesListChatModel + a minimal LangGraph agent
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from graphcore.tools.memory import MemoryBackendError, MemoryToolImpl, memory_tool

from .conftest import AnyBackend, BackendFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def run_agent(
    backend: MemoryToolImpl[str], llm_responses: list[BaseMessage]
) -> list[BaseMessage]:
    """
    Build a minimal ReAct-style agent wired to the memory tool, drive it with
    ``FakeMessagesListChatModel``, and return the full message list.
    """
    tool = memory_tool(backend)
    tool_node = ToolNode([tool], handle_tool_errors=False)
    llm = FakeMessagesListChatModel(responses=llm_responses)

    def agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

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

    result = graph.compile().invoke({"messages": [HumanMessage(content="Do the task.")]})
    return result["messages"]


# ---------------------------------------------------------------------------
# Backend-direct: basic CRUD
# ---------------------------------------------------------------------------


class TestBasicCRUD:
    def test_create_and_read(self, backend: AnyBackend) -> None:
        result = backend.create("/memories/hello.txt", "hello world\n")
        assert "created" in result.lower()
        assert backend.read_file("/memories/hello.txt") == "hello world\n"

    def test_view_file_numbered(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "alpha\nbeta\ngamma\n")
        viewed = backend.view("/memories/f.txt", None)
        assert "alpha" in viewed
        assert "beta" in viewed
        assert "gamma" in viewed

    def test_view_file_range(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "line1\nline2\nline3\nline4\n")
        viewed = backend.view("/memories/f.txt", (2, 3))
        assert "line2" in viewed
        assert "line3" in viewed
        assert "line1" not in viewed
        assert "line4" not in viewed

    def test_view_nonexistent_returns_error(self, backend: AnyBackend) -> None:
        result = backend.view("/memories/nope.txt", None)
        assert "ERROR" in result

    def test_overwrite_file(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "v1\n")
        backend.create("/memories/f.txt", "v2\n")
        assert backend.read_file("/memories/f.txt") == "v2\n"

    def test_str_replace(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "hello world\n")
        backend.str_replace("/memories/f.txt", "hello", "goodbye")
        assert backend.read_file("/memories/f.txt") == "goodbye world\n"

    def test_str_replace_not_found_returns_error(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "hello world\n")
        result = backend.str_replace("/memories/f.txt", "foobar", "baz")
        assert "ERROR" in result

    def test_str_replace_ambiguous_returns_error(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "hello hello\n")
        result = backend.str_replace("/memories/f.txt", "hello", "world")
        assert "ERROR" in result

    def test_insert_mid_file(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "line1\nline3\n")
        backend.insert("/memories/f.txt", 1, "line2")
        assert backend.read_file("/memories/f.txt") == "line1\nline2\nline3\n"

    def test_insert_at_start(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "line2\n")
        backend.insert("/memories/f.txt", 0, "line1")
        assert backend.read_file("/memories/f.txt") == "line1\nline2\n"

    def test_insert_at_end(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "line1\n")
        backend.insert("/memories/f.txt", 1, "line2")
        assert backend.read_file("/memories/f.txt") == "line1\nline2\n"

    def test_insert_out_of_bounds_returns_error(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "line1\n")
        result = backend.insert("/memories/f.txt", 99, "extra")
        assert "ERROR" in result

    def test_delete_file(self, backend: AnyBackend) -> None:
        backend.create("/memories/f.txt", "data")
        backend.delete("/memories/f.txt")
        assert backend.read_file("/memories/f.txt") is None


# ---------------------------------------------------------------------------
# Backend-direct: directories
# ---------------------------------------------------------------------------


class TestDirectories:
    def test_auto_mkdir_on_create(self, backend: AnyBackend) -> None:
        backend.create("/memories/a/b/c/deep.txt", "deep")
        assert backend.read_file("/memories/a/b/c/deep.txt") == "deep"

    def test_view_directory_lists_children(self, backend: AnyBackend) -> None:
        backend.create("/memories/file1.txt", "a")
        backend.create("/memories/file2.txt", "b")
        listing = backend.view("/memories", None)
        assert "file1.txt" in listing
        assert "file2.txt" in listing

    def test_view_directory_shows_subdirs(self, backend: AnyBackend) -> None:
        backend.create("/memories/subdir/file.txt", "x")
        listing = backend.view("/memories", None)
        assert "subdir" in listing

    def test_delete_directory_cascades_to_children(self, backend: AnyBackend) -> None:
        backend.create("/memories/dir/file1.txt", "a")
        backend.create("/memories/dir/nested/file2.txt", "b")
        backend.delete("/memories/dir")
        assert backend.read_file("/memories/dir/file1.txt") is None
        assert backend.read_file("/memories/dir/nested/file2.txt") is None


# ---------------------------------------------------------------------------
# Backend-direct: rename
# ---------------------------------------------------------------------------


class TestRename:
    def test_rename_file(self, backend: AnyBackend) -> None:
        backend.create("/memories/old.txt", "content\n")
        result = backend.rename("/memories/old.txt", "/memories/new.txt")
        assert "Renamed" in result
        assert backend.read_file("/memories/old.txt") is None
        assert backend.read_file("/memories/new.txt") == "content\n"

    def test_rename_file_to_new_subdir(self, backend: AnyBackend) -> None:
        backend.create("/memories/flat.txt", "data\n")
        backend.rename("/memories/flat.txt", "/memories/sub/flat.txt")
        assert backend.read_file("/memories/flat.txt") is None
        assert backend.read_file("/memories/sub/flat.txt") == "data\n"

    def test_rename_directory_updates_all_descendants(self, backend: AnyBackend) -> None:
        backend.create("/memories/old_dir/file.txt", "top\n")
        backend.create("/memories/old_dir/child/nested.txt", "nested\n")
        backend.rename("/memories/old_dir", "/memories/new_dir")
        assert backend.read_file("/memories/old_dir/file.txt") is None
        assert backend.read_file("/memories/old_dir/child/nested.txt") is None
        assert backend.read_file("/memories/new_dir/file.txt") == "top\n"
        assert backend.read_file("/memories/new_dir/child/nested.txt") == "nested\n"

    def test_rename_nonexistent_returns_error(self, backend: AnyBackend) -> None:
        result = backend.rename("/memories/ghost.txt", "/memories/new.txt")
        assert "ERROR" in result

    def test_rename_destination_exists_returns_error(self, backend: AnyBackend) -> None:
        backend.create("/memories/a.txt", "a")
        backend.create("/memories/b.txt", "b")
        result = backend.rename("/memories/a.txt", "/memories/b.txt")
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# Backend-direct: path validation
# ---------------------------------------------------------------------------


class TestPathValidation:
    def test_create_outside_memories_raises(self, backend: AnyBackend) -> None:
        with pytest.raises(MemoryBackendError, match="/memories"):
            backend.create("/tmp/evil.txt", "bad")

    def test_view_outside_memories_raises(self, backend: AnyBackend) -> None:
        with pytest.raises(MemoryBackendError):
            backend.view("/etc/passwd", None)

    def test_rename_source_outside_memories_raises(self, backend: AnyBackend) -> None:
        with pytest.raises(MemoryBackendError):
            backend.rename("/etc/passwd", "/memories/stolen.txt")

    def test_path_traversal_rejected(self, backend: AnyBackend) -> None:
        with pytest.raises(MemoryBackendError):
            backend.create("/memories/../etc/shadow", "evil")


# ---------------------------------------------------------------------------
# Namespace copying (init_from)
# ---------------------------------------------------------------------------


class TestNamespaceCopy:
    def test_copy_flat_files(self, factory: BackendFactory) -> None:
        src = factory.make("src")
        src.create("/memories/note.txt", "important\n")
        src.create("/memories/other.txt", "also important\n")

        dst = factory.make("dst", init_from="src")
        assert dst.read_file("/memories/note.txt") == "important\n"
        assert dst.read_file("/memories/other.txt") == "also important\n"

    def test_copy_includes_nested_directories(self, factory: BackendFactory) -> None:
        src = factory.make("src")
        src.create("/memories/docs/readme.txt", "readme\n")
        src.create("/memories/docs/sub/detail.txt", "detail\n")

        dst = factory.make("dst", init_from="src")
        assert dst.read_file("/memories/docs/readme.txt") == "readme\n"
        assert dst.read_file("/memories/docs/sub/detail.txt") == "detail\n"

    def test_copy_is_independent(self, factory: BackendFactory) -> None:
        src = factory.make("src")
        src.create("/memories/shared.txt", "original\n")

        dst = factory.make("dst", init_from="src")
        dst.str_replace("/memories/shared.txt", "original", "modified")

        assert src.read_file("/memories/shared.txt") == "original\n"
        assert dst.read_file("/memories/shared.txt") == "modified\n"

    def test_copy_skipped_when_namespace_not_empty(self, factory: BackendFactory) -> None:
        src = factory.make("src")
        src.create("/memories/src_only.txt", "from src\n")

        dst = factory.make("dst")
        dst.create("/memories/preexisting.txt", "already here\n")

        # Re-open the same namespace with init_from — should be a no-op
        dst2 = factory.make("dst", init_from="src")
        assert dst2.read_file("/memories/preexisting.txt") == "already here\n"
        assert dst2.read_file("/memories/src_only.txt") is None

    def test_transitive_copy_flat(self, factory: BackendFactory) -> None:
        ns1 = factory.make("ns1")
        ns1.create("/memories/data.txt", "original\n")

        ns2 = factory.make("ns2", init_from="ns1")
        assert ns2.read_file("/memories/data.txt") == "original\n"

        ns3 = factory.make("ns3", init_from="ns2")
        assert ns3.read_file("/memories/data.txt") == "original\n"

    def test_transitive_copy_with_directories(self, factory: BackendFactory) -> None:
        ns1 = factory.make("ns1")
        ns1.create("/memories/category/item.txt", "item\n")
        ns1.create("/memories/category/sub/deep.txt", "deep\n")

        ns2 = factory.make("ns2", init_from="ns1")
        ns3 = factory.make("ns3", init_from="ns2")

        assert ns3.read_file("/memories/category/item.txt") == "item\n"
        assert ns3.read_file("/memories/category/sub/deep.txt") == "deep\n"

    def test_transitive_copy_independence(self, factory: BackendFactory) -> None:
        """Mutating ns2 (copy of ns1) must not affect ns3 (copy of ns2 taken before mutation)."""
        ns1 = factory.make("ns1")
        ns1.create("/memories/file.txt", "v1\n")

        ns2 = factory.make("ns2", init_from="ns1")
        ns3 = factory.make("ns3", init_from="ns2")

        ns2.str_replace("/memories/file.txt", "v1", "v2")

        assert ns1.read_file("/memories/file.txt") == "v1\n"
        assert ns3.read_file("/memories/file.txt") == "v1\n"
        assert ns2.read_file("/memories/file.txt") == "v2\n"


# ---------------------------------------------------------------------------
# Tool integration (FakeMessagesListChatModel + LangGraph agent)
# ---------------------------------------------------------------------------


class TestMemoryToolIntegration:
    def test_agent_creates_file(self, backend: AnyBackend) -> None:
        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("c1", "memory", {
                    "command": "create",
                    "path": "/memories/agent_note.txt",
                    "file_text": "agent wrote this\n",
                })
            ]),
            AIMessage(content="Done."),
        ]
        run_agent(backend, responses)
        assert backend.read_file("/memories/agent_note.txt") == "agent wrote this\n"

    def test_agent_view_file_contents_returned_as_tool_message(
        self, backend: AnyBackend
    ) -> None:
        backend.create("/memories/preexisting.txt", "line one\nline two\n")

        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("v1", "memory", {
                    "command": "view",
                    "path": "/memories/preexisting.txt",
                })
            ]),
            AIMessage(content="I read it."),
        ]
        messages = run_agent(backend, responses)

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "line one" in tool_msgs[0].content
        assert "line two" in tool_msgs[0].content

    def test_agent_create_then_rename(self, backend: AnyBackend) -> None:
        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("c1", "memory", {
                    "command": "create",
                    "path": "/memories/draft.txt",
                    "file_text": "draft content\n",
                })
            ]),
            AIMessage(content="", tool_calls=[
                _tool_call("r1", "memory", {
                    "command": "rename",
                    "old_path": "/memories/draft.txt",
                    "new_path": "/memories/final.txt",
                })
            ]),
            AIMessage(content="All done."),
        ]
        run_agent(backend, responses)
        assert backend.read_file("/memories/draft.txt") is None
        assert backend.read_file("/memories/final.txt") == "draft content\n"

    def test_agent_str_replace(self, backend: AnyBackend) -> None:
        backend.create("/memories/doc.txt", "The quick brown fox\n")
        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("s1", "memory", {
                    "command": "str_replace",
                    "path": "/memories/doc.txt",
                    "old_str": "brown fox",
                    "new_str": "lazy dog",
                })
            ]),
            AIMessage(content="Replaced."),
        ]
        run_agent(backend, responses)
        assert backend.read_file("/memories/doc.txt") == "The quick lazy dog\n"

    def test_agent_reads_from_copied_namespace(self, factory: BackendFactory) -> None:
        src = factory.make("src")
        src.create("/memories/knowledge.txt", "important knowledge\n")

        dst = factory.make("dst", init_from="src")

        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("v1", "memory", {
                    "command": "view",
                    "path": "/memories/knowledge.txt",
                })
            ]),
            AIMessage(content="Got it."),
        ]
        messages = run_agent(dst, responses)

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "important knowledge" in tool_msgs[0].content

    def test_agent_multiple_tool_calls_per_turn(self, backend: AnyBackend) -> None:
        """The agent can issue multiple tool calls in a single AIMessage."""
        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("c1", "memory", {
                    "command": "create",
                    "path": "/memories/a.txt",
                    "file_text": "alpha\n",
                }),
                _tool_call("c2", "memory", {
                    "command": "create",
                    "path": "/memories/b.txt",
                    "file_text": "beta\n",
                }),
            ]),
            AIMessage(content="Created both."),
        ]
        run_agent(backend, responses)
        assert backend.read_file("/memories/a.txt") == "alpha\n"
        assert backend.read_file("/memories/b.txt") == "beta\n"

    def test_agent_list_directory(self, backend: AnyBackend) -> None:
        backend.create("/memories/x.txt", "x")
        backend.create("/memories/y.txt", "y")

        responses: list[BaseMessage] = [
            AIMessage(content="", tool_calls=[
                _tool_call("v1", "memory", {"command": "view", "path": "/memories"})
            ]),
            AIMessage(content="I see the files."),
        ]
        messages = run_agent(backend, responses)

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "x.txt" in tool_msgs[0].content
        assert "y.txt" in tool_msgs[0].content
