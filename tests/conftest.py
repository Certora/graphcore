import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Awaitable, Iterator,
    Mapping, Protocol, TYPE_CHECKING, cast,
    Generic, TypeVar, Callable
)

from pydantic import BaseModel
from typing_extensions import TypedDict

import pathlib

import psycopg
from psycopg.connection_async import AsyncConnection as AsyncPG
import pytest
import pytest_asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph._internal._typing import StateLike

from graphcore.tools.memory import (
    AsyncPostgresBackend,
    FileSystemMemoryBackend,
    MemoryToolImplForTest,
    PostgresMemoryBackend,
    SqliteMemoryBackend,
)


if TYPE_CHECKING:
    from testcontainers.postgres import PostgresContainer

try:
    from testcontainers.postgres import PostgresContainer

    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False

_MEMORIES_DDL = """
CREATE TABLE IF NOT EXISTS memories_fs(
    namespace TEXT NOT NULL,
    entry_name TEXT NOT NULL,
    full_path TEXT,
    parent_path TEXT,
    is_directory BOOL NOT NULL,
    contents TEXT,
    FOREIGN KEY(parent_path, namespace) REFERENCES memories_fs(full_path, namespace) ON DELETE CASCADE,
    UNIQUE (namespace, full_path),
    UNIQUE (namespace, parent_path, entry_name),
    CHECK (parent_path is NOT NULL OR (full_path = '/memories' AND is_directory AND entry_name = 'memories')),
    CHECK (parent_path is NULL OR (full_path = concat(parent_path, '/', entry_name))),
    CHECK (contents IS NOT NULL != is_directory)
);
CREATE INDEX IF NOT EXISTS memories_namespace_path ON memories_fs(namespace, full_path text_pattern_ops);
"""

_postgres_param = pytest.param(
    "postgres",
    marks=pytest.mark.skipif(
        not _HAS_TESTCONTAINERS,
        reason="testcontainers[postgres] not installed",
    ),
    id="postgres",
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

type AnyBackend = MemoryToolImplForTest[str, str | None]
type AnyAsyncBackend = MemoryToolImplForTest[Awaitable[str], Awaitable[str | None]]


class BackendFactory(Protocol):
    def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> AnyBackend: ...


class AsyncBackendFactory(Protocol):
    async def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> AnyAsyncBackend: ...


# ---------------------------------------------------------------------------
# Factory implementations
# ---------------------------------------------------------------------------


@dataclass
class SqliteFactory:
    """Creates :class:`SqliteMemoryBackend` instances sharing one connection."""

    conn: sqlite3.Connection
    _prefix: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _counter: int = field(default=0, init=False)

    def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> SqliteMemoryBackend:
        if ns is None:
            self._counter += 1
            ns = str(self._counter)
        real_ns = f"{self._prefix}_{ns}"
        real_init = f"{self._prefix}_{init_from}" if init_from else None
        return SqliteMemoryBackend(real_ns, self.conn, real_init)


@dataclass
class PostgresFactory:
    """Creates :class:`PostgresMemoryBackend` instances sharing one connection."""

    conn: psycopg.Connection[Any]
    _prefix: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _counter: int = field(default=0, init=False)

    def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> PostgresMemoryBackend:
        if ns is None:
            self._counter += 1
            ns = str(self._counter)
        real_ns = f"{self._prefix}_{ns}"
        real_init = f"{self._prefix}_{init_from}" if init_from else None
        return PostgresMemoryBackend(real_ns, self.conn, real_init)

@dataclass
class FilesystemFactory:
    """Creates :class:`FileSystemMemoryBackend` instances in a temp directory."""

    root: pathlib.Path
    _counter: int = field(default=0, init=False)

    def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> FileSystemMemoryBackend:
        if ns is None:
            self._counter += 1
            ns = str(self._counter)
        storage = self.root / ns
        storage.mkdir(parents=True, exist_ok=True)
        init_path = (self.root / init_from) if init_from is not None else None
        return FileSystemMemoryBackend(storage, init_path)


@pytest.fixture
def filesystem_factory(tmp_path: pathlib.Path) -> BackendFactory:
    return FilesystemFactory(tmp_path)


@pytest.fixture
def sqlite_factory() -> Iterator[BackendFactory]:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    yield SqliteFactory(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_container() -> Iterator[PostgresContainer]:
    """Start a Postgres container once per session and yield a conninfo URI."""
    if not _HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed")
    with PostgresContainer("postgres:16") as pg:
        yield pg

@pytest.fixture
def pg_connection(pg_container: PostgresContainer) -> Iterator[psycopg.Connection]:
    uniq_db = "test_" + uuid.uuid4().hex[:16]
    from psycopg.sql import SQL, Identifier

    with psycopg.connect(pg_container.get_connection_url(driver=None), autocommit=True) as conn:
        conn.execute(SQL("CREATE DATABASE {}").format(Identifier(uniq_db)))
    
    conn_string = f"postgresql://{pg_container.username}:{pg_container.password}@{pg_container.get_container_host_ip()}:{pg_container.get_exposed_port(5432)}/{uniq_db}"
    with psycopg.connect(conn_string) as to_yield:
        to_yield.execute(_MEMORIES_DDL)
        to_yield.commit()
        yield to_yield

    with psycopg.connect(pg_container.get_connection_url(driver=None), autocommit=True) as conn:
        conn.execute(SQL("DROP DATABASE {}").format(Identifier(uniq_db)))

@pytest.fixture
def postgres_factory(pg_connection: psycopg.Connection) -> Iterator[BackendFactory]:
    yield PostgresFactory(pg_connection)

@pytest.fixture(params=["filesystem", "sqlite", _postgres_param])
def factory(request: pytest.FixtureRequest) -> Iterator[BackendFactory]:
    """
    Parameterised factory — every test runs once per backend.

    Yields a factory whose ``make(ns, init_from)`` creates backends that
    share one DB connection but are namespace-isolated via a random prefix.
    """
    yield request.getfixturevalue(f"{request.param}_factory")

@pytest.fixture
def backend(factory: BackendFactory) -> AnyBackend:
    """Convenience: a single backend with an auto-generated namespace."""
    return factory.make()


# ---------------------------------------------------------------------------
# Async factory + fixtures (Postgres-only)
# ---------------------------------------------------------------------------


@dataclass
class AsyncPostgresFactory:
    """Creates :class:`AsyncPostgresBackend` instances sharing one connection."""

    conn: AsyncPG[Any]
    _prefix: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _counter: int = field(default=0, init=False)

    async def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> AsyncPostgresBackend:
        if ns is None:
            self._counter += 1
            ns = str(self._counter)
        real_ns = f"{self._prefix}_{ns}"
        real_init = f"{self._prefix}_{init_from}" if init_from else None
        to_ret = AsyncPostgresBackend(real_ns, self.conn)
        if real_init is not None:
            await to_ret.init_from(real_init)
        return to_ret

@pytest_asyncio.fixture
async def async_pg_connection(
    pg_container: PostgresContainer,
) -> AsyncIterator[AsyncPG[Any]]:
    """Per-test async connection to a throwaway database."""
    uniq_db = "test_async_" + uuid.uuid4().hex[:16]
    from psycopg.sql import SQL, Identifier

    # Create/drop the throwaway DB synchronously (admin connection).
    admin_url = pg_container.get_connection_url(driver=None)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("CREATE DATABASE {}").format(Identifier(uniq_db)))

    conn_string = (
        f"postgresql://{pg_container.username}:{pg_container.password}"
        f"@{pg_container.get_container_host_ip()}"
        f":{pg_container.get_exposed_port(5432)}/{uniq_db}"
    )
    aconn = await AsyncPG.connect(conn_string)
    await aconn.execute(_MEMORIES_DDL)
    await aconn.commit()
    try:
        yield aconn
    finally:
        await aconn.close()

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("DROP DATABASE {}").format(Identifier(uniq_db)))


@pytest_asyncio.fixture
async def async_postgres_factory(
    async_pg_connection: AsyncPG[Any],
) -> AsyncBackendFactory:
    return AsyncPostgresFactory(async_pg_connection)

@pytest_asyncio.fixture()
async def async_factory(async_postgres_factory) -> AsyncBackendFactory:
    return async_postgres_factory

@pytest_asyncio.fixture
async def async_backend(async_factory: AsyncBackendFactory) -> AnyAsyncBackend:
    return await async_factory.make()


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
