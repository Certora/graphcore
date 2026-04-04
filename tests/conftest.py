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

