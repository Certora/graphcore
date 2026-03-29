from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Protocol, runtime_checkable

import psycopg
from psycopg.connection_async import AsyncConnection as AsyncPG
import pytest
import pytest_asyncio

from graphcore.tools.memory import (
    AsyncPostgresBackend,
    AsyncSQLBackend,
    PostgresMemoryBackend,
    SqliteMemoryBackend,
    SyncSqlBackend,
)

try:
    from testcontainers.postgres import PostgresContainer

    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False

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

# MemoryToolImpl covers the tool surface; read_file is the one extra method
# every test backend needs.
type AnyBackend = SyncSqlBackend[Any]
type AnyAsyncBackend = AsyncSQLBackend[Any]


@runtime_checkable
class BackendFactory(Protocol):
    def make(
        self, ns: str | None = None, init_from: str | None = None
    ) -> AnyBackend: ...


@runtime_checkable
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
        yield to_yield

    with psycopg.connect(pg_container.get_connection_url(driver=None), autocommit=True) as conn:
        conn.execute(SQL("DROP DATABASE {}").format(Identifier(uniq_db)))

@pytest.fixture
def postgres_factory(pg_connection: psycopg.Connection) -> Iterator[BackendFactory]:
    yield PostgresFactory(pg_connection)

@pytest.fixture(params=["sqlite", _postgres_param])
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
    try:
        yield aconn
    finally:
        await aconn.close()

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("DROP DATABASE {}").format(Identifier(uniq_db)))


@pytest_asyncio.fixture
async def async_factory(
    async_pg_connection: AsyncPG[Any],
) -> AsyncBackendFactory:
    return AsyncPostgresFactory(async_pg_connection)


@pytest_asyncio.fixture
async def async_backend(async_factory: AsyncBackendFactory) -> AnyAsyncBackend:
    return await async_factory.make()
