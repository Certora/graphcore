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

# =============================================================================
# Memory tool architecture
# =============================================================================
#
# Three layers, each one a translation of the one above:
#
#   PURE LOGIC  ──>  BACKEND  ──>  TOOL
#  (effects as     (drives the      (langchain BaseTool,
#   generators)     generators       provider-specific
#                   against I/O)     schema)
#
# ── Pure layer (`PureMemoryBackend[P, Update, Row, RList]`) ────────────────
#   The semantic operations (view / create / str_replace / insert / rename /
#   delete, plus their _pure helpers) are written as generators that *yield*
#   I/O requests of type `P` and *receive* I/O results (`Row` for one-row
#   reads, `RList` for multi-row reads, `Update` for write row-counts).
#   These generators do no I/O themselves — they describe the sequence of
#   primitive operations they need, and each primitive op is itself a
#   generator method (`read_file_pure`, `write_file_pure`, `stat_pure`,
#   `list_dir_pure`, `rm_pure`, `do_rename_pure`) that subclasses define in
#   terms of `P`.
#
#   Why this shape? **It solves the function-coloring problem.** Python's
#   `async` is infectious — if the core "view a file with a line range"
#   logic awaited an I/O call, every caller up the stack would have to be
#   async too, and you couldn't share that body between a sync driver
#   (e.g. a synchronous Postgres connection) and an async driver (e.g. an
#   async pool). By representing I/O as yielded requests instead of awaited
#   calls, the core generator is *colorless*: a sync driver advances it
#   with `next()` / `.send(result)`, an async driver does the same with
#   `await some_io(...)` between the steps. Same logic, two drivers.
#
#   That's also how `SQLBackendPure` (`P = (sql, params)`, yields SQL,
#   consumes rows) ends up shared by both `SyncSqlBackend` and
#   `AsyncSQLBackend`. `PureFilesystemLogic` uses `P = Never` via
#   `to_generator` since its primitives don't need a wedge — but the same
#   `_view_file_pure` / `create_pure` / etc. logic is reused by both
#   `FileSystemMemoryBackend` and `AsyncFileMemoryBackend`.
#
# ── Backend layer (`MemoryBackend` / `AsyncMemoryBackend`) ─────────────────
#   These are the drivers. They hold a `PureMemoryBackend` (`self.logic`)
#   and implement `_run_row` / `_run_multi` / `_run_update`: feed the
#   generator, get a request `P`, perform the I/O, send the result back into
#   the generator, repeat until `StopIteration`.
#
#   Concrete impls:
#     • `PostgresMemoryBackend`, `SqliteMemoryBackend` — sync SQL drivers
#       holding a real connection/pool and a `SQLBackendPure` logic.
#     • `AsyncPostgresBackend` — async variant of the same.
#     • `FileSystemMemoryBackend` / `AsyncFileMemoryBackend` — drivers over
#       `PureFilesystemLogic` whose `_run_*` methods just exhaust the
#       generator (no I/O wedge needed; primitives already did the work).
#
#   The public surface of a Backend is the six `MemoryToolImpl` methods
#   (view, create, delete, rename, insert, str_replace). Sync impls return
#   `str`; async impls return `Awaitable[str]`. This is what the tool layer
#   consumes.
#
# ── Tool layer (`*_memory_tool` factories) ────────────────────────────────
#   `MemoryToolImpl[R]` is a structural Protocol — both `MemoryBackend` and
#   `AsyncMemoryBackend` satisfy it, parameterized on `R = str` vs
#   `R = Awaitable[str]`. The tool factories close over a `MemoryToolImpl`
#   and wrap it in a `BaseTool` with a provider-specific args schema:
#
#     • `memory_tool`            — sync,  Anthropic schema (UnifiedMemorySchema)
#     • `async_memory_tool`      — async, Anthropic schema
#     • `openai_memory_tool`     — sync,  OpenAI schema (_OpenAIMemorySchema)
#     • `openai_async_memory_tool` — async, OpenAI schema
#
#   Anthropic's `UnifiedMemorySchema` is a flat bag of nullable fields,
#   shape-matched to what Anthropic's trained-on memory tool emits. The
#   Anthropic Files-API beta (`memory_20250818`) replaces this schema
#   server-side, so the "sparse" tool and field documentations don't matter.
#
#   OpenAI's `_OpenAIMemorySchema` wraps a single `memory_op` field whose
#   type is a Pydantic discriminated union over six variant BaseModels
#   (one per command). Top level is `type: "object"` so strict-mode JSON
#   schema validation works; the `anyOf` lives inside `memory_op`.
#
#   Dispatch: `_memory_tool_impl` handles the flat-bag → backend-method
#   path; `_dispatch_openai_op` handles the sum-type → backend-method path.
#   Both end up calling the same six MemoryToolImpl methods.
# =============================================================================

from typing import (
    Annotated, Literal, Optional, override, Protocol, Any, TypeVar, Iterator,
    ContextManager, LiteralString, cast, Generator,
    AsyncContextManager, AsyncIterator, Callable, Awaitable, Sequence,
    Never, ParamSpec
)
from typing_extensions import Iterable
from abc import ABC, abstractmethod
import shutil
import sqlite3
from dataclasses import dataclass
import pathlib
from contextlib import contextmanager, asynccontextmanager
import re

from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from psycopg_pool.pool_async import AsyncConnectionPool
from psycopg.connection_async import AsyncConnection

from langchain_core.tools import BaseTool

from pydantic import BaseModel, Field

from .schemas import WithImplementation, WithAsyncImplementation

class UnifiedMemorySchema(BaseModel):
    """
    unified tool schema. Not actually sent to the LLM, but necessary
    to adapt the rigid input schemas of langgraph tools
    to the "dependent" schema used by the anthropic memory tool.

    This comment is just here to make the tool annotation happy.
    """
    command: Literal["view", "create", "str_replace", "insert", "delete", "rename"] = Field(
        description="Memory command: create, update, delete, search, etc."
    )
    path: Optional[str] = Field(default=None, description="path to view/create/str_replace/delete")
    view_range: Optional[list[int]] = Field(default=None, description="range for view")
    # Include all possible fields across commands
    file_text: Optional[str] = Field(default=None, description="Content for create")

    old_str: Optional[str] = Field(description="old string for str_replace", default=None)
    new_str: Optional[str] = Field(description="new string for str_replace", default=None)

    insert_line: Optional[int] = Field(description="Line number at which to insert", default=None)
    insert_text: Optional[str] = Field(description="Text to insert", default=None)

    old_path: Optional[str] = Field(description="original path", default=None)
    new_path: Optional[str] = Field(description="new path", default=None)

@dataclass
class FileStat:
    exists: bool
    is_dir: bool


class MemoryBackendError(RuntimeError):
    """
    Special exception caught by the memory tool and transformed into an error message
    for the LLM.
    """
    def __init__(self, msg: str):
        super().__init__(msg)

class PureMemoryBackend[P, Update, Row, RList](ABC):

    @abstractmethod
    def read_file_pure(self, path: str) -> Generator[P, Row, Optional[str]]:
        ...

    @abstractmethod
    def write_file_pure(self, path: str, content: str) -> Generator[P, Row, None]:
        ...

    @abstractmethod
    def stat_pure(self, path: str) -> Generator[P, Row, FileStat]:
        ...

    @abstractmethod
    def list_dir_pure(self, path: str) -> Generator[P, RList, list[tuple[str, bool]]]:
        ...

    @abstractmethod
    def rm_pure(self, path: str) -> Generator[P, Update, str]:
        ...

    @abstractmethod
    def do_rename_pure(self, old_path: str, new_path: str) -> Generator[P, Row, str]:
        ...

    def _validate(self, path: str) -> str:
        r = pathlib.PurePosixPath(path)
        if not r.is_absolute() or not r.is_relative_to("/memories"):
            raise MemoryBackendError(f"{str(r)} is not rooted in /memories")

        return str(r)
    
    def _view_file_pure(self, path: str, view_range: tuple[int, int] | None) -> Generator[P, Row, str]:
        content = yield from self.read_file_pure(path)
        assert content is not None
        lines = content.splitlines()
        if view_range:
            start_line = max(1, view_range[0]) - 1
            end_line = len(lines) if view_range[1] == -1 else view_range[1]
            lines = lines[start_line:end_line]
            start_num = start_line + 1
        else:
            start_num = 1
        numbered_lines = [f"{i + start_num:4d}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def _view_dir_pure(self, path: str) -> Generator[P, RList, str]:
        to_ret = []
        dir_list = yield from self.list_dir_pure(path)
        for (p, is_dir) in dir_list:
            child_path = p
            if is_dir:
                child_path += "/"
            to_ret.append(f"- {child_path}")
        return "\n".join(to_ret)

    def create_pure(self, path: str, content: str) -> Generator[P, Row, str]:
        if path == "/memories":
            return "`/memories` is a directory and cannot be written to."
        curr_path = self._validate(path)
        stat = yield from self.stat_pure(curr_path)
        if stat.exists and stat.is_dir:
            return f"Error: {path} exists and is a directory"
        yield from self.write_file_pure(self._validate(path), content)
        return f"File created successfully at {path}"

    def delete_pure(self, path: str) -> Generator[P, Update, str]:
        path = self._validate(path)
        if path == "/memories":
            return "Error: cannot delete the /memories directory"

        to_ret = yield from self.rm_pure(path)
        return to_ret

    def rename_pure(self, old_path: str, new_path: str) -> Generator[P, Row, str]:
        old_path = self._validate(old_path)
        new_path = self._validate(new_path)

        old_stat = yield from self.stat_pure(old_path)
        new_stat = yield from self.stat_pure(new_path)
        if not old_stat.exists:
            return (f"ERROR: Source path not found: {old_path}")
        if new_stat.exists:
            return (f"ERROR: Destination already exists: {new_path}")

        to_ret = yield from self.do_rename_pure(old_path, new_path)
        return to_ret

    def insert_pure(self, path: str, insert_line: int, insert_text: str) -> Generator[P, Row, str]:
        path = self._validate(path)
        stat = yield from self.stat_pure(path)

        if not stat.exists or stat.is_dir:
            return (f"ERROR: File not found: {path}")
        contents = yield from self.read_file_pure(path)
        assert contents is not None
        lines = contents.splitlines()

        if insert_line < 0 or insert_line > len(lines):
            return (f"ERROR: Invalid insert_line {insert_line}. Must be 0-{len(lines)}")

        lines.insert(insert_line, insert_text.rstrip("\n"))

        to_write = "\n".join(lines) + "\n"
        yield from self.write_file_pure(path, to_write)

        return f"Text inserted at line {insert_line} in {path}"

    def str_replace_pure(self, path: str, old_str: str, new_str: str) -> Generator[P, Row, str]:
        path = self._validate(path)
        full_path = yield from self.stat_pure(path)

        if not full_path.exists or full_path.is_dir:
            return f"File not found: {path}"

        content = yield from self.read_file_pure(path)
        assert content is not None

        count = content.count(old_str)
        if count == 0:
            return (f"ERROR: Text not found in {path}")
        elif count > 1:
            return (f"ERROR: Text appears {count} times in {path}. Must be unique.")

        new_content = content.replace(old_str, new_str)
        yield from self.write_file_pure(path, new_content)
        return f"File {path} has been edited"

class MemoryToolImpl[RespType](Protocol):
    def view(self, path: str, view_range: Optional[tuple[int, int]]) -> RespType:
        ...

    def create(self, path: str, content: str) -> RespType:
        ...

    def delete(self, path: str) -> RespType:
        ...

    def rename(self, old_path: str, new_path: str) -> RespType:
        ...

    def insert(self, path: str, insert_line: int, insert_text: str) -> RespType:
        ...

    def str_replace(self, path: str, old_str: str, new_str: str) -> RespType:
        ...


class MemoryToolImplForTest[RespType, ReadResp](MemoryToolImpl[RespType], Protocol):
    """Extends MemoryToolImpl with read_file for test assertions only."""
    def read_file(self, path: str) -> ReadResp:
        ...

class MemoryBackend[P, Update, Row, RList](ABC):
    """
    Implements most of the operations required by the memory tool, but leaves the details of
    - file stat
    - directory listing
    - creating/reading/deleting files/folders
    - renaming

    To the concrete backends. Most of implementations here were actually adapted from Anthropic's reference
    code.
    """

    def __init__(self, logic: PureMemoryBackend[P, Update, Row, RList]):
        self.logic = logic

    @abstractmethod
    def _run_multi[T](
        self, d: Generator[P, RList, T]
    ) -> T:
        ...

    @abstractmethod
    def _run_row[T](
        self, d: Generator[P, Row, T]
    ) -> T:
        ...

    @abstractmethod
    def _run_update[T](
        self, d: Generator[P, Update, T]
    ) -> T:
        ...

    def read_file(self, path: str) -> Optional[str]:
        """Test-only: read raw file content bypassing path validation and view formatting."""
        return self._run_row(
            self.logic.read_file_pure(path)
        )

    def stat(self, path: str) -> FileStat:
        return self._run_row(
            self.logic.stat_pure(path)
        )

    def list_dir(self, path: str) -> Iterable[tuple[str, bool]]:
        return self._run_multi(
            self.logic.list_dir_pure(path)
        )

    def do_rename(self, old_path: str, new_path: str) -> str:
        return self._run_row(
            self.logic.do_rename_pure(old_path, new_path)
        )
    
    def view(self, path: str, view_range: Optional[tuple[int, int]]) -> str:
        path = self.logic._validate(path)
        s = self.stat(path)
        if not s.exists:
            return f"ERROR: Path not found: {path}"

        if s.is_dir:
            return self._run_multi(self.logic._view_dir_pure(path))

        else:
            return self._run_row(self.logic._view_file_pure(path, view_range))

    def create(self, path: str, content: str) -> str:
        return self._run_row(
            self.logic.create_pure(path, content)
        )

    def delete(self, path: str) -> str:
        return self._run_update(self.logic.rm_pure(path))

    def rename(self, old_path: str, new_path: str) -> str:
        return self._run_row(
            self.logic.rename_pure(old_path, new_path)
        )

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        return self._run_row(
            self.logic.insert_pure(path, insert_line, insert_text)
        )

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        return self._run_row(
            self.logic.str_replace_pure(path, old_str, new_str)
        )




class AsyncMemoryBackend[P, Update, Row, RList](ABC):

    def __init__(self, logic: PureMemoryBackend[P, Update, Row, RList]):
        self.logic = logic

    @abstractmethod
    async def _run_multi[T](
        self, d: Generator[P, RList, T]
    ) -> T:
        ...

    @abstractmethod
    async def _run_row[T](
        self, d: Generator[P, Row, T]
    ) -> T:
        ...

    @abstractmethod
    async def _run_update[T](
        self, d: Generator[P, Update, T]
    ) -> T:
        ...

    async def read_file(self, path: str) -> Optional[str]:
        """Test-only: read raw file content bypassing path validation and view formatting."""
        return await self._run_row(
            self.logic.read_file_pure(path)
        )

    async def stat(self, path: str) -> FileStat:
        return await self._run_row(
            self.logic.stat_pure(path)
        )

    async def list_dir(self, path: str) -> Iterable[tuple[str, bool]]:
        return await self._run_multi(
            self.logic.list_dir_pure(path)
        )

    async def do_rename(self, old_path: str, new_path: str) -> str:
        return await self._run_row(
            self.logic.do_rename_pure(old_path, new_path)
        )
    
    async def view(self, path: str, view_range: Optional[tuple[int, int]]) -> str:
        path = self.logic._validate(path)
        s = await self.stat(path)
        if not s.exists:
            return f"ERROR: Path not found: {path}"

        if s.is_dir:
            return await self._run_multi(self.logic._view_dir_pure(path))

        else:
            return await self._run_row(self.logic._view_file_pure(path, view_range))

    async def create(self, path: str, content: str) -> str:
        return await self._run_row(
            self.logic.create_pure(path, content)
        )

    async def delete(self, path: str) -> str:
        return await self._run_update(self.logic.rm_pure(path))

    async def rename(self, old_path: str, new_path: str) -> str:
        return await self._run_row(
            self.logic.rename_pure(old_path, new_path)
        )

    async def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        return await self._run_row(
            self.logic.insert_pure(path, insert_line, insert_text)
        )

    async def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        return await self._run_row(
            self.logic.str_replace_pure(path, old_str, new_str)
        )


class DBCursorG[Q](Protocol):
    """
    Intersection type necessary for the SQLBackend, implemented by both the
    postgres and sqlite cursor classes.
    """
    def execute(self, query: Q, vars: tuple[Any, ...] | dict[str, Any], /) -> Any:
        ...

    def fetchone(self, /) -> None | tuple[Any, ...]:
        ...

    def close(self) -> None:
        ...

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        ...

    @property
    def rowcount(self) -> int:
        ...
    
    @property
    def description(self) -> Sequence[Any] | None:
        ...

class AsyncCursorG[Q](Protocol):
    async def execute(self, query: Q, params: tuple[Any, ...] | dict[str, Any], /) -> Any:
        ...

    async def fetchone(self, /) -> None | tuple[Any, ...]:
        ...

    async def close(self) -> None:
        ...

    def __aiter__(self) -> AsyncIterator[tuple[Any, ...]]:
        ...

    @property
    def rowcount(self) -> int:
        ...
    
    @property
    def description(self) -> Sequence[Any] | None:
        ...


type DBCursor = DBCursorG[str]
type AsyncDBCursor = AsyncCursorG[str]

CONN = TypeVar('CONN')

CURSOR = TypeVar('CURSOR', bound=DBCursor)

type _SqlParams = tuple[Any, ...] | dict[str, Any]
type _SqlRow = tuple[Any, ...]
type _SqlQuery = tuple[str, _SqlParams]

type _SqlRowProtocol[T] = Generator[
    _SqlQuery,
    _SqlRow | None,
    T
]

type _SqlUpdateProtocol[T] = Generator[
    _SqlQuery,
    int,
    T
]

type _SqlMultiRowProtocol[T] = Generator[
    _SqlQuery,
    list[_SqlRow],
    T
]

class SQLBackendPure(PureMemoryBackend[_SqlQuery, int, _SqlRow | None, list[_SqlRow]]):
    def __init__(self, ns: str):
        self.ns = ns

    @abstractmethod
    def named_placeholder(self, nm: str) -> str:
        """
        again, sqlite uses :name for named placeholders, postgres uses %(name)s. push
        this into the implementers.
        """
        ...

    @property
    @abstractmethod
    def pos_placeholder(self) -> str:
        """
        sqlite's driver uses ? for placeholder, postgres uses %s. make this a property
        """
        ...


    def _init_from(self, other_ns: str) -> _SqlRowProtocol[None]:
        yield from self._execute(f"""
INSERT INTO memories_fs(namespace, full_path, entry_name, parent_path, is_directory, contents)
SELECT {self.pos_placeholder}, full_path, entry_name, parent_path, is_directory, contents FROM memories_fs as s
WHERE s.namespace = {self.pos_placeholder}
                    """, (self.ns, other_ns))
        return None
    
    def _is_empty(self) -> _SqlRowProtocol[bool]:
        r = yield from self._execute(f"SELECT count(*) FROM memories_fs WHERE namespace = {self.pos_placeholder} AND full_path != '/memories'", (self.ns,))
        return r is not None and r[0] == 0

    def _execute(self, query: str, params: tuple[Any, ...] | dict[str, Any]) -> _SqlRowProtocol[tuple[Any, ...] | None]:
        res = yield (query, params)
        return res
    
    def _execute_multi(self, query: str, params: tuple[Any, ...] | dict[str, Any]) -> _SqlMultiRowProtocol[list[tuple[Any, ...]]]:
        res = yield (query, params)
        return res
    
    def _execute_update(self, query: str, params: tuple[Any, ...] | dict[str, Any]) -> _SqlUpdateProtocol[int]:
        res = yield (query, params)
        return res

    def _mkdirs(self, path: pathlib.Path) -> _SqlRowProtocol[str | None]:
        if str(path) == '/':
            return None
        parent = yield from self._mkdirs(path.parent)
        r = yield from self._execute(f"""
SELECT is_directory FROM memories_fs WHERE namespace = {self.pos_placeholder} AND full_path = {self.pos_placeholder}
                       """, (self.ns, str(path)))
        if r is not None:
            if not r[0]:
                raise MemoryBackendError(f"Cannot create directory at {path}: path already exists and is a file")
            return str(path)
        curr_name = path.name
        _ = yield from self._execute(f"""
INSERT INTO memories_fs(
    namespace,
    entry_name,
    full_path,
    parent_path,
    is_directory
) VALUES ({self.pos_placeholder}, {self.pos_placeholder}, {self.pos_placeholder}, {self.pos_placeholder}, TRUE);
                       """, (self.ns, curr_name, str(path), parent))
        return str(path)

    @abstractmethod
    def do_replace_first(self, replace_attribute: str, to_replace: str, replace_with: str, seq: int) -> tuple[str, dict[str, str]]:
        """
        Generate sql syntax to replace the first occurence of to_replace with replace_with in the column replace_attribute.

        Return a tuple of the sql, and a dictionary binding the parameters (if any) used in that sql. seq is passed to help give fresh names
        to these parameters.
        """
        ...

    @override
    def do_rename_pure(self, old_path: str, new_path: str) -> _SqlRowProtocol[str]:
        orig_stat = yield from self.stat_pure(old_path)
        orig_is_dir = orig_stat.is_dir
        new_path_obj = pathlib.Path(new_path)
        if not orig_is_dir:
            new_parent_object = new_path_obj.parent
            yield from self._mkdirs(new_parent_object)
            yield from self._execute(f"""
UPDATE memories_fs SET entry_name = {self.pos_placeholder}, full_path = {self.pos_placeholder}, parent_path = {self.pos_placeholder} WHERE full_path = {self.pos_placeholder} AND namespace = {self.pos_placeholder}
                        """, (new_path_obj.name, new_path, str(new_parent_object), old_path, self.ns))
            return f"Renamed file {old_path} -> {new_path}"
        yield from self._mkdirs(new_path_obj.parent)
        params = {
            "orig_path": old_path,
            "new_name": new_path_obj.name,
            "new_parent": str(new_path_obj.parent),
            "ns": self.ns,
            "path_like": old_path + "%"
        }
        (replace_full_path, replace_full_param) = self.do_replace_first(
            "m.full_path",
            old_path,
            str(new_path_obj),
            0
        )
        (replace_parent_path, replace_parent_param) = self.do_replace_first(
            "m.parent_path",
            old_path,
            str(new_path_obj),
            1
        )
        params.update(replace_parent_param)
        params.update(replace_full_param)

        yield from self._execute(f"""
UPDATE memories_fs AS m SET
entry_name = CASE m.full_path
WHEN {self.named_placeholder("orig_path")} THEN {self.named_placeholder("new_name")}
ELSE m.entry_name
END,
full_path = {replace_full_path},
parent_path = CASE m.full_path
WHEN {self.named_placeholder("orig_path")} THEN {self.named_placeholder("new_parent")}
ELSE {replace_parent_path}
END WHERE namespace = {self.named_placeholder("ns")} AND full_path LIKE {self.named_placeholder("path_like")}
                    """, params)
        return f"Renamed directory {old_path} -> {new_path}"

    @override
    def list_dir_pure(self, path: str) -> _SqlMultiRowProtocol[list[tuple[str, bool]]]:
        res = yield from self._execute_multi(
            f"SELECT entry_name, is_directory FROM memories_fs WHERE namespace = {self.pos_placeholder} AND parent_path = {self.pos_placeholder}",
            (self.ns, path)
        )
        return [ (r[0], r[1]) for r in res ]

    @override
    def rm_pure(self, path: str) -> _SqlUpdateProtocol[str]:
        row_count = yield from self._execute_update(f"DELETE FROM memories_fs WHERE namespace = {self.pos_placeholder} AND full_path = {self.pos_placeholder}", (self.ns, path))
        return f"Deleted {row_count} entries"

    @override
    def read_file_pure(self, path: str) -> _SqlRowProtocol[str | None]:
        r = yield from self._execute(f"SELECT contents FROM memories_fs WHERE namespace = {self.pos_placeholder} AND full_path = {self.pos_placeholder} AND contents IS NOT NULL", (self.ns, path))
        if r is None:
            return None
        return r[0]

    @override
    def write_file_pure(self, path: str, content: str) -> _SqlRowProtocol[None]:
        target_path = pathlib.Path(path)
        yield from self._mkdirs(path=target_path.parent)
        yield from self._execute(f"""
INSERT INTO memories_fs(namespace, full_path, entry_name, parent_path, is_directory, contents) VALUES
({self.pos_placeholder}, {self.pos_placeholder}, {self.pos_placeholder}, {self.pos_placeholder}, False, {self.pos_placeholder})
ON CONFLICT(namespace, full_path) DO UPDATE SET contents = excluded.contents
                    """, (self.ns, path, target_path.name, str(target_path.parent), content))
        return None

    @override
    def stat_pure(self, path: str) -> _SqlRowProtocol[FileStat]:
        r = yield from self._execute(f"SELECT is_directory FROM memories_fs WHERE namespace = {self.pos_placeholder} AND full_path = {self.pos_placeholder}", (self.ns, path))
        if not r:
            return FileStat(exists=False, is_dir=False)
        return FileStat(exists=True, is_dir=r[0])

class PostgresSQLBackedPure(SQLBackendPure):
    @property
    def pos_placeholder(self) -> str:
        return "%s"

    @override
    def named_placeholder(self, nm: str) -> str:
        return f"%({nm})s"
    
    @override
    def do_replace_first(self, replace_attribute: str, to_replace: str, replace_with: str, seq: int) -> tuple[str, dict[str, str]]:
        replace_quot = re.escape(to_replace)
        src_patt_p = f"replace_src{seq}"
        replace_str_p = f"replace_dst{seq}"
        return (f"regexp_replace({replace_attribute}, %({src_patt_p})s, %({replace_str_p})s)", {
            src_patt_p: "^" + replace_quot,
            replace_str_p: replace_with
        })

class SqliteBackendPure(SQLBackendPure):
    @property
    def pos_placeholder(self) -> str:
        return '?'

    @override
    def named_placeholder(self, nm: str) -> str:
        return f":{nm}"

    @override
    def do_replace_first(self, replace_attribute: str, to_replace: str, replace_with: str, seq: int) -> tuple[str, dict[str, str]]:
        src_p = f"replace_src_{seq}"
        dst_p = f"replace_dst_{seq}"
        return (f"replace_first({replace_attribute}, :{src_p}, :{dst_p})", {
            src_p: to_replace,
            dst_p: replace_with
        })

class AsyncSQLBackend[CONN](AsyncMemoryBackend[_SqlQuery, int, _SqlRow | None, list[_SqlRow]]):
    def __init__(self, conn: CONN, sql_logic: SQLBackendPure):
        super().__init__(sql_logic)
        self.conn = conn
        self._sql_logic = sql_logic

    @abstractmethod
    def _cursor(self) -> AsyncContextManager[AsyncDBCursor]:
        ...

    async def init_from(self, init_from: str):
        is_empty = await self._run_row(
            self._sql_logic._is_empty()
        )
        if is_empty and init_from is not None:
            await self._run_row(
                self._sql_logic._init_from(init_from)
            )

    async def _run_loop[R, T](
        self, d: Generator[_SqlQuery, R, T], db: Callable[[AsyncDBCursor], Awaitable[R]]
    ) -> T:
        async with self._cursor() as curr:
            try:
                req = next(d)
                while True:
                    (query, params) = req
                    await curr.execute(query, params)
                    resp = await db(curr)
                    req = d.send(resp)
            except StopIteration as e:
                return e.value

    @override
    async def _run_multi[T](
        self,
        d: Generator[tuple[str, tuple[Any, ...] | dict[str, Any]], list[tuple[Any, ...]], T
    ]) -> T:
        async def extract(curr: AsyncDBCursor) -> list[tuple[Any, ...]]:
            rows = []
            async for r in curr:
                rows.append(r)
            return rows
        return await self._run_loop(d, extract)
    
    @override
    async def _run_row[Y](
        self,
        d: Generator[tuple[str, tuple[Any, ...] | dict[str, Any]], tuple[Any, ...] | None, Y]
    ) -> Y:
        async def getter(d: AsyncDBCursor) -> tuple[Any, ...] | None:
            if d.description is None:
                return None
            return await d.fetchone()
        return await self._run_loop(d, getter)
            
    @override
    async def _run_update[T](
        self,
        d: Generator[tuple[str, tuple[Any, ...] | dict[str, Any]], int, T]
    ) -> T:
        async def getter(c: AsyncDBCursor) -> int:
            return c.rowcount
        return await self._run_loop(d, getter)

class SyncSqlBackend[CONN](MemoryBackend[_SqlQuery, int, _SqlRow | None, list[_SqlRow]]):
    def __init__(self, conn: CONN, init_from: str | None, sql_logic: SQLBackendPure):
        super().__init__(sql_logic)
        self.conn = conn
        self._init(init_from, sql_logic)

    def _cursor(self) -> ContextManager[DBCursor]:
        ...

    @abstractmethod
    def _setup(self) -> None:
        ...

    def _init(self, init_from: str | None, sql_logic: SQLBackendPure) -> None:
        self._setup()
        if self._run_row(sql_logic._is_empty()) and init_from is not None:
            self._run_row(
                sql_logic._init_from(init_from)
            )
    
    def _run_loop[R, T](
        self, d: Generator[_SqlQuery, R, T], db: Callable[[DBCursor], R]
    ) -> T:
        with self._cursor() as curr:
            try:
                req = next(d)
                while True:
                    (query, params) = req
                    curr.execute(query, params)
                    resp = db(curr)
                    req = d.send(resp)
            except StopIteration as e:
                return e.value

    @override
    def _run_multi[T](
        self,
        d: Generator[tuple[str, tuple[Any, ...] | dict[str, Any]], list[tuple[Any, ...]], T
    ]) -> T:
        def extract(curr: DBCursor) -> list[tuple[Any, ...]]:
            rows = []
            for r in curr:
                rows.append(r)
            return rows
        return self._run_loop(d, extract)
    
    @override
    def _run_row[T](
        self,
        d: Generator[tuple[str, tuple[Any, ...] | dict[str, Any]], tuple[Any, ...] | None, T]
    ) -> T:
        def getter(d: DBCursor) -> tuple[Any, ...] | None:
            if d.description is None:
                return None
            return d.fetchone()
        return self._run_loop(d, getter)
            
    @override
    def _run_update[T](
        self,
        d: Generator[tuple[str, tuple[Any, ...] | dict[str, Any]], int, T]
    ) -> T:
        def getter(c: DBCursor) -> int:
            return c.rowcount
        return self._run_loop(d, getter)

from threading import Lock

class PostgresMemoryBackend(SyncSqlBackend[Connection | ConnectionPool]):
    def __init__(self, ns: str, conn: Connection | ConnectionPool, init_from: str | None = None):
        if isinstance(conn, Connection):
            self._lock = Lock()
        else:
            self._lock = None
        super().__init__(conn, init_from, PostgresSQLBackedPure(ns))

    def _adapt(self, q: DBCursorG[LiteralString]) -> DBCursor:
        """
        The only reason postgres' cursor isn't a DBCursor is because of their "clever"
        insistence queries are literalstrings. The attempt to prevent sql injections is admirable (I guess)
        but annoying for our purposes. This function ensure that the ONLY difference between the postgres cursor
        is that it expects a literal string (vs a normal string) and then cast them.

        This casting is itself safe because the runtime representation of literal strings is the same as regular strings,
        and we are extra careful to not have sql injections in our string interpolation.
        """
        return cast(DBCursor, q)

    @contextmanager
    def _connection_get(self) -> Iterator[Connection]:
        if isinstance(self.conn, ConnectionPool):
            # TX management handled for us by the pool
            with self.conn.connection() as conn:
                yield conn
        else:
            assert self._lock is not None
            with self._lock:
                try:
                    yield self.conn
                    self.conn.commit()
                except:
                    self.conn.rollback()
                    raise

    @contextmanager
    def _cursor(self) -> Iterator['DBCursor']:
        """
        Override to handle PostgreSQL connection properly.
        Don't use the connection as context manager to avoid closing it.
        Instead, manually manage transactions per cursor operation.
        """
        with self._connection_get() as conn:
            cur = conn.cursor()
            try:
                yield self._adapt(cur)
            finally:
                cur.close()

    @override
    def _setup(self):
        pass

    def __del__(self):
        try:
            if hasattr(self, 'conn') and self.conn and not self.conn.closed and isinstance(self.conn, Connection):
                # Commit any pending transaction before closing
                if not self.conn.autocommit:
                    try:
                        self.conn.commit()
                    except Exception:
                        pass
        except Exception:
            pass

import asyncio

class AsyncPostgresBackend(AsyncSQLBackend[AsyncConnectionPool | AsyncConnection]):
    def __init__(self, ns: str, conn: AsyncConnectionPool[AsyncConnection[TupleRow]] | AsyncConnection[TupleRow]):
        super().__init__(conn, PostgresSQLBackedPure(ns))
        self._lock = asyncio.Lock() if isinstance(conn, AsyncConnection) else None
    
    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncConnection]:
        if isinstance(self.conn, AsyncConnection):
            # we intentionally do NOT use the connection as a context manager here
            assert self._lock is not None
            async with self._lock:
                try:
                    yield self.conn
                    await self.conn.commit()
                except:
                    await self.conn.rollback()
                    raise
        else:
            # connection pool handles TX management for us
            async with self.conn.connection() as conn:
                yield conn

    def _adapt(self, curr: AsyncCursorG[LiteralString]) -> AsyncDBCursor:
        return cast(AsyncDBCursor, curr)

    @override
    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncDBCursor]:
        # TX management done by the _connection yielder
        async with self._connection() as conn:
            curr = conn.cursor()
            try:
                yield self._adapt(curr)
            finally:
                await curr.close()


class SqliteMemoryBackend(SyncSqlBackend[sqlite3.Connection]):
    def __init__(self, ns: str, conn: sqlite3.Connection, init_from: str | None = None):
        self.conn_lock = Lock()
        super().__init__(conn, init_from, SqliteBackendPure(ns))


    @contextmanager
    def _cursor(self) -> Iterator[DBCursor]:
        """
        postgres' cursor is a context manager, sqlite's isn't.
        Both use the connection as a context manager for transaction control.
        Paper over this difference, so that when `with self._cursor()` exits
        the transaction is committed (or rolled back) and the cursor is closed.
        """
        with self.conn_lock:
            with self.conn:
                cur = self.conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()

    @override
    def _setup(self):
        with self.conn:
            cur = self.conn.cursor()
            cur.executescript("""
CREATE TABLE IF NOT EXISTS memories_fs(
    namespace TEXT NOT NULL,
    entry_name TEXT NOT NULL,
    full_path TEXT COLLATE NOCASE, -- required so LIKE uses the index
    parent_path TEXT,
    is_directory BOOL NOT NULL,
    contents TEXT,
    FOREIGN KEY(parent_path, namespace) REFERENCES memories_fs(full_path, namespace) ON DELETE CASCADE, -- good hierarchy
    UNIQUE (namespace, full_path), -- unique paths within ns
    UNIQUE (namespace, parent_path, entry_name), -- unique names within directories
    CHECK (parent_path is NOT NULL OR (full_path = '/memories' AND is_directory AND entry_name = 'memories')),
    CHECK (parent_path is NULL OR (full_path = concat(parent_path, '/', entry_name))), -- entry, path consistency
    CHECK (contents IS NOT NULL != is_directory)
);

CREATE INDEX IF NOT EXISTS memories_namespace_path ON memories_fs(namespace, full_path);
                              """)
        self.conn.create_function("replace_first", 3, self._replace_first, deterministic=True)
        self.conn.execute("PRAGMA foreign_keys = ON;")  # easy trip up!

    def _replace_first(self, target, src_str, result) -> str:
        return str(target).replace(str(src_str), str(result), 1)
    
type _NoYieldProtocol[T] = Generator[Never, Never, T]

GenParam = ParamSpec("GenParam")

RET_T = TypeVar("RET_T")

def to_generator(cb: Callable[GenParam, RET_T]) -> Callable[GenParam, _NoYieldProtocol[RET_T]]:
    def wrapper(*args: GenParam.args, **kwargs: GenParam.kwargs) -> _NoYieldProtocol[RET_T]:
        to_wrap = cb(*args, **kwargs)
        if False:
            yield
        return to_wrap
    return wrapper


class PureFilesystemLogic(PureMemoryBackend[Never, Never, Never, Never]):
    def __init__(self, storage_folder: pathlib.Path):
        self.memory_root = storage_folder

    def _relativize(self, path: str) -> pathlib.Path:
        r = pathlib.Path(path).relative_to("/memories")
        return (self.memory_root / r).resolve()

    @to_generator
    @override
    def stat_pure(self, path: str) -> FileStat:
        actual = self._relativize(path)
        if not actual.exists():
            return FileStat(exists=False, is_dir=False)
        return FileStat(exists=True, is_dir=actual.is_dir())

    @to_generator
    @override
    def write_file_pure(self, path: str, content: str):
        r = self._relativize(path)
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_text(content)

    @to_generator
    @override
    def read_file_pure(self, path: str) -> str | None:
        r = self._relativize(path)
        if not r.is_file():
            return None
        return r.read_text()

    @to_generator
    @override
    def do_rename_pure(self, old_path: str, new_path: str) -> str:
        old_full_path = self._relativize(old_path)
        new_full_path = self._relativize(new_path)

        new_full_path.parent.mkdir(parents=True, exist_ok=True)
        old_full_path.rename(new_full_path)

        return f"Renamed {old_path} -> {new_path}"

    @to_generator
    @override
    def list_dir_pure(self, path: str) -> list[tuple[str, bool]]:
        r = self._relativize(path)
        to_ret : list[tuple[str, bool]]= []
        for it in r.iterdir():
            to_ret.append(
                (str(it.relative_to(r)), it.is_dir())
            )
        return to_ret
    
    @to_generator
    @override
    def rm_pure(self, path: str) -> str:
        r = self._relativize(path)
        if r.is_dir():
            shutil.rmtree(r)
        else:
            r.unlink()
        return f"Removed {path}"


class FileSystemBackendCommon():
    def __init__(self, mem_root: pathlib.Path, init_from: pathlib.Path | None = None):
        super().__init__(PureFilesystemLogic(mem_root)) #type: ignore
        if self._is_empty(mem_root) and init_from is not None:
            self._init_from(mem_root, init_from)
        
    

    def _is_empty(self, mem_root: pathlib.Path) -> bool:
        try:
            next(mem_root.rglob("*"))
            return False
        except StopIteration:
            return True

    def _init_from(self, memory_root: pathlib.Path, other_dir: pathlib.Path) -> None:
        for r in other_dir.glob("*"):
            if r.is_dir():
                shutil.copytree(src=r, dst=pathlib.Path(memory_root) / r.name)
            else:
                shutil.copy(r, memory_root)

    def _run_all[T](self, d: Generator[Never, Never, T]) -> T:
        try:
            next(d)
            assert False
        except StopIteration as e:
            return e.value


class FileSystemMemoryBackend(FileSystemBackendCommon, MemoryBackend[Never, Never, Never, Never]):
    """
    A simple backend that "mounts" `/memories` to the storage folder given as the constructor arg.
    """
    def __init__(self, storage_folder: pathlib.Path, init_from: pathlib.Path | None = None):
        super().__init__(storage_folder, init_from)

    @override
    def _run_multi[T](self, d: Generator[Never, Never, T]) -> T:
        return self._run_all(d)

    @override
    def _run_row[T](self, d: Generator[Never, Never, T]) -> T:
        return self._run_all(d)
    
    @override
    def _run_update[T](self, d: Generator[Never, Never, T]) -> T:
        return self._run_all(d)
    
class AsyncFileMemoryBackend(FileSystemBackendCommon, AsyncMemoryBackend[Never, Never, Never, Never]):
    def __init__(self, storage_folder: pathlib.Path, init_from: pathlib.Path | None = None):
        super().__init__(storage_folder, init_from)

    @override
    async def _run_multi[T](self, d: Generator[Never, Never, T]) -> T:
        return self._run_all(d)
    
    @override
    async def _run_row[T](self, d: Generator[Never, Never, T]) -> T:
        return self._run_all(d)
    
    @override
    async def _run_update[T](self, d: Generator[Never, Never, T]) -> T:
        return self._run_all(d)
    
class InvalidPathError(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg

def _validate_path(s: str):
    p = pathlib.PurePosixPath(s)
    if not p.is_absolute():
        raise InvalidPathError(f"Memory path: {s} is not an absolute path")
    if not p.is_relative_to("/memories"):
        raise InvalidPathError(f"Memory path: {s} is not rooted at /memories")

def _memory_tool_impl[R](
    backend: MemoryToolImpl[R],
    args: UnifiedMemorySchema,
    missing_required: Callable[[str], R]
) -> R:
    match args.command:
        case "create":
            if args.path is None:
                return missing_required("path")
            elif args.file_text is None:
                return missing_required("file_text")
            _validate_path(args.path)
            return backend.create(args.path, args.file_text)

        case "delete":
            if args.path is None:
                return missing_required("path")
            return backend.delete(args.path)

        case "insert":
            if args.path is None:
                return missing_required("path")
            elif args.insert_line is None:
                return missing_required("insert_line")
            elif args.insert_text is None:
                return missing_required("insert_text")
            _validate_path(args.path)
            return backend.insert(args.path, args.insert_line, args.insert_text)
        case "rename":
            if args.old_path is None:
                return missing_required("old_path")
            elif args.new_path is None:
                return missing_required("new_path")
            _validate_path(args.old_path)
            _validate_path(args.new_path)
            return backend.rename(args.old_path, args.new_path)

        case "str_replace":
            if args.path is None:
                return missing_required("path")
            elif args.old_str is None:
                return missing_required("old_str")
            elif args.new_str is None:
                return missing_required("new_str")
            _validate_path(args.path)
            return backend.str_replace(args.path, args.old_str, args.new_str)

        case "view":
            if args.path is None:
                return missing_required("path")
            _validate_path(args.path)
            range : tuple[int, int] | None = None
            if args.view_range is not None and len(args.view_range) >= 2:
                range = (args.view_range[0], args.view_range[1])
            return backend.view(args.path, range)

def async_memory_tool(backend: MemoryToolImpl[Awaitable[str]]) -> BaseTool:
    async def missing_required(s: str):
        return f"Error: missing required {s} argument"
    
    class MemoryTool(WithAsyncImplementation[str], UnifiedMemorySchema):
        """
        Here to make the tool maker happy
        """
        @override
        async def run(self) -> str:
            try:
                return await _memory_tool_impl(
                    backend, self, missing_required
                )
            except InvalidPathError as e:
                return f"Error: {e.msg}"
    return MemoryTool.as_tool("memory")

def memory_tool(backend: MemoryToolImpl[str]) -> BaseTool:
    """
    Generates a memory tool using the given backend.
    """
    def missing_required(s: str):
        return f"Error: missing required {s} argument"

    class MemoryTool(WithImplementation[str], UnifiedMemorySchema):
        """
        Here to make the tool annotation happy
        """
        @override
        def run(self) -> str:
            try:
                return _memory_tool_impl(
                    backend, self, missing_required
                )
            except InvalidPathError as e:
                return f"Error: {e.msg}"
    return MemoryTool.as_tool("memory")


# ---------------------------------------------------------------------------
# OpenAI-targeted memory tool: discriminated-union schema.
#
# OpenAI (and OpenAI-compat backends) doesn't have a trained-on memory tool
# schema the way Anthropic does, so we can pick the shape. A sum type with a
# single ``memory_op`` field is much friendlier to the model — each variant
# only carries the fields its command actually needs — and stays
# strict-mode-compatible because the top level is still ``type: "object"``.
# ---------------------------------------------------------------------------


class _CreateOp(BaseModel):
    """Create a file at ``path`` with the given contents."""
    op: Literal["create"]
    path: str = Field(description="Absolute memory path (must start with /memories).")
    file_text: str = Field(description="Full contents of the new file.")


class _ViewOp(BaseModel):
    """Read a file or list a directory under ``path``."""
    op: Literal["view"]
    path: str = Field(description="Absolute memory path (must start with /memories).")
    view_range: list[int] | None = Field(
        default=None,
        description=(
            "Optional [start, end] line range (1-indexed, inclusive). "
            "Use -1 for end to read to EOF. Ignored for directories."
        ),
    )


class _StrReplaceOp(BaseModel):
    """Replace a unique substring in the file at ``path``."""
    op: Literal["str_replace"]
    path: str = Field(description="Absolute memory path (must start with /memories).")
    old_str: str = Field(description="Exact substring to replace. Must occur exactly once.")
    new_str: str = Field(description="Replacement text.")


class _InsertOp(BaseModel):
    """Insert a new line at ``insert_line`` in the file at ``path``."""
    op: Literal["insert"]
    path: str = Field(description="Absolute memory path (must start with /memories).")
    insert_line: int = Field(description="0-based line index at which to insert.")
    insert_text: str = Field(description="Text to insert (a trailing newline is added).")


class _DeleteOp(BaseModel):
    """Delete the file or directory at ``path``."""
    op: Literal["delete"]
    path: str = Field(description="Absolute memory path (must start with /memories).")


class _RenameOp(BaseModel):
    """Rename or move a file or directory."""
    op: Literal["rename"]
    old_path: str = Field(description="Existing absolute memory path.")
    new_path: str = Field(description="Destination absolute memory path.")


type _MemoryOpUnion = Annotated[
    _CreateOp | _ViewOp | _StrReplaceOp | _InsertOp | _DeleteOp | _RenameOp,
    Field(discriminator="op"),
]


class _OpenAIMemorySchema(BaseModel):
    """OpenAI-targeted memory tool schema.

    A single ``memory_op`` field carrying a discriminated union over
    the six memory commands. Top level is still ``type: "object"`` so
    OpenAI strict-mode validation works."""
    memory_op: _MemoryOpUnion = Field(
        description=(
            "The memory operation to perform. The 'op' tag selects the variant; "
            "each variant carries only the fields relevant to that operation."
        ),
    )

type ErrHandler[R] = Callable[[str], R]

def _dispatch_openai_op[R](
    backend: MemoryToolImpl[R],
    op: _MemoryOpUnion,
    err_handler: ErrHandler
) -> R:
    """Route a parsed sum-type variant into the backend's typed methods."""
    match op:
        case _CreateOp(path=p, file_text=ft):
            return backend.create(p, ft)
        case _ViewOp(path=p, view_range=vr):
            rng: tuple[int, int] | None = None
            if vr is not None and len(vr) == 2:
                rng = (vr[0], vr[1])
            elif vr is not None:
                return err_handler(f"Error: view range invalid, expected an array of exactly 2 elements, received {len(vr)}")
            return backend.view(p, rng)
        case _StrReplaceOp(path=p, old_str=os_, new_str=ns):
            return backend.str_replace(p, os_, ns)
        case _InsertOp(path=p, insert_line=il, insert_text=it):
            return backend.insert(p, il, it)
        case _DeleteOp(path=p):
            return backend.delete(p)
        case _RenameOp(old_path=op_, new_path=np):
            return backend.rename(op_, np)


_OPENAI_MEMORY_TOOL_DESCRIPTION = """\
Persistent filesystem-style memory that survives across turns and
across conversations within this workflow. All paths live under
``/memories`` and are sandboxed there. Use this tool to record
intermediate observations, decisions, partial results, and any
context you want to recall later — anything not written here is
forgotten when the conversation ends.

The ``memory_op`` field selects the operation. Each variant carries
only the fields that operation needs:

- ``view``: read a file (optionally a line range) or list a directory.
- ``create``: write a brand-new file.
- ``str_replace``: replace a unique substring in an existing file.
- ``insert``: insert a line at a specific index in an existing file.
- ``delete``: remove a file or directory.
- ``rename``: move/rename a file or directory.

Prefer ``view`` before ``create`` to avoid clobbering, and prefer
``str_replace`` over rewriting a whole file when only part changes.
"""


def openai_async_memory_tool(
    backend: MemoryToolImpl[Awaitable[str]],
) -> BaseTool:
    async def err_handler(s: str) -> str:
        return s

    """Async OpenAI-flavored memory tool. Same backend contract as
    :func:`async_memory_tool`; differs only in the tool-args schema
    seen by the model."""

    class OpenAIMemoryTool(WithAsyncImplementation[str], _OpenAIMemorySchema):
        __doc__ = _OPENAI_MEMORY_TOOL_DESCRIPTION

        @override
        async def run(self) -> str:
            return await _dispatch_openai_op(backend, self.memory_op, err_handler)
    return OpenAIMemoryTool.as_tool("memory")


def openai_memory_tool(backend: MemoryToolImpl[str]) -> BaseTool:
    """Sync OpenAI-flavored memory tool. Companion to
    :func:`memory_tool`."""

    class OpenAIMemoryTool(WithImplementation[str], _OpenAIMemorySchema):
        __doc__ = _OPENAI_MEMORY_TOOL_DESCRIPTION

        @override
        def run(self) -> str:
            return _dispatch_openai_op(backend, self.memory_op, lambda s: s)
    return OpenAIMemoryTool.as_tool("memory")
