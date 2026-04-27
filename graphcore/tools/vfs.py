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

from typing_extensions import TypedDict
from typing import NotRequired, TypeVar, Any, Annotated, Type, Callable, Iterable, Sequence, ContextManager, Protocol, Iterator, Generic
import asyncio
import re
import shutil
from dataclasses import dataclass, field
from functools import cache
import pathlib
import contextlib
import tempfile

from pydantic import BaseModel, Field, create_model

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.tools.base import BaseTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from ..graph import FlowInput

from ..graph import tool_output


def _copy_base_doc[T](cls: T) -> T:
    """Decorator to copy __doc__ from the first base class."""
    for base in cls.__bases__:  # type: ignore
        if base.__doc__:
            cls.__doc__ = base.__doc__
            break
    return cls


# returns true if the file is okay to access
def _make_checker(patt: str | None) -> Callable[[str], bool]:
    if patt is None:
        return lambda f_name: True
    match = re.compile(patt)
    return lambda f_name: match.fullmatch(f_name) is None


# Always-on exclude floor. ``.git`` is never useful for any consumer
# (VFS tools, materialization, audit) — including it is pure waste at
# best and actively misleading at worst (every workflow would re-snapshot
# megabytes of history). User-supplied ``global_exclude`` patterns and
# predicates union on top of this floor.


def _floor_include(path: str) -> bool:
    """``.git`` directory and contents are always excluded. True if the
    path passes the floor (i.e. is NOT under .git anywhere in its path
    parts; covers root .git, nested .git from submodules, etc.)."""
    return ".git" not in pathlib.PurePosixPath(path).parts


# Type alias for the user-facing ``global_exclude`` config. Both forms
# return True to *exclude* the path. The callable form (preferred) is
# more natural when the predicate cares about Path semantics like
# ``suffix`` or ``parts``; the regex form is retained for symmetry with
# ``forbidden_read``/``forbidden_write`` but is clunky for path logic
# (you end up writing ``(.*/)?node_modules(/.*)?`` for what's just
# ``"node_modules" in p.parts``).
type GlobalExcludeArg = str | Callable[[pathlib.PurePath], bool] | None


def _make_global_include_pred(arg: GlobalExcludeArg) -> Callable[[str], bool]:
    """Returns an *include* predicate (True = okay to access) that
    composes the always-on ``.git`` floor with the user-supplied
    exclusion (if any).

    Polarity matches ``_make_checker``: True means the path is
    accessible. The user-facing argument is named ``global_exclude``
    because the user thinks in terms of "what to exclude"; the
    polarity flip happens here at the boundary so internal predicate
    composition stays consistent.
    """
    if arg is None:
        return _floor_include
    if isinstance(arg, str):
        rx = re.compile(arg)
        # fullmatch for symmetry with forbidden_read/forbidden_write.
        return lambda p: _floor_include(p) and rx.fullmatch(p) is None
    # Callable form: caller's predicate returns True to exclude. We flip.
    user_excludes = arg
    return lambda p: _floor_include(p) and not user_excludes(pathlib.PurePosixPath(p))

class FileRange(BaseModel):
    start_line: int = Field(description="The line to start reading from; lines are numbered starting from 1.")
    end_line: int = Field(description="The line to read until EXCLUSIVE.")

def _get_file(cont: str | None, range: FileRange | None) -> str:
    if cont is None:
        return "File not found"
    if not range:
        return cont
    start = range.start_line - 1
    to_ret = cont.splitlines()[start:range.end_line - 1]
    return "\n".join(to_ret)



def _grep_impl(
    search_string: str,
    matching_lines: bool,
    file_contents: Iterator[tuple[str, str]],
    get_content: Callable[[str], str | None],
    match_in: list[str] | None
) -> str:
    """
    Generic grep implementation over file contents.

    Args:
        search_string: Regex pattern to search for
        file_contents: Iterator of (filename, content) tuples
        check_allowed: Filter function for allowed filenames

    Returns:
        Newline-separated list of matching filenames, or error message
    """
    comp: re.Pattern
    try:
        comp = re.compile(search_string, re.MULTILINE)
    except Exception:
        return "Illegal pattern name, check your syntax and try again."
    
    matches: list[str] = []

    match_set = None if not match_in else set(match_in)

    should_search = \
        (lambda _: True) if match_set is None else \
        (lambda f: f in match_set)

    for (k, v) in file_contents:
        if not should_search(k):
            continue
        if comp.search(v) is not None:
            matches.append(k)
    
    if not matching_lines:
        return "\n".join(matches)

    matched_lines = []

    for match_name in matches:
        cont_s = get_content(match_name)
        assert cont_s is not None
        cont = cont_s.splitlines()
        for (lno, l) in enumerate(cont, start=1):
            if comp.search(l):
                matched_lines.append(f"{match_name}:{lno}:{l}")

    return "\n".join(matched_lines)



class _GetFileSchemaBase(BaseModel):
    """
    Read the contents of the VFS at some relative path.

    If the path doesn't exist, this function returns "File not found".
    """
    path: str = Field(description="The relative path of the file on the VFS. IMPORTANT: Do NOT include a leading `./` it is implied")
    range: FileRange | None = Field(description="If set, (start, end) indicates to return lines starting from line `start` (lines are 1 indexed) until `end` (exclusive). If unset, the entire file is returned.", default=None)


class _ListFileSchemaBase(BaseModel):
    """
    Lists all file contents of the VFS, including in any subdirectories. Directory entries are *not* included.
    Each file in the VFS has its own line in the output, any empty lines should be ignored.
    """
    pass


class _PutFileSchemaBase(BaseModel):
    """
    Put file contents onto the virtual file system used by this workflow.
    """
    files: dict[str, str] = \
        Field(description="A dictionary associating RELATIVE pathnames to the contents to store at those path names. Do NOT include a leading `./` it is always implied. "
              "The provided contents for the file are durably stored into the virtual filesystem. "
              "Any files contents with the same path named stored in previous tool calls are overwritten.")


class _GrepFileSchemaBase(BaseModel):
    """
    Search for a specific string in the files on the VFS. The output depends on the
    value of the `matching_lines` argument. If false, returns a list of
    file names which contain the query somewhere in their contents, with one file name per line.
    If true, returns a list of matching lines in files with the format:
    ```
    $filename:$lineno:$line
    ```
    $line is a line matching the search string in $filename at $lineno (starting at 1).

    In both output modes, empty lines should be ignored.

    In both modes, the paths to search can be restricted with `match_in`.

    IMPORTANT: This tool searches file *contents*, NOT their names. DO NOT use this tool to search for files
    whose name make a regex, it will not work.
    """

    search_string: str = Field(description="The query string to search for provided as is as an input to a python regex.")
    matching_lines: bool = Field(description="If true, show the matching lines and the line number; if false, simply list the matching files")
    match_in: list[str] | None = Field(description="If set, narrow the search to only the paths listed here.", default=None)

def merge_vfs(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    new_left = left.copy()
    for (f_name, cont) in right.items():
        new_left[f_name] = cont
    return new_left


class VFSState(TypedDict):
    vfs: Annotated[dict[str, str], merge_vfs]

class VFSInput(FlowInput):
    vfs: dict[str, str]

InputType = TypeVar("InputType", bound=VFSState)

StateVar = TypeVar("StateVar", contravariant=True)

class TempDirectoryProvider(Protocol):
    def __call__(self) -> ContextManager[str]:
        ...

class VFSAccessor(Protocol[StateVar]):
    def materialize(self, state: StateVar, debug: bool = False) -> ContextManager[str]:
        ...

    def iterate(self, state: StateVar) -> Iterator[tuple[str, bytes]]:
        ...

    def get(self, state: StateVar, file: str) -> bytes | None:
        ...


@contextlib.contextmanager
def debugging_tmp_directory() -> Iterator[str]:
    temp_dir = tempfile.mkdtemp()
    print(f"DEBUG: Working directory: {temp_dir}")
    yield temp_dir

@contextlib.contextmanager
def _materialize(
    provider: TempDirectoryProvider,
    fs_layer: str | None,
    state: VFSState,
    include_path: Callable[[str], bool],
) -> Iterator[str]:
    with provider() as dir:
        files = state["vfs"]
        target = pathlib.Path(dir)
        for (k, v) in files.items():
            if not include_path(k):
                continue
            tgt = target / k
            tgt.parent.mkdir(exist_ok=True, parents=True)
            tgt.write_text(v)

        if fs_layer is not None:
            mounted_path = pathlib.Path(fs_layer)
            for p in mounted_path.rglob("*"):
                if not p.is_file():
                    continue
                rel_path = p.relative_to(mounted_path)
                if not include_path(str(rel_path)):
                    continue
                copy_path = target / rel_path
                if copy_path.exists():
                    continue
                copy_path.parent.mkdir(exist_ok=True, parents=True)
                copy_path.write_bytes(p.read_bytes())
        yield dir


class VFSToolConfig(TypedDict):
    immutable: bool
    fs_layer: NotRequired[str | None]
    forbidden_read: NotRequired[str]
    forbidden_write: NotRequired[str]

    # Paths invisible to every consumer (tools, materialization, audit).
    # Either a fullmatch regex (BC; clunky for path logic) or a
    # ``Callable[[PurePath], bool]`` returning True to exclude
    # (preferred). ``.git`` is always excluded; this composes on top.
    # See ``fs_tools_layered`` for the contract.
    global_exclude: NotRequired[GlobalExcludeArg]

    put_doc_extra: NotRequired[str]
    get_doc_extra: NotRequired[str]

class NormalizationError(RuntimeError):
    pass

def _normalize_and_validate(
    s: str
) -> str:
    res = pathlib.PurePath(s)
    if ".." in res.parts:
        raise NormalizationError(f"Invalid path: {s} contains `..`")
    if res.is_absolute():
        raise NormalizationError(f"Invalid path: {s} is absolute")
    return str(res)

from typing_extensions import ParamSpec
import functools

PS = ParamSpec("PS")
TOOL_RET = TypeVar("TOOL_RET", str, str | Command)

def handle_path_errors(f: Callable[PS, TOOL_RET]) -> Callable[PS, TOOL_RET]:
    @functools.wraps(f)
    def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> TOOL_RET:
        try:
            return f(*args, **kwargs)
        except NormalizationError as e:
            return f"Tool call failed: {str(e)}" # type: ignore
    return wrapper
        

class _VFSAccess(Generic[InputType]):
    def __init__(
        self,
        conf: VFSToolConfig,
        global_include: Callable[[str], bool],
    ):
        self.conf = conf
        # Single global-include predicate (True = okay to access),
        # consistent in polarity with ``_make_checker``. Composed by
        # the parent factory and shared with the tool closures so we
        # don't rebuild the regex per call AND there's no chance of
        # one site forgetting to consult it.
        self._global_include = global_include

    def materialize(self, state: InputType, debug: bool = False) -> ContextManager[str]:
        manager : TempDirectoryProvider = tempfile.TemporaryDirectory
        if debug:
            manager = debugging_tmp_directory
        return _materialize(
            manager, self.conf.get("fs_layer"), state,
            include_path=self._global_include,
        )

    def get(self, state: InputType, file: str) -> bytes | None:
        if not self._global_include(file):
            return None
        if file in state["vfs"]:
            return state["vfs"][file].encode("utf-8")
        fs_layer = self.conf.get("fs_layer", None)
        if fs_layer is None:
            return None
        path = pathlib.Path(fs_layer) / file
        if path.is_file():
            return path.read_bytes()
        else:
            return None

    def iterate(self, state: InputType) -> Iterator[tuple[str, bytes]]:
        d = state["vfs"]
        for (p, v) in d.items():
            if not self._global_include(p):
                continue
            yield (p, v.encode("utf-8"))

        if (fs_layer := self.conf.get("fs_layer", None)) is not None:
            root = pathlib.Path(fs_layer)
            for child in root.rglob("*"):
                if not child.is_file():
                    continue
                rel_path = child.relative_to(root)
                rel_str = str(rel_path)
                if not self._global_include(rel_str):
                    continue
                if rel_str in d:
                    continue
                yield (rel_str, child.read_bytes())


def vfs_tools(conf: VFSToolConfig, ty: Type[InputType]) -> tuple[list[BaseTool], VFSAccessor[InputType]]:
    def inject(doc_extra: str | None = None) -> Callable[[type[BaseModel]], type[BaseModel]]:
        def to_ret(s: type[BaseModel]) -> type[BaseModel]:
            f: dict[str, Any] = {k: (v.annotation, v) for (k, v) in s.model_fields.items()}
            f["state"] = Annotated[ty, InjectedState]
            d_string = s.__doc__
            if doc_extra is not None:
                d_string = f"{d_string}\n\n{doc_extra}"
            return create_model(
                s.__name__,
                __doc__ = d_string,
                **f
            )
        return to_ret

    pf_doc = _PutFileSchemaBase.__doc__ or ""
    if "put_doc_extra" in conf:
        pf_doc += f"\n\n{conf['put_doc_extra']}"

    class PutFileSchema(_PutFileSchemaBase):
        __doc__ = pf_doc
        tool_call_id: Annotated[str, InjectedToolCallId]

    # Single global-include predicate (True = okay to access) compiled
    # once and folded into the read/write filters below. Sharing avoids
    # rebuilding the regex per call AND removes the "remember to also
    # check the global exclude" foot-gun — every site only consults
    # ``can_read`` / ``can_write``.
    global_include = _make_global_include_pred(conf.get("global_exclude"))
    forbidden_write_chk = _make_checker(conf.get("forbidden_write"))
    forbidden_read_chk = _make_checker(conf.get("forbidden_read"))
    can_write: Callable[[str], bool] = (
        lambda p: forbidden_write_chk(p) and global_include(p)
    )
    can_read: Callable[[str], bool] = (
        lambda p: forbidden_read_chk(p) and global_include(p)
    )

    @tool(args_schema=PutFileSchema)
    @handle_path_errors
    def put_file(
        files: dict[str, str],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str | Command:
        to_update = {}
        for (k, upd) in files.items():
            norm_path = _normalize_and_validate(k)
            if not can_write(norm_path):
                return f"Illegal put operation: cannot write {k} on the VFS"
            to_update[norm_path] = upd
        return tool_output(
            tool_call_id=tool_call_id,
            res={
                "vfs": to_update
            }
        )

    def _get_content_raw(s: InputType, path: str) -> str | None:
        # Note: ``can_read`` is enforced upstream in ``_get_content`` and
        # ``_list_files``; this helper is the raw underlay reader.
        # ``global_include`` is still checked here because callers
        # (e.g. ``grep_files`` via ``_get_content_raw``) bypass
        # ``can_read``'s ``forbidden_read`` filter but must NEVER see
        # globally-excluded paths.
        if not global_include(path):
            return None
        vfs = s["vfs"]
        if path not in vfs:
            layer = conf.get("fs_layer", None)
            if layer is None:
                return None
            child = pathlib.Path(layer) / path
            if not child.is_file():
                return None
            try:
                return child.read_text()
            except:
                return None
        else:
            return vfs[path]

    def _get_content(s: InputType, path: str) -> str | None:
        if not can_read(path):
            return None
        return _get_content_raw(s, path)

    @inject(doc_extra=conf.get('get_doc_extra'))
    @_copy_base_doc
    class GetFileSchema(_GetFileSchemaBase):
        pass

    @tool(args_schema=GetFileSchema)
    @handle_path_errors
    def get_file(
        path: str,
        state: Annotated[InputType, InjectedState],
        range: FileRange | None = None
    ) -> str:
        norm_path = _normalize_and_validate(path)
        cont = _get_content(state, norm_path)
        return _get_file(cont, range)

    @cache
    def list_underlying() -> Sequence[str]:
        layer = conf.get("fs_layer", None)
        if layer is None:
            return []
        base = pathlib.Path(layer)
        return [str(f.relative_to(base)) for f in base.rglob("*") if f.is_file()]

    @inject()
    @_copy_base_doc
    class ListFileSchema(_ListFileSchemaBase):
        pass

    def _list_files(
        state: InputType
    ) -> Iterator[str]:
        for (k, _) in state["vfs"].items():
            if not can_read(k):
                continue
            yield k
        for f_name in list_underlying():
            if not can_read(f_name) or f_name in state["vfs"]:
                continue
            yield f_name

    @tool(args_schema=ListFileSchema)
    def list_files(
        state: Annotated[InputType, InjectedState]
    ) -> str:
        to_ret = list(_list_files(state))
        return "\n".join(to_ret)

    @inject()
    @_copy_base_doc
    class GrepFileSchema(_GrepFileSchemaBase):
        pass

    @tool(args_schema=GrepFileSchema)
    def grep_files(
        state: Annotated[InputType, InjectedState],
        search_string: str,
        matching_lines: bool,
        match_in: list[str] | None = None
    ) -> str:
        def file_contents() -> Iterator[tuple[str, str]]:
            for path in _list_files(state):
                cont = _get_content_raw(state, path)
                if not cont:
                    continue
                yield (path, cont)
            
        return _grep_impl(search_string, matching_lines, file_contents(), lambda p: _get_content_raw(state, p), match_in)

    tools: list[BaseTool] = [get_file, list_files, grep_files]
    if not conf["immutable"]:
        tools.append(put_file)

    materializer = _VFSAccess[InputType](conf=conf, global_include=global_include)

    return (tools, materializer)


class FSBackend(Protocol):
    """A read-only filesystem-like source.

    ``get`` returns the text contents at ``path`` or ``None`` if the path is
    not present in this backend. ``list`` enumerates every path this backend
    can serve. ``dump_to`` writes every file the backend serves into
    ``target`` so downstream tools (e.g. solc) can read them from disk.

    ``dump_to`` accepts an optional ``include_path`` predicate; when set,
    only paths for which ``include_path(path)`` returns ``True`` are
    written to ``target``. ``None`` (the default) writes every file.
    Backends should honor the predicate efficiently when possible (e.g.
    ``DirBackend`` passes it through to ``shutil.copytree``'s ``ignore``
    argument so excluded directory subtrees are short-circuited at the
    root rather than recursed into and filtered file-by-file).

    The predicate is the implementation primitive used by callers like
    ``_LayeredMaterializer`` to enforce ``global_exclude`` patterns
    uniformly across every backend in a stack — but the filter itself
    is general and not tied to that one use case.
    """

    def get(self, path: str) -> str | None:
        ...

    def list(self) -> Iterable[str]:
        ...

    async def dump_to(
        self,
        target: pathlib.Path,
        include_path: Callable[[str], bool] | None = None,
    ) -> None:
        ...


class Materializer(Protocol):
    """Writes a composite filesystem view into a real on-disk directory.

    Typically the caller creates a tmpdir and passes it as ``target``. The
    materializer is responsible for honoring layering order: higher-priority
    backends overwrite lower-priority ones so the disk layout matches what
    ``fs_tools_layered`` reads would see.

    ``get`` is a single-file query against the same composite view used by
    ``dump_to``; returns the file's content (first-hit across layers in
    priority order) or ``None`` if no layer serves the path. This is the
    unfiltered view — ``forbidden_read`` only affects the tool surface, not
    materialization or existence queries.
    """

    async def dump_to(self, target: pathlib.Path) -> None:
        ...

    def get(self, path: str) -> str | None:
        ...


@dataclass
class DirBackend:
    """``FSBackend`` adapter over a real on-disk directory."""

    root: pathlib.Path
    cache_listing: bool = True
    _cached: list[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = pathlib.Path(self.root)

    def _enumerate(self) -> list[str]:
        return [
            str(f.relative_to(self.root))
            for f in self.root.rglob("*")
            if f.is_file()
        ]

    def get(self, path: str) -> str | None:
        child = self.root / path
        if not child.is_file():
            return None
        try:
            return child.read_text()
        except Exception:
            return None

    def list(self) -> Iterable[str]:
        if not self.cache_listing:
            return self._enumerate()
        if self._cached is None:
            self._cached = self._enumerate()
        return self._cached

    async def dump_to(
        self,
        target: pathlib.Path,
        include_path: Callable[[str], bool] | None = None,
    ) -> None:
        root = self.root
        if include_path is None:
            ignore = None
        else:
            # Adapt the predicate to ``shutil.copytree``'s ``ignore``
            # callback. The callback is invoked once per directory; we
            # relativize ``src`` back to ``self.root`` so the predicate
            # sees the same project-relative paths it does in ``list``
            # and ``get``. Returning a name in the ignore list short-
            # circuits the entire subtree at that point.
            def _ignore(src: str, names: list[str]) -> list[str]:
                src_rel = pathlib.Path(src).relative_to(root)
                prefix = "" if str(src_rel) == "." else f"{src_rel}/"
                return [n for n in names if not include_path(f"{prefix}{n}")]
            ignore = _ignore

        def _copy() -> None:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(root, target, dirs_exist_ok=True, ignore=ignore)
        await asyncio.to_thread(_copy)


class _LayeredMaterializer:
    """Materializer that dumps backends in reverse priority order.

    Lowest-priority (last in the list) dumps first, so higher-priority
    (earlier in the list) backends' writes overwrite on collision. This
    preserves the first-hit read semantics of ``fs_tools_layered`` at
    materialization time.

    ``global_exclude`` (regex) defines paths that are invisible to every
    consumer — VFS tools, materialization, and (transitively) any
    downstream the materializer feeds (audit DB, prover, typecheck).
    Distinct from ``forbidden_read``, which only filters the agent's
    tool surface and lets materialization through. ``.git`` is hardcoded
    into the exclude floor; the user pattern unions on top.
    """

    def __init__(
        self,
        backends: Sequence[FSBackend],
        global_exclude: GlobalExcludeArg = None,
    ) -> None:
        self._backends = list(backends)
        # Single include predicate (True = okay to access). Polarity
        # matches ``_make_checker`` and ``FSBackend.dump_to``'s
        # ``include_path`` parameter, so backends and tool closures
        # consume the same shape.
        self._include: Callable[[str], bool] = _make_global_include_pred(global_exclude)

    async def dump_to(self, target: pathlib.Path) -> None:
        for backend in reversed(self._backends):
            await backend.dump_to(target, include_path=self._include)

    def get(self, path: str) -> str | None:
        if not self._include(path):
            return None
        for backend in self._backends:
            content = backend.get(path)
            if content is not None:
                return content
        return None


def fs_tools_layered(
    backends: Sequence[FSBackend],
    forbidden_read: str | None = None,
    global_exclude: GlobalExcludeArg = None,
) -> tuple[list[BaseTool], Materializer]:
    """Create stateless read-only filesystem tools over a layered backend stack.

    ``backends`` are consulted in priority order (first wins) for ``get``,
    and unioned (with de-duplication by path) for ``list``.

    ``forbidden_read`` (fullmatch regex) filters the agent-facing tool
    surface only — ``get_file``/``list_files``/``grep_files``. The
    materializer still dumps every non-globally-excluded file so
    downstream consumers (e.g. solc) see a complete tree.

    ``global_exclude`` filters every consumer — the tool surface, the
    materializer, and (transitively) anyone the materializer feeds
    (audit DB, prover). Accepts either a fullmatch regex (BC; clunky
    for path logic) or a ``Callable[[PurePath], bool]`` returning True
    to exclude (preferred). The ``.git`` directory is always excluded;
    the user pattern/predicate composes on top.

    Returns ``(tools, materializer)``. ``tools`` is the usual
    ``[get_file, list_files, grep_files]``; ``materializer`` dumps the
    composite view into a caller-provided directory, honoring layer
    priority and the global exclude.
    """
    # Single fold: tool-surface readability is "forbidden_read allows
    # AND not globally excluded". One predicate = no foot-gun about
    # remembering to also check the global exclude at every call site.
    forbidden_chk = _make_checker(forbidden_read)
    global_include = _make_global_include_pred(global_exclude)
    can_read: Callable[[str], bool] = lambda p: forbidden_chk(p) and global_include(p)
    backend_list = list(backends)

    def _lookup_unfiltered(path: str) -> str | None:
        # ``forbidden_read`` is tool-only; the materializer's existence
        # check and similar paths use this unfiltered hook. We still
        # short-circuit on ``global_include`` because globally excluded
        # paths must be invisible to *every* consumer including this one.
        if not global_include(path):
            return None
        for backend in backend_list:
            content = backend.get(path)
            if content is not None:
                return content
        return None

    def _lookup(path: str) -> str | None:
        if not can_read(path):
            return None
        for backend in backend_list:
            content = backend.get(path)
            if content is not None:
                return content
        return None

    def _enumerate() -> Iterator[str]:
        seen: set[str] = set()
        for backend in backend_list:
            for path in backend.list():
                if path in seen:
                    continue
                seen.add(path)
                if not can_read(path):
                    continue
                yield path

    @_copy_base_doc
    class GetFileSchema(_GetFileSchemaBase):
        pass

    @tool(args_schema=GetFileSchema)
    @handle_path_errors
    def get_file(path: str, range: FileRange | None = None) -> str:
        norm_path = _normalize_and_validate(path)
        return _get_file(_lookup(norm_path), range)

    @_copy_base_doc
    class ListFileSchema(_ListFileSchemaBase):
        pass

    @tool(args_schema=ListFileSchema)
    def list_files() -> str:
        return "\n".join(_enumerate())

    @_copy_base_doc
    class GrepFileSchema(_GrepFileSchemaBase):
        pass

    @tool(args_schema=GrepFileSchema)
    def grep_files(
        search_string: str,
        matching_lines: bool,
        match_in: list[str] | None = None,
    ) -> str:
        def file_contents() -> Iterator[tuple[str, str]]:
            for path in _enumerate():
                content = _lookup_unfiltered(path)
                if content is None:
                    continue
                yield (path, content)
        return _grep_impl(
            search_string,
            matching_lines,
            file_contents(),
            _lookup_unfiltered,
            match_in,
        )

    tools: list[BaseTool] = [get_file, list_files, grep_files]
    return tools, _LayeredMaterializer(backend_list, global_exclude=global_exclude)


def fs_tools(
    fs_layer: str,
    forbidden_read: str | None = None,
    *,
    cache_listing: bool = True,
    global_exclude: GlobalExcludeArg = None,
) -> list[BaseTool]:
    """
    Create stateless file system tools that operate directly on a directory.

    Unlike vfs_tools, these tools don't use langgraph state - they simply
    read from the provided filesystem path. Useful for immutable file access
    where no VFS overlay is needed.

    Args:
        fs_layer: Path to the directory to expose
        forbidden_read: Optional regex pattern for paths that cannot be read
        cache_listing: If True (default), cache the directory listing after first call.
            Set to False if the agent needs to react to filesystem changes.
        global_exclude: Optional fullmatch regex for paths invisible to *every*
            consumer (tools, materialization, audit). ``.git`` is always
            excluded; this pattern unions on top. See
            ``fs_tools_layered`` for the full contract.

    Returns:
        List of tools: [get_file, list_files, grep_files]
    """
    backend = DirBackend(pathlib.Path(fs_layer), cache_listing=cache_listing)
    tools, _ = fs_tools_layered(
        [backend],
        forbidden_read=forbidden_read,
        global_exclude=global_exclude,
    )
    return tools
