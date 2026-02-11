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
from typing import NotRequired, TypeVar, Any, Annotated, Type, Callable, Sequence, ContextManager, Protocol, Iterator, Generic
import re
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
    """

    search_string: str = Field(description="The query string to search for provided as a python regex. Thus, you must escape any special characters (like [, |, etc.)")
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
    state: VFSState
) -> Iterator[str]:
    with provider() as dir:
        files = state["vfs"]
        target = pathlib.Path(dir)
        for (k, v) in files.items():
            tgt = target / k
            tgt.parent.mkdir(exist_ok=True, parents=True)
            tgt.write_text(v)

        if fs_layer is not None:
            mounted_path = pathlib.Path(fs_layer)
            for p in mounted_path.rglob("*"):
                if not p.is_file():
                    continue
                rel_path = p.relative_to(mounted_path)
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

    put_doc_extra: NotRequired[str]
    get_doc_extra: NotRequired[str]


class _VFSAccess(Generic[InputType]):
    def __init__(self, conf: VFSToolConfig):
        self.conf = conf

    def materialize(self, state: InputType, debug: bool = False) -> ContextManager[str]:
        manager : TempDirectoryProvider = tempfile.TemporaryDirectory
        if debug:
            manager = debugging_tmp_directory
        return _materialize(manager, self.conf.get("fs_layer"), state)

    def get(self, state: InputType, file: str) -> bytes | None:
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
            yield (p, v.encode("utf-8"))

        if (fs_layer := self.conf.get("fs_layer", None)) is not None:
            root = pathlib.Path(fs_layer)
            for child in root.rglob("*"):
                if not child.is_file():
                    continue
                rel_path = child.relative_to(root)
                if str(rel_path) in d:
                    continue
                yield (str(rel_path), child.read_bytes())


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

    class PutFileSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]

        files: dict[str, str] = \
            Field(description="A dictionary associating RELATIVE pathnames to the contents to store at those path names. Do NOT include a leading `./` it is always implied. "
                  "The provided contents for the file are durably stored into the virtual filesystem. "
                  "Any files contents with the same path named stored in previous tool calls are overwritten.")

    pf_doc = "Put file contents onto the virtual file system used by this workflow."
    if "put_doc_extra" in conf:
        pf_doc += f"\n\n{conf['put_doc_extra']}"

    PutFileSchema.__doc__ = pf_doc

    put_filter = _make_checker(conf.get("forbidden_write"))
    get_filter = _make_checker(conf.get("forbidden_read"))

    @tool(args_schema=PutFileSchema)
    def put_file(
        files: dict[str, str],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str | Command:
        for (k, _) in files.items():
            if not put_filter(k):
                return f"Illegal put operation: cannot write {k} on the VFS"
        return tool_output(
            tool_call_id=tool_call_id,
            res={
                "vfs": files
            }
        )
    
    def _get_content_raw(s: InputType, path: str) -> str | None:
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
        if not get_filter(path):
            return None
        return _get_content_raw(s, path)

    @inject(doc_extra=conf.get('get_doc_extra'))
    @_copy_base_doc
    class GetFileSchema(_GetFileSchemaBase):
        pass

    @tool(args_schema=GetFileSchema)
    def get_file(
        path: str,
        state: Annotated[InputType, InjectedState],
        range: FileRange | None = None
    ) -> str:
        cont = _get_content(state, path)
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
            if not get_filter(k):
                continue
            yield k
        for f_name in list_underlying():
            if not get_filter(f_name) or f_name in state["vfs"]:
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

    materializer = _VFSAccess[InputType](conf=conf)

    return (tools, materializer)


def fs_tools(fs_layer: str, forbidden_read: str | None = None) -> list[BaseTool]:
    """
    Create stateless file system tools that operate directly on a directory.

    Unlike vfs_tools, these tools don't use langgraph state - they simply
    read from the provided filesystem path. Useful for immutable file access
    where no VFS overlay is needed.

    Args:
        fs_layer: Path to the directory to expose
        forbidden_read: Optional regex pattern for paths that cannot be read

    Returns:
        List of tools: [get_file, list_files, grep_files]
    """
    base_path = pathlib.Path(fs_layer)
    check_allowed = _make_checker(forbidden_read)

    @cache
    def list_all_files() -> Sequence[str]:
        return [str(f.relative_to(base_path)) for f in base_path.rglob("*") if f.is_file()]

    @_copy_base_doc
    class GetFileSchema(_GetFileSchemaBase):
        pass

    @tool(args_schema=GetFileSchema)
    def get_file(path: str, range: FileRange | None = None) -> str:
        if not check_allowed(path):
            return "File not found"
        child = base_path / path
        if child.is_file():
            try:
                return _get_file(child.read_text(), range)
            except Exception:
                return "File not found"
        return "File not found"

    @_copy_base_doc
    class ListFileSchema(_ListFileSchemaBase):
        pass

    @tool(args_schema=ListFileSchema)
    def list_files() -> str:
        return "\n".join(f for f in list_all_files() if check_allowed(f))

    @_copy_base_doc
    class GrepFileSchema(_GrepFileSchemaBase):
        pass

    @tool(args_schema=GrepFileSchema)
    def grep_files(
        search_string: str,
        matching_lines: bool,
        match_in: list[str] | None = None
    ) -> str:
        def file_contents() -> Iterator[tuple[str, str]]:
            for f in base_path.rglob("*"):
                if not f.is_file():
                    continue
                rel_name = str(f.relative_to(base_path))
                if not check_allowed(rel_name):
                    continue
                try:
                    yield (rel_name, f.read_text())
                except Exception:
                    continue
        def read_file(p: str) -> str | None:
            try:
                return (base_path / p).read_text()
            except:
                return None
        return _grep_impl(search_string, matching_lines, file_contents(), read_file, match_in)

    return [get_file, list_files, grep_files]
