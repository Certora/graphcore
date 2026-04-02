"""
Tests for VFS tool infrastructure: put/get/list/grep, forbidden paths,
path traversal prevention, and VFS merge reducer.
"""
import pytest

from langgraph.graph import MessagesState

from graphcore.tools.vfs import vfs_tools, VFSState, VFSToolConfig
from graphcore.testing import Scenario, tool_call_raw, ToolCallDict

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class VFSTestState(MessagesState, VFSState):
    pass


# ---------------------------------------------------------------------------
# Tool call constructors
# ---------------------------------------------------------------------------

_PUT = "put_file"
_GET = "get_file"
_LIST = "list_files"
_GREP = "grep_files"


def _put(**files: str) -> ToolCallDict:
    return tool_call_raw(_PUT, files=files)


def _put_files(files: dict[str, str]) -> ToolCallDict:
    return tool_call_raw(_PUT, files=files)


def _get(path: str) -> ToolCallDict:
    return tool_call_raw(_GET, path=path)


def _list() -> ToolCallDict:
    return tool_call_raw(_LIST)


def _grep(search_string: str, matching_lines: bool = False) -> ToolCallDict:
    return tool_call_raw(_GREP, search_string=search_string, matching_lines=matching_lines)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


def _make_tools(
    immutable: bool = False,
    forbidden_read: str | None = None,
    forbidden_write: str | None = None,
):
    conf: VFSToolConfig = {
        "immutable": immutable
    }
    if forbidden_read is not None:
        conf["forbidden_read"] = forbidden_read
    if forbidden_write is not None:
        conf["forbidden_write"] = forbidden_write
    tools, _ = vfs_tools(conf, VFSTestState)
    return tools


def _scenario(
    files: dict[str, str] | None = None,
    *,
    immutable: bool = False,
    forbidden_read: str | None = None,
    forbidden_write: str | None = None,
):
    tools = _make_tools(immutable=immutable, forbidden_read=forbidden_read, forbidden_write=forbidden_write)
    return Scenario(VFSTestState, *tools).init(vfs=files or {})


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _vfs(st: VFSTestState) -> dict[str, str]:
    return st["vfs"]



# =========================================================================
# Basic operations
# =========================================================================


class TestPutFile:
    async def test_put_single_file(self):
        vfs = await _scenario().turn(
            _put(**{"src/Foo.sol": "contract Foo {}"}),
        ).map_run(_vfs)
        assert vfs["src/Foo.sol"] == "contract Foo {}"

    async def test_put_multiple_files(self):
        vfs = await _scenario().turn(
            _put_files({"src/Foo.sol": "contract Foo {}", "src/Bar.sol": "contract Bar {}"}),
        ).map_run(_vfs)
        assert len(vfs) == 2

    async def test_put_overwrites_existing(self):
        vfs = await _scenario({"src/Foo.sol": "contract Foo {}"}).turn(
            _put(**{"src/Foo.sol": "contract Foo { uint x; }"}),
        ).map_run(_vfs)
        assert vfs["src/Foo.sol"] == "contract Foo { uint x; }"

    async def test_put_preserves_other_files(self):
        vfs = await _scenario({"src/Foo.sol": "contract Foo {}"}).turn(
            _put(**{"src/Bar.sol": "contract Bar {}"}),
        ).map_run(_vfs)
        assert vfs["src/Foo.sol"] == "contract Foo {}"
        assert vfs["src/Bar.sol"] == "contract Bar {}"

    async def test_immutable_vfs_has_no_put(self):
        tools = _make_tools(immutable=True)
        tool_names = {t.name for t in tools}
        assert "put_file" not in tool_names


class TestGetFile:
    async def test_get_existing_file(self):
        resp = await _scenario({"src/Foo.sol": "contract Foo {}"}).turn(
            _get("src/Foo.sol"),
        ).run_last_single_tool(_GET)
        assert "contract Foo {}" in resp

    async def test_get_nonexistent_file(self):
        resp = await _scenario().turn(
            _get("missing.sol"),
        ).run_last_single_tool(_GET)
        assert "not found" in resp.lower()


class TestListFiles:
    async def test_list_files(self):
        resp = await _scenario({
            "src/Foo.sol": "contract Foo {}",
            "src/Bar.sol": "contract Bar {}",
        }).turn(
            _list(),
        ).run_last_single_tool(_LIST)
        assert "src/Foo.sol" in resp
        assert "src/Bar.sol" in resp


class TestGrepFiles:
    async def test_grep_matching_files(self):
        resp = await _scenario({
            "src/Foo.sol": "contract Foo { uint x; }",
            "src/Bar.sol": "contract Bar { bool y; }",
        }).turn(
            _grep("uint"),
        ).run_last_single_tool(_GREP)
        assert "src/Foo.sol" in resp
        assert "src/Bar.sol" not in resp

    async def test_grep_matching_lines(self):
        resp = await _scenario({
            "src/Foo.sol": "line1\nuint x;\nline3",
        }).turn(
            _grep("uint", matching_lines=True),
        ).run_last_single_tool(_GREP)
        assert "src/Foo.sol:2:" in resp


# =========================================================================
# Forbidden write enforcement
# =========================================================================


class TestForbiddenWrite:
    async def test_forbidden_write_blocks(self):
        vfs = await _scenario(forbidden_write=r"^certora/.*").turn(
            _put(**{"certora/spec.spec": "rule foo {}"}),
        ).map_run(_vfs)
        assert "certora/spec.spec" not in vfs

    async def test_allowed_write_passes(self):
        vfs = await _scenario(forbidden_write=r"^certora/.*").turn(
            _put(**{"src/Foo.sol": "contract Foo {}"}),
        ).map_run(_vfs)
        assert vfs["src/Foo.sol"] == "contract Foo {}"

    async def test_dot_slash_normalized_and_blocked(self):
        """./certora/spec.spec normalizes to certora/spec.spec → blocked."""
        vfs = await _scenario(forbidden_write=r"^certora/.*").turn(
            _put_files({"./certora/spec.spec": "rule foo {}"}),
        ).map_run(_vfs)
        assert len(vfs) == 0

    async def test_dotdot_rejected(self):
        """x/../certora/spec.spec is rejected outright (.. in path)."""
        resp = await _scenario(forbidden_write=r"^certora/.*").turn(
            _put_files({"x/../certora/spec.spec": "rule foo {}"}),
        ).run_last_single_tool(_PUT)
        assert ".." in resp

    async def test_absolute_path_rejected(self):
        """/etc/passwd is rejected outright (absolute path)."""
        resp = await _scenario().turn(
            _put_files({"/etc/passwd": "root:x:0:0"}),
        ).run_last_single_tool(_PUT)
        assert "absolute" in resp.lower()

    async def test_mixed_batch_one_forbidden(self):
        """If any file in a batch is forbidden, the entire batch is rejected."""
        vfs = await _scenario(forbidden_write=r"^certora/.*").turn(
            _put_files({
                "src/Foo.sol": "contract Foo {}",
                "certora/spec.spec": "rule foo {}",
            }),
        ).map_run(_vfs)
        assert "src/Foo.sol" not in vfs
        assert "certora/spec.spec" not in vfs

    async def test_normalized_key_stored(self):
        """./src/Foo.sol normalizes to src/Foo.sol in the VFS."""
        vfs = await _scenario().turn(
            _put_files({"./src/Foo.sol": "contract Foo {}"}),
        ).map_run(_vfs)
        assert "src/Foo.sol" in vfs
        assert "./src/Foo.sol" not in vfs


# =========================================================================
# Forbidden read enforcement
# =========================================================================


class TestForbiddenRead:
    async def test_forbidden_read_blocks_get(self):
        resp = await _scenario(
            {"secrets/key.txt": "supersecret"},
            forbidden_read=r"^secrets/.*",
        ).turn(
            _get("secrets/key.txt"),
        ).run_last_single_tool(_GET)
        assert "supersecret" not in resp
        assert "not found" in resp.lower()

    async def test_forbidden_read_blocks_list(self):
        resp = await _scenario(
            {"src/Foo.sol": "contract Foo {}", "secrets/key.txt": "supersecret"},
            forbidden_read=r"^secrets/.*",
        ).turn(
            _list(),
        ).run_last_single_tool(_LIST)
        assert "src/Foo.sol" in resp
        assert "secrets/key.txt" not in resp

    async def test_forbidden_read_blocks_grep(self):
        resp = await _scenario(
            {"src/Foo.sol": "contract Foo {}", "secrets/key.txt": "supersecret"},
            forbidden_read=r"^secrets/.*",
        ).turn(
            _grep("supersecret"),
        ).run_last_single_tool(_GREP)
        assert "secrets" not in resp

    async def test_dot_slash_read_normalized_and_blocked(self):
        """./secrets/key.txt normalizes to secrets/key.txt → blocked."""
        resp = await _scenario(
            {"secrets/key.txt": "supersecret"},
            forbidden_read=r"^secrets/.*",
        ).turn(
            _get("./secrets/key.txt"),
        ).run_last_single_tool(_GET)
        assert "supersecret" not in resp

    async def test_dotdot_read_rejected(self):
        """x/../secrets/key.txt is rejected outright (.. in path)."""
        resp = await _scenario(
            {"secrets/key.txt": "supersecret"},
            forbidden_read=r"^secrets/.*",
        ).turn(
            _get("x/../secrets/key.txt"),
        ).run_last_single_tool(_GET)
        assert "supersecret" not in resp
        assert ".." in resp


# =========================================================================
# VFS merge reducer
# =========================================================================


class TestVFSMerge:
    async def test_sequential_puts_merge(self):
        vfs = await _scenario().turn(
            _put(**{"src/Foo.sol": "contract Foo {}"}),
        ).turn(
            _put(**{"src/Bar.sol": "contract Bar {}"}),
        ).map_run(_vfs)
        assert vfs["src/Foo.sol"] == "contract Foo {}"
        assert vfs["src/Bar.sol"] == "contract Bar {}"

    async def test_sequential_put_overwrites(self):
        vfs = await _scenario().turn(
            _put(**{"src/Foo.sol": "v1"}),
        ).turn(
            _put(**{"src/Foo.sol": "v2"}),
        ).map_run(_vfs)
        assert vfs["src/Foo.sol"] == "v2"
