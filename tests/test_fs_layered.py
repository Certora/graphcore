"""Tests for the layered FS surface in ``graphcore.tools.vfs``.

Covers:

* ``DirBackend`` — reads / list / caching / ``dump_to`` behaviour.
* ``_LayeredMaterializer`` — priority-ordered dump so higher-priority layers
  overwrite lower ones on collision.
* ``fs_tools_layered`` — first-hit read semantics, union-deduplicated listing,
  ``forbidden_read`` filtering, and grep behaviour across layers.
"""

import pathlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pytest

from graphcore.tools.vfs import (
    DirBackend,
    _LayeredMaterializer,
    _make_global_include_pred,
    fs_tools_layered,
)

# Async / sync test classes opt in to ``pytest.mark.asyncio`` individually
# rather than via a module-level mark — synchronous predicate tests in
# ``TestGlobalExcludePredicate`` don't want the asyncio handling and were
# producing "marked asyncio but not async" warnings under the module-wide
# mark.


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class InMemoryBackend:
    """Minimal in-memory ``FSBackend`` implementation for layering tests.

    Declared test-local rather than imported, since the runtime counterpart
    in ``composer/spec/natspec/pipeline.py`` has extra responsibilities we
    don't want to drag into the graphcore test suite.
    """

    files: dict[str, str] = field(default_factory=dict)

    def get(self, path: str) -> str | None:
        return self.files.get(path)

    def list(self) -> Iterable[str]:
        return list(self.files)

    async def dump_to(
        self,
        target: pathlib.Path,
        include_path: Callable[[str], bool] | None = None,
    ) -> None:
        for rel, content in self.files.items():
            if include_path is not None and not include_path(rel):
                continue
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)


def _tools_by_name(backends, **kw):
    tools, mat = fs_tools_layered(backends, **kw)
    return {t.name: t for t in tools}, mat


def _write_tree(root: pathlib.Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


# ---------------------------------------------------------------------------
# DirBackend
# ---------------------------------------------------------------------------


class TestDirBackend:
    pytestmark = pytest.mark.asyncio

    async def test_get_returns_content(self, tmp_path: pathlib.Path):
        _write_tree(tmp_path, {"a/b.sol": "hello"})
        b = DirBackend(tmp_path)
        assert b.get("a/b.sol") == "hello"

    async def test_get_missing_returns_none(self, tmp_path: pathlib.Path):
        b = DirBackend(tmp_path)
        assert b.get("missing.sol") is None

    async def test_get_on_directory_returns_none(self, tmp_path: pathlib.Path):
        (tmp_path / "subdir").mkdir()
        b = DirBackend(tmp_path)
        assert b.get("subdir") is None

    async def test_list_enumerates_nested_files(self, tmp_path: pathlib.Path):
        _write_tree(tmp_path, {
            "a/b.sol": "x",
            "c.sol": "y",
            "a/d/e.sol": "z",
        })
        b = DirBackend(tmp_path)
        assert set(b.list()) == {"a/b.sol", "c.sol", "a/d/e.sol"}

    async def test_list_caches_by_default(self, tmp_path: pathlib.Path):
        _write_tree(tmp_path, {"a.sol": "v1"})
        b = DirBackend(tmp_path)
        list(b.list())  # prime the cache
        (tmp_path / "b.sol").write_text("added after cache")
        # Cache hit: b.sol should NOT appear.
        assert set(b.list()) == {"a.sol"}

    async def test_list_live_when_cache_disabled(self, tmp_path: pathlib.Path):
        _write_tree(tmp_path, {"a.sol": "v1"})
        b = DirBackend(tmp_path, cache_listing=False)
        list(b.list())
        (tmp_path / "b.sol").write_text("added")
        assert set(b.list()) == {"a.sol", "b.sol"}

    async def test_dump_to_copies_tree(self, tmp_path: pathlib.Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_tree(src, {"a/b.sol": "x", "c.sol": "y"})
        b = DirBackend(src)
        await b.dump_to(dst)
        assert (dst / "a/b.sol").read_text() == "x"
        assert (dst / "c.sol").read_text() == "y"

    async def test_dump_to_merges_into_existing_dir(self, tmp_path: pathlib.Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.sol").write_text("keep")
        _write_tree(src, {"new.sol": "new"})
        b = DirBackend(src)
        await b.dump_to(dst)
        assert (dst / "existing.sol").read_text() == "keep"
        assert (dst / "new.sol").read_text() == "new"


# ---------------------------------------------------------------------------
# _LayeredMaterializer
# ---------------------------------------------------------------------------


class TestLayeredMaterializer:
    pytestmark = pytest.mark.asyncio

    async def test_empty_backends_is_noop(self, tmp_path: pathlib.Path):
        mat = _LayeredMaterializer([])
        await mat.dump_to(tmp_path)
        assert list(tmp_path.iterdir()) == []

    async def test_single_backend_dumps(self, tmp_path: pathlib.Path):
        b = InMemoryBackend({"x.sol": "content"})
        mat = _LayeredMaterializer([b])
        await mat.dump_to(tmp_path)
        assert (tmp_path / "x.sol").read_text() == "content"

    async def test_collision_highest_priority_wins(self, tmp_path: pathlib.Path):
        """First element in the backend list is the highest-priority layer.

        The materializer dumps in reverse order, so the highest-priority
        backend writes last and survives on disk. This preserves the
        first-hit read semantics of ``fs_tools_layered``.
        """
        high = InMemoryBackend({"shared.sol": "from_high"})
        low = InMemoryBackend({"shared.sol": "from_low"})
        mat = _LayeredMaterializer([high, low])
        await mat.dump_to(tmp_path)
        assert (tmp_path / "shared.sol").read_text() == "from_high"

    async def test_three_layer_priority(self, tmp_path: pathlib.Path):
        high = InMemoryBackend({"shared.sol": "H"})
        mid = InMemoryBackend({"shared.sol": "M"})
        low = InMemoryBackend({"shared.sol": "L"})
        mat = _LayeredMaterializer([high, mid, low])
        await mat.dump_to(tmp_path)
        assert (tmp_path / "shared.sol").read_text() == "H"

    async def test_unique_paths_union(self, tmp_path: pathlib.Path):
        b1 = InMemoryBackend({"a.sol": "a"})
        b2 = InMemoryBackend({"b.sol": "b"})
        mat = _LayeredMaterializer([b1, b2])
        await mat.dump_to(tmp_path)
        assert (tmp_path / "a.sol").read_text() == "a"
        assert (tmp_path / "b.sol").read_text() == "b"


# ---------------------------------------------------------------------------
# fs_tools_layered — read operations across layers
# ---------------------------------------------------------------------------


class TestLayeredReads:
    pytestmark = pytest.mark.asyncio

    async def test_get_first_hit_wins(self):
        high = InMemoryBackend({"shared.sol": "from_high"})
        low = InMemoryBackend({"shared.sol": "from_low"})
        tools, _ = _tools_by_name([high, low])
        assert tools["get_file"].invoke({"path": "shared.sol"}) == "from_high"

    async def test_get_falls_through_to_lower_backend(self):
        high = InMemoryBackend({"a.sol": "a"})
        low = InMemoryBackend({"b.sol": "b"})
        tools, _ = _tools_by_name([high, low])
        assert tools["get_file"].invoke({"path": "b.sol"}) == "b"

    async def test_get_missing_returns_not_found(self):
        tools, _ = _tools_by_name([InMemoryBackend({})])
        resp = tools["get_file"].invoke({"path": "nope.sol"})
        assert "not found" in resp.lower()

    async def test_list_unions_and_dedupes(self):
        b1 = InMemoryBackend({"a.sol": "a", "shared.sol": "v1"})
        b2 = InMemoryBackend({"b.sol": "b", "shared.sol": "v2"})
        tools, _ = _tools_by_name([b1, b2])
        listing = set(tools["list_files"].invoke({}).splitlines())
        assert listing == {"a.sol", "b.sol", "shared.sol"}

    async def test_list_empty_when_no_backends(self):
        tools, _ = _tools_by_name([])
        assert tools["list_files"].invoke({}) == ""

    async def test_grep_finds_across_layers(self):
        b1 = InMemoryBackend({"a.sol": "contract Alpha {}"})
        b2 = InMemoryBackend({"b.sol": "contract Beta {}"})
        tools, _ = _tools_by_name([b1, b2])
        resp = tools["grep_files"].invoke({
            "search_string": "contract",
            "matching_lines": False,
        })
        assert "a.sol" in resp
        assert "b.sol" in resp

    async def test_grep_uses_first_hit_content(self):
        """When a path appears in multiple layers, grep should search the
        content the ``get_file`` tool would return — i.e. the highest
        layer's copy."""
        high = InMemoryBackend({"shared.sol": "MATCH"})
        low = InMemoryBackend({"shared.sol": "no_hit"})
        tools, _ = _tools_by_name([high, low])
        resp = tools["grep_files"].invoke({
            "search_string": "MATCH",
            "matching_lines": False,
        })
        assert "shared.sol" in resp

    async def test_grep_match_in_filters(self):
        b = InMemoryBackend({
            "a.sol": "contract A {}",
            "b.sol": "contract B {}",
        })
        tools, _ = _tools_by_name([b])
        resp = tools["grep_files"].invoke({
            "search_string": "contract",
            "matching_lines": False,
            "match_in": ["a.sol"],
        })
        assert "a.sol" in resp
        assert "b.sol" not in resp

    async def test_grep_matching_lines_includes_line_number(self):
        b = InMemoryBackend({"x.sol": "line1\ncontract X {}\nline3"})
        tools, _ = _tools_by_name([b])
        resp = tools["grep_files"].invoke({
            "search_string": "contract",
            "matching_lines": True,
        })
        assert "x.sol:2:" in resp


# ---------------------------------------------------------------------------
# fs_tools_layered — forbidden_read filtering
# ---------------------------------------------------------------------------


class TestForbiddenRead:
    pytestmark = pytest.mark.asyncio

    async def test_forbidden_read_blocks_get(self):
        b = InMemoryBackend({"secrets/key.txt": "supersecret"})
        tools, _ = _tools_by_name([b], forbidden_read=r"^secrets/.*")
        resp = tools["get_file"].invoke({"path": "secrets/key.txt"})
        assert "supersecret" not in resp
        assert "not found" in resp.lower()

    async def test_forbidden_read_filters_list(self):
        b = InMemoryBackend({
            "a.sol": "ok",
            "secrets/key.txt": "supersecret",
        })
        tools, _ = _tools_by_name([b], forbidden_read=r"^secrets/.*")
        listing = tools["list_files"].invoke({}).splitlines()
        assert "a.sol" in listing
        assert "secrets/key.txt" not in listing

    async def test_forbidden_read_filters_grep(self):
        b = InMemoryBackend({
            "a.sol": "hello",
            "secrets/key.txt": "hello supersecret",
        })
        tools, _ = _tools_by_name([b], forbidden_read=r"^secrets/.*")
        resp = tools["grep_files"].invoke({
            "search_string": "hello",
            "matching_lines": False,
        })
        assert "a.sol" in resp
        assert "secrets" not in resp

    async def test_forbidden_read_does_not_affect_materializer(self, tmp_path: pathlib.Path):
        """Materialization dumps EVERY file the backend serves; the
        forbidden_read filter only governs the tool-visible view. This is
        intentional so downstream consumers (e.g. solc) see a complete
        source tree."""
        b = InMemoryBackend({
            "a.sol": "visible",
            "secrets/key.txt": "supersecret",
        })
        _, mat = fs_tools_layered([b], forbidden_read=r"^secrets/.*")
        await mat.dump_to(tmp_path)
        assert (tmp_path / "a.sol").read_text() == "visible"
        assert (tmp_path / "secrets/key.txt").read_text() == "supersecret"


# ---------------------------------------------------------------------------
# End-to-end: tool reads match materialized disk state
# ---------------------------------------------------------------------------


class TestMaterializerRoundtrip:
    pytestmark = pytest.mark.asyncio

    async def test_tool_view_matches_dumped_tree(self, tmp_path: pathlib.Path):
        high = InMemoryBackend({"shared.sol": "from_high"})
        low = InMemoryBackend({
            "shared.sol": "from_low",
            "only_low.sol": "low_only",
        })
        tools, mat = _tools_by_name([high, low])

        # Tool surface: high wins on collision, unique paths visible.
        assert tools["get_file"].invoke({"path": "shared.sol"}) == "from_high"
        assert tools["get_file"].invoke({"path": "only_low.sol"}) == "low_only"

        # Materialized disk state mirrors the tool surface.
        out = tmp_path / "out"
        await mat.dump_to(out)
        assert (out / "shared.sol").read_text() == "from_high"
        assert (out / "only_low.sol").read_text() == "low_only"

    async def test_dirbackend_participates_in_layering(self, tmp_path: pathlib.Path):
        """Ensure the real on-disk backend layers correctly alongside the
        in-memory one."""
        on_disk_root = tmp_path / "fs"
        _write_tree(on_disk_root, {"shared.sol": "from_disk", "disk_only.sol": "D"})
        disk = DirBackend(on_disk_root)
        mem = InMemoryBackend({"shared.sol": "from_mem", "mem_only.sol": "M"})

        # Mem first — should win on collision.
        tools, mat = _tools_by_name([mem, disk])
        assert tools["get_file"].invoke({"path": "shared.sol"}) == "from_mem"
        assert tools["get_file"].invoke({"path": "disk_only.sol"}) == "D"
        assert tools["get_file"].invoke({"path": "mem_only.sol"}) == "M"

        out = tmp_path / "out"
        await mat.dump_to(out)
        assert (out / "shared.sol").read_text() == "from_mem"
        assert (out / "disk_only.sol").read_text() == "D"
        assert (out / "mem_only.sol").read_text() == "M"


# ---------------------------------------------------------------------------
# global_exclude: invisible-to-every-consumer pattern
# ---------------------------------------------------------------------------


class TestGlobalExcludePredicate:
    """Direct tests of ``_make_global_include_pred`` — the predicate
    factory all the higher layers compose. Polarity: True = okay to
    access (consistent with ``_make_checker``)."""

    # The module-level ``pytestmark = pytest.mark.asyncio`` (line 24)
    # applies asyncio handling to every test by default; these are
    # synchronous predicate tests, so opt out at the class level to
    # silence the "marked asyncio but not async" warning.
    pytestmark: list = []

    def test_default_floor_excludes_dot_git(self):
        pred = _make_global_include_pred(None)
        # .git directory and contents always excluded.
        assert not pred(".git")
        assert not pred(".git/HEAD")
        assert not pred(".git/refs/heads/main")
        assert not pred("submodule/.git")
        assert not pred("submodule/.git/config")
        # Things that should NOT match the .git filter:
        assert pred("src/Foo.sol")
        assert pred(".gitignore")  # not a .git path-segment
        assert pred("code.git.txt")  # name happens to contain .git
        assert pred("notgit/file")

    def test_regex_form_unions_with_floor(self):
        # Exclude PDFs in addition to the .git floor.
        pred = _make_global_include_pred(r".*\.pdf")
        assert not pred(".git/HEAD")  # floor still active
        assert not pred("docs/spec.pdf")
        assert not pred("spec.pdf")
        assert pred("src/Foo.sol")

    def test_callable_form_true_means_exclude(self):
        # Polarity at the user-facing API: True = exclude.
        pred = _make_global_include_pred(lambda p: p.suffix in {".pdf", ".jpg"})
        assert not pred("docs/spec.pdf")
        assert not pred("img/photo.jpg")
        assert not pred(".git/HEAD")  # floor still active
        assert pred("src/Foo.sol")
        assert pred("docs/README.md")

    def test_callable_form_receives_purepath(self):
        # The callable should receive a PurePath, so path semantics like
        # .parts and .suffix are usable without manual parsing.
        seen: list[object] = []
        def _pred(p) -> bool:
            seen.append(p)
            return False  # never exclude
        pred = _make_global_include_pred(_pred)
        pred("src/sub/Foo.sol")
        assert len(seen) == 1
        assert isinstance(seen[0], pathlib.PurePath)
        assert seen[0].suffix == ".sol"
        assert seen[0].parts == ("src", "sub", "Foo.sol")

    def test_callable_form_returning_false_allows(self):
        pred = _make_global_include_pred(lambda p: False)  # never exclude
        assert pred("anything")
        assert pred("docs/spec.pdf")
        assert not pred(".git/HEAD")  # floor still active

    def test_none_form_only_floor(self):
        pred = _make_global_include_pred(None)
        assert pred("anything.txt")
        assert pred("src/very/deep/nested/Foo.sol")
        assert not pred(".git/HEAD")


class TestGlobalExclude:
    """Integration tests: ``global_exclude`` propagates through the tool
    surface, the materializer, and the unfiltered lookup hook."""

    pytestmark = pytest.mark.asyncio

    async def test_floor_filters_tool_get(self):
        b = InMemoryBackend({
            "src/Foo.sol": "contract Foo {}",
            ".git/HEAD": "ref: refs/heads/main",
        })
        tools, _ = _tools_by_name([b])  # no global_exclude — floor only
        assert "contract Foo" in tools["get_file"].invoke({"path": "src/Foo.sol"})
        resp = tools["get_file"].invoke({"path": ".git/HEAD"})
        assert "ref:" not in resp
        assert "not found" in resp.lower()

    async def test_floor_filters_tool_list(self):
        b = InMemoryBackend({
            "src/Foo.sol": "x",
            ".git/HEAD": "y",
            ".git/config": "z",
        })
        tools, _ = _tools_by_name([b])
        listing = tools["list_files"].invoke({}).splitlines()
        assert "src/Foo.sol" in listing
        assert ".git/HEAD" not in listing
        assert ".git/config" not in listing

    async def test_floor_filters_materializer(self, tmp_path: pathlib.Path):
        """Distinct from ``forbidden_read`` — global_exclude (including
        the .git floor) DOES affect materialization."""
        b = InMemoryBackend({
            "src/Foo.sol": "ok",
            ".git/HEAD": "should not be dumped",
        })
        _, mat = fs_tools_layered([b])
        await mat.dump_to(tmp_path)
        assert (tmp_path / "src/Foo.sol").read_text() == "ok"
        assert not (tmp_path / ".git").exists()
        assert not (tmp_path / ".git/HEAD").exists()

    async def test_user_regex_filters_tool_surface(self):
        b = InMemoryBackend({
            "src/Foo.sol": "code",
            "docs/spec.pdf": "binary blob",
            "img/photo.jpg": "another blob",
        })
        tools, _ = _tools_by_name([b], global_exclude=r".*\.(pdf|jpg)")
        listing = tools["list_files"].invoke({}).splitlines()
        assert "src/Foo.sol" in listing
        assert "docs/spec.pdf" not in listing
        assert "img/photo.jpg" not in listing

    async def test_user_callable_filters_tool_surface(self):
        b = InMemoryBackend({
            "src/Foo.sol": "code",
            "docs/spec.pdf": "blob",
            "src/Bar.sol": "code",
        })
        tools, _ = _tools_by_name([b], global_exclude=lambda p: p.suffix == ".pdf")
        listing = tools["list_files"].invoke({}).splitlines()
        assert "src/Foo.sol" in listing
        assert "src/Bar.sol" in listing
        assert "docs/spec.pdf" not in listing

    async def test_user_pattern_filters_materializer(self, tmp_path: pathlib.Path):
        b = InMemoryBackend({
            "src/Foo.sol": "code",
            "docs/spec.pdf": "blob",
        })
        _, mat = fs_tools_layered([b], global_exclude=lambda p: p.suffix == ".pdf")
        await mat.dump_to(tmp_path)
        assert (tmp_path / "src/Foo.sol").exists()
        assert not (tmp_path / "docs/spec.pdf").exists()

    async def test_floor_and_user_pattern_compose(self, tmp_path: pathlib.Path):
        """Both .git AND the user pattern are excluded — the floor
        composes with the user-supplied predicate, not replaced by it."""
        b = InMemoryBackend({
            "src/Foo.sol": "code",
            ".git/HEAD": "ref",
            "docs/spec.pdf": "blob",
        })
        _, mat = fs_tools_layered([b], global_exclude=lambda p: p.suffix == ".pdf")
        await mat.dump_to(tmp_path)
        assert (tmp_path / "src/Foo.sol").exists()
        assert not (tmp_path / ".git").exists()
        assert not (tmp_path / "docs/spec.pdf").exists()

    async def test_globally_excluded_path_invisible_to_get_via_materializer(self):
        """The materializer's ``get`` (used by FileRegistry's existence
        check, etc.) must respect global_exclude — distinct from
        ``forbidden_read`` which does not affect the materializer."""
        b = InMemoryBackend({"src/Foo.sol": "code", ".git/HEAD": "ref"})
        _, mat = fs_tools_layered([b])
        assert mat.get("src/Foo.sol") == "code"
        assert mat.get(".git/HEAD") is None

    async def test_dirbackend_dump_skips_excluded_subtree(self, tmp_path: pathlib.Path):
        """``DirBackend.dump_to`` uses ``copytree(ignore=...)`` with
        relativized paths so ``.git/`` is short-circuited at the root —
        not recursed into and filtered file-by-file. Verifies the
        relativization is correct for both root-level skips and
        deeply-nested file matches."""
        src = tmp_path / "src"
        _write_tree(src, {
            "Foo.sol": "code",
            ".git/HEAD": "ref",
            ".git/objects/abc/def": "blob",
            "docs/spec.pdf": "binary",
            "subdir/Bar.sol": "more code",
        })
        out = tmp_path / "out"
        backend = DirBackend(src)
        # Exclude .pdf via callable; .git via the floor.
        mat = _LayeredMaterializer(
            [backend], global_exclude=lambda p: p.suffix == ".pdf",
        )
        await mat.dump_to(out)
        assert (out / "Foo.sol").exists()
        assert (out / "subdir/Bar.sol").exists()
        assert not (out / ".git").exists()
        assert not (out / ".git/HEAD").exists()
        assert not (out / ".git/objects/abc/def").exists()
        assert not (out / "docs/spec.pdf").exists()

    async def test_grep_excludes_globally_excluded(self):
        """grep_files must not return matches from globally-excluded files."""
        b = InMemoryBackend({
            "src/Foo.sol": "needle",
            ".git/HEAD": "needle in git",
        })
        tools, _ = _tools_by_name([b])
        resp = tools["grep_files"].invoke({
            "search_string": "needle",
            "matching_lines": False,
        })
        assert "src/Foo.sol" in resp
        assert ".git" not in resp

    async def test_global_exclude_layers_with_forbidden_read(self, tmp_path: pathlib.Path):
        """``forbidden_read`` (tool-only) and ``global_exclude``
        (everywhere) compose: forbidden_read narrows the tool view but
        not materialization; global_exclude narrows both."""
        b = InMemoryBackend({
            "src/Foo.sol": "ok",
            "secrets/key.txt": "supersecret",  # tool-blind only
            "docs/spec.pdf": "blob",            # globally excluded
        })
        tools, mat = _tools_by_name(
            [b],
            forbidden_read=r"^secrets/.*",
            global_exclude=lambda p: p.suffix == ".pdf",
        )
        listing = tools["list_files"].invoke({}).splitlines()
        assert "src/Foo.sol" in listing
        assert "secrets/key.txt" not in listing  # forbidden_read blocks
        assert "docs/spec.pdf" not in listing    # global_exclude blocks

        # Materializer: forbidden_read does NOT filter; global_exclude DOES.
        await mat.dump_to(tmp_path)
        assert (tmp_path / "src/Foo.sol").exists()
        assert (tmp_path / "secrets/key.txt").exists()  # forbidden_read tool-only
        assert not (tmp_path / "docs/spec.pdf").exists()  # global_exclude everywhere
