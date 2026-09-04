"""Tests for ``PersistentMaterializer`` — the dump strategy for a target that is reused.

The default materializer fills a fresh directory, so every dump writes every file. That is right
for a temp dir and wrong once something else owns state in the target, which is what happens as
soon as a build system runs there. Three properties follow, and each is tested here because each
one is invisible until a real toolchain is pointed at the directory:

* an unchanged file keeps its mtime, because a build that fingerprints on mtime rebuilds
  everything downstream of a rewritten-but-identical file;
* content the dump did not put there survives it, because a warm build directory accumulates
  compiler output and package caches that are not view content;
* a path the view stops serving is removed, because a stale file left behind still compiles.
"""

import asyncio
import json
import pathlib
from dataclasses import dataclass, field
from typing import Iterable

import pytest

from graphcore.tools.vfs import (
    MATERIALIZED_MANIFEST,
    DirBackend,
    PersistentMaterializer,
    fs_tools_layered,
    persistent_materializer,
)

pytestmark = pytest.mark.asyncio


@dataclass
class DictBackend:
    """An in-memory ``FSBackend``, standing in for an edit overlay."""

    files: dict[str, str] = field(default_factory=dict)

    def get(self, path: str) -> str | None:
        return self.files.get(path)

    def list(self) -> Iterable[str]:
        return list(self.files)

    async def dump_to(self, target, include_path=None) -> None:
        for path, content in self.files.items():
            if include_path is not None and not include_path(path):
                continue
            dest = target / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)


def _project(root: pathlib.Path, **files: str) -> DirBackend:
    for name, content in files.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return DirBackend(root, cache_listing=False)


async def test_the_first_dump_fills_an_empty_target(tmp_path):
    base = _project(tmp_path / "src", lib__rs="fn main() {}")
    target = tmp_path / "out"

    await PersistentMaterializer([base]).dump_to(target)

    assert (target / "lib/rs").read_text() == "fn main() {}"
    assert json.loads((target / MATERIALIZED_MANIFEST).read_text())["written"] == ["lib/rs"]


async def test_an_unchanged_file_is_not_rewritten(tmp_path):
    """The property the whole class exists for.

    cargo fingerprints on mtime, so a rewritten-but-identical file is a change, and everything
    downstream of it rebuilds. Over a dependency graph that is minutes per dump — the entire cost a
    warm directory exists to avoid.
    """
    base = _project(tmp_path / "src", a__rs="const A: u8 = 1;")
    target = tmp_path / "out"
    mat = PersistentMaterializer([base])
    await mat.dump_to(target)
    before = (target / "a/rs").stat().st_mtime_ns

    await mat.dump_to(target)

    assert (target / "a/rs").stat().st_mtime_ns == before


async def test_a_changed_file_is_rewritten(tmp_path):
    base = _project(tmp_path / "src", a__rs="const A: u8 = 1;")
    edits = DictBackend()
    target = tmp_path / "out"
    mat = PersistentMaterializer([edits, base])
    await mat.dump_to(target)

    edits.files["a/rs"] = "const A: u8 = 2;"
    await mat.dump_to(target)

    assert (target / "a/rs").read_text() == "const A: u8 = 2;"


async def test_content_the_dump_did_not_write_survives(tmp_path):
    """A persistent target accumulates build output, a package cache, lock files. None of it is
    view content, and a dump that cleaned the directory would throw away exactly what reusing it
    was for."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    target = tmp_path / "out"
    mat = PersistentMaterializer([base])
    await mat.dump_to(target)

    artifact = target / "target" / "debug" / "a.rlib"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("compiled")
    cache = target / ".cargo" / "registry" / "x.crate"
    cache.parent.mkdir(parents=True)
    cache.write_text("fetched")

    await mat.dump_to(target)

    assert artifact.read_text() == "compiled"
    assert cache.read_text() == "fetched"


async def test_a_path_the_view_stops_serving_is_removed(tmp_path):
    """A reverted edit that left its file behind would keep compiling — the stale copy is still on
    disk and still valid Rust. Removal is what makes an edit undoable in a reused target."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    edits = DictBackend({"b/rs": "fn b() {}"})
    target = tmp_path / "out"
    mat = PersistentMaterializer([edits, base])
    await mat.dump_to(target)
    assert (target / "b/rs").exists()

    del edits.files["b/rs"]
    await mat.dump_to(target)

    assert not (target / "b/rs").exists()
    assert (target / "a/rs").exists()


async def test_removal_is_limited_to_what_this_materializer_wrote(tmp_path):
    """The manifest is the boundary. A file that merely *looks* like view content — same name, put
    there by something else — is not this materializer's to delete."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    target = tmp_path / "out"
    mat = PersistentMaterializer([base])
    await mat.dump_to(target)

    intruder = target / "not_ours.rs"
    intruder.write_text("someone else's")
    await mat.dump_to(target)

    assert intruder.read_text() == "someone else's"


async def test_a_corrupt_manifest_costs_a_bulk_copy_and_loses_nothing(tmp_path):
    """Read as absent rather than as empty: an unreadable note must never be taken to mean "this
    target has nothing in it", which would make the next dump skip the bulk copy that carries
    binary content."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    target = tmp_path / "out"
    mat = PersistentMaterializer([base])
    await mat.dump_to(target)
    (target / MATERIALIZED_MANIFEST).write_text("{not json")

    await mat.dump_to(target)

    assert (target / "a/rs").read_text() == "fn a() {}"
    assert json.loads((target / MATERIALIZED_MANIFEST).read_text())["written"] == ["a/rs"]


async def test_binary_content_survives_the_first_dump(tmp_path):
    """``get`` is text-only, so incremental dumps cannot carry bytes. The first dump delegates to
    the backends' own ``dump_to``, which copies them — and since an edit layer's content is ``str``,
    no edit can ever change a binary file, so writing it once is enough."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")
    (src / "a.rs").write_text("fn a() {}")
    target = tmp_path / "out"

    mat = PersistentMaterializer([DirBackend(src, cache_listing=False)])
    await mat.dump_to(target)
    await mat.dump_to(target)

    assert (target / "logo.png").read_bytes() == b"\x89PNG\r\n\x1a\n\x00\xff"


async def test_layer_priority_holds_at_materialization(tmp_path):
    base = _project(tmp_path / "src", a__rs="pristine")
    edits = DictBackend({"a/rs": "edited"})
    target = tmp_path / "out"

    await PersistentMaterializer([edits, base]).dump_to(target)

    assert (target / "a/rs").read_text() == "edited"


async def test_globally_excluded_paths_are_never_written(tmp_path):
    base = _project(tmp_path / "src", a__rs="keep", secret__rs="drop")
    target = tmp_path / "out"

    await PersistentMaterializer([base], global_exclude=r"secret/.*").dump_to(target)

    assert (target / "a/rs").exists()
    assert not (target / "secret/rs").exists()


async def test_the_factory_wires_it_through_fs_tools_layered(tmp_path):
    """The read tools and the materializer come from one call over one stack, which is the property
    that stops them disagreeing about what the view contains."""
    base = _project(tmp_path / "src", a__rs="pristine")
    edits = DictBackend({"a/rs": "edited"})
    target = tmp_path / "out"

    tools, mat = fs_tools_layered([edits, base], materializer=persistent_materializer)
    await mat.dump_to(target)

    assert isinstance(mat, PersistentMaterializer)
    get_file = next(t for t in tools if t.name == "get_file")
    # What the agent reads and what the build compiles are the same text.
    assert "edited" in get_file.invoke({"path": "a/rs"})
    assert (target / "a/rs").read_text() == "edited"


async def test_concurrent_readers_never_see_a_partial_file(tmp_path):
    """Writes are atomic because a persistent target invites a build running against it while a
    dump is in flight; a truncated source file is a compile error nobody can reproduce."""
    base = _project(tmp_path / "src", a__rs="x" * 100_000)
    edits = DictBackend()
    target = tmp_path / "out"
    mat = PersistentMaterializer([edits, base])
    await mat.dump_to(target)

    seen: list[int] = []

    async def read_repeatedly() -> None:
        for _ in range(200):
            try:
                seen.append(len((target / "a/rs").read_text()))
            except FileNotFoundError:
                seen.append(-1)
            await asyncio.sleep(0)

    edits.files["a/rs"] = "y" * 200_000
    await asyncio.gather(mat.dump_to(target), read_repeatedly())

    assert set(seen) <= {100_000, 200_000}, "a reader saw a partially written file"
