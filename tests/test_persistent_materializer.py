"""Tests for ``PersistentMaterializer`` — the dump strategy for a target that is reused.

The default materializer fills a fresh directory, so every dump writes every file. That is right
for a temp dir and wrong once something else owns state in the target, which is what happens as
soon as a build system runs there. Three properties follow, and each is tested here because each
one is invisible until a real toolchain is pointed at the directory:

* an unchanged file keeps its mtime, because a build that fingerprints on mtime rebuilds
  everything downstream of a rewritten-but-identical file;
* content the dump did not put there survives it, because a warm build directory accumulates
  compiler output and package caches that are not view content;
* a path an overlay stops serving is restored from the base if the base has one and removed if
  not, because a stale file left behind still compiles and a deleted project file breaks a build
  that was fine before anyone edited anything;
* the base is read once however many dumps follow, because re-comparing a checkout that never
  changes is the copy this class exists to avoid, moved rather than removed.
"""

import asyncio
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

import pytest

from graphcore.tools.vfs import (
    MATERIALIZED_MANIFEST,
    DictBackend,
    DirBackend,
    PersistentMaterializer,
    fs_tools_layered,
    persistent_materializer,
)

pytestmark = pytest.mark.asyncio


@dataclass
class _CountingBackend:
    """Wraps a backend and counts what the materializer asks of it."""

    inner: DirBackend
    reads: int = 0
    dumps: int = 0

    def get(self, path: str) -> str | None:
        self.reads += 1
        return self.inner.get(path)

    def list(self) -> Iterable[str]:
        self.reads += 1
        return self.inner.list()

    async def dump_to(self, target, include_path=None) -> None:
        self.dumps += 1
        await self.inner.dump_to(target, include_path=include_path)


def _project(root: pathlib.Path, **files: str) -> DirBackend:
    for name, content in files.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return DirBackend(root, cache_listing=False)


async def test_the_first_dump_fills_an_empty_target(tmp_path):
    base = _project(tmp_path / "src", lib__rs="fn main() {}")
    target = tmp_path / "out"

    await PersistentMaterializer(base).dump_to(target)

    assert (target / "lib/rs").read_text() == "fn main() {}"
    assert json.loads((target / MATERIALIZED_MANIFEST).read_text())["overlaid"] == []


async def test_an_unchanged_file_is_not_rewritten(tmp_path):
    """The property the whole class exists for.

    cargo fingerprints on mtime, so a rewritten-but-identical file is a change, and everything
    downstream of it rebuilds. Over a dependency graph that is minutes per dump — the entire cost a
    warm directory exists to avoid.
    """
    base = _project(tmp_path / "src", a__rs="const A: u8 = 1;")
    target = tmp_path / "out"
    mat = PersistentMaterializer(base)
    await mat.dump_to(target)
    before = (target / "a/rs").stat().st_mtime_ns

    await mat.dump_to(target)

    assert (target / "a/rs").stat().st_mtime_ns == before


async def test_a_changed_file_is_rewritten(tmp_path):
    base = _project(tmp_path / "src", a__rs="const A: u8 = 1;")
    edits = DictBackend()
    target = tmp_path / "out"
    mat = PersistentMaterializer(base, [edits])
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
    mat = PersistentMaterializer(base)
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


async def test_an_overlay_that_invented_a_file_takes_it_away(tmp_path):
    """A reverted edit that left its file behind would keep compiling — the stale copy is still on
    disk and still valid Rust. Nothing else serves this path, so undo means removal."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    edits = DictBackend({"b/rs": "fn b() {}"})
    target = tmp_path / "out"
    mat = PersistentMaterializer(base, [edits])
    await mat.dump_to(target)
    assert (target / "b/rs").exists()

    del edits.files["b/rs"]
    await mat.dump_to(target)

    assert not (target / "b/rs").exists()
    assert (target / "a/rs").exists()


async def test_an_overlay_that_modified_a_file_restores_the_base(tmp_path):
    """The other half of undo, and the one a flat stack cannot express. A file the project ships
    and an overlay rewrote must come *back*, not vanish — deleting it would break a build that was
    fine before anyone edited anything."""
    base = _project(tmp_path / "src", a__rs="pristine")
    edits = DictBackend({"a/rs": "munged"})
    target = tmp_path / "out"
    mat = PersistentMaterializer(base, [edits])
    await mat.dump_to(target)
    assert (target / "a/rs").read_text() == "munged"

    del edits.files["a/rs"]
    await mat.dump_to(target)

    assert (target / "a/rs").read_text() == "pristine"


async def test_the_base_is_read_once_however_many_dumps_follow(tmp_path):
    """The third reason this class exists: a project checkout is the bulk of the view and none of
    the churn, so comparing all of it on every dump is the copy the class avoids, moved rather than
    removed."""
    base = _CountingBackend(_project(tmp_path / "src", a__rs="fn a() {}"))
    edits = DictBackend({"b/rs": "fn b() {}"})
    target = tmp_path / "out"
    mat = PersistentMaterializer(base, [edits])

    await mat.dump_to(target)
    after_first = base.reads
    edits.files["b/rs"] = "fn b() { todo!() }"
    await mat.dump_to(target)

    assert base.dumps == 1
    assert base.reads == after_first, "the base was re-read on a later dump"


async def test_removal_is_limited_to_what_this_materializer_wrote(tmp_path):
    """The manifest is the boundary. A file that merely *looks* like view content — same name, put
    there by something else — is not this materializer's to delete."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    target = tmp_path / "out"
    mat = PersistentMaterializer(base)
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
    mat = PersistentMaterializer(base)
    await mat.dump_to(target)
    (target / MATERIALIZED_MANIFEST).write_text("{not json")

    await mat.dump_to(target)

    assert (target / "a/rs").read_text() == "fn a() {}"
    assert json.loads((target / MATERIALIZED_MANIFEST).read_text())["overlaid"] == []


async def test_binary_content_survives_the_first_dump(tmp_path):
    """``get`` is text-only, so incremental dumps cannot carry bytes. The first dump delegates to
    the backends' own ``dump_to``, which copies them — and since an edit layer's content is ``str``,
    no edit can ever change a binary file, so writing it once is enough."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")
    (src / "a.rs").write_text("fn a() {}")
    target = tmp_path / "out"

    mat = PersistentMaterializer(DirBackend(src, cache_listing=False))
    await mat.dump_to(target)
    await mat.dump_to(target)

    assert (target / "logo.png").read_bytes() == b"\x89PNG\r\n\x1a\n\x00\xff"


async def test_layer_priority_holds_at_materialization(tmp_path):
    base = _project(tmp_path / "src", a__rs="pristine")
    edits = DictBackend({"a/rs": "edited"})
    target = tmp_path / "out"

    await PersistentMaterializer(base, [edits]).dump_to(target)

    assert (target / "a/rs").read_text() == "edited"


async def test_globally_excluded_paths_are_never_written(tmp_path):
    base = _project(tmp_path / "src", a__rs="keep", secret__rs="drop")
    target = tmp_path / "out"

    await PersistentMaterializer(base, global_exclude=r"secret/.*").dump_to(target)

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

    # The factory splits the read stack the way the class wants it: lowest priority is the base.
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
    mat = PersistentMaterializer(base, [edits])
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
