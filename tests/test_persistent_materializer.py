"""Tests for ``PersistentMaterializer`` — dump into a reused target."""

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
    """Identical bytes must keep their mtime: many builds fingerprint on it."""
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
    """Compiler output and package caches are not view content."""
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
    """A file this materializer did not write is not its to delete."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    target = tmp_path / "out"
    mat = PersistentMaterializer(base)
    await mat.dump_to(target)

    intruder = target / "not_ours.rs"
    intruder.write_text("someone else's")
    await mat.dump_to(target)

    assert intruder.read_text() == "someone else's"


async def test_a_corrupt_manifest_costs_a_bulk_copy_and_loses_nothing(tmp_path):
    """Unreadable is absent, not empty: empty would skip the base copy."""
    base = _project(tmp_path / "src", a__rs="fn a() {}")
    target = tmp_path / "out"
    mat = PersistentMaterializer(base)
    await mat.dump_to(target)
    (target / MATERIALIZED_MANIFEST).write_text("{not json")

    await mat.dump_to(target)

    assert (target / "a/rs").read_text() == "fn a() {}"
    assert json.loads((target / MATERIALIZED_MANIFEST).read_text())["overlaid"] == []


async def test_binary_content_survives_the_first_dump(tmp_path):
    """Overlays are text-only; binary files ride the base copy and are not rewritten."""
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
    base = _project(tmp_path / "src", a__rs="pristine")
    edits = DictBackend({"a/rs": "edited"})
    target = tmp_path / "out"

    tools, mat = fs_tools_layered([edits, base], materializer=persistent_materializer)
    await mat.dump_to(target)

    assert isinstance(mat, PersistentMaterializer)
    get_file = next(t for t in tools if t.name == "get_file")
    assert "edited" in get_file.invoke({"path": "a/rs"})
    assert (target / "a/rs").read_text() == "edited"


async def test_concurrent_readers_never_see_a_partial_file(tmp_path):
    """A reader mid-dump must see the old file or the new one, never a truncated one."""
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
