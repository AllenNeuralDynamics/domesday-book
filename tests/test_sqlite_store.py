"""Integration tests for SQLiteDocumentStore against a real temp database."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from domesday.core.models import Snippet, SnippetType
from domesday.stores.sqlite_store import SQLiteDocumentStore


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteDocumentStore:
    db_path = tmp_path / "test.db"
    s = SQLiteDocumentStore(path=db_path)
    await s.initialize()
    yield s
    await s.close()


def _make_snippet(**kwargs) -> Snippet:
    defaults = dict(raw_text="Some text.", project="proj")
    defaults.update(kwargs)
    return Snippet(**defaults)


# ---------------------------------------------------------------------------
# CRUD basics
# ---------------------------------------------------------------------------


async def test_add_and_get_round_trip(store: SQLiteDocumentStore) -> None:
    snippet = _make_snippet(author="alice", tags=["data", "bug"], source_file="notes.md")
    await store.add(snippet)
    got = await store.get(snippet.id)
    assert got is not None
    assert got.id == snippet.id
    assert got.raw_text == "Some text."
    assert got.project == "proj"
    assert got.author == "alice"
    assert got.tags == ["data", "bug"]
    assert got.source_file == "notes.md"
    assert got.is_active is True
    assert got.snippet_type == SnippetType.PROSE


async def test_get_nonexistent_returns_none(store: SQLiteDocumentStore) -> None:
    assert await store.get("no-such-id") is None


async def test_deactivate_marks_inactive(store: SQLiteDocumentStore) -> None:
    snippet = _make_snippet()
    await store.add(snippet)
    await store.deactivate(snippet.id)
    got = await store.get(snippet.id)
    assert got is not None
    assert got.is_active is False


async def test_update_modifies_fields(store: SQLiteDocumentStore) -> None:
    snippet = _make_snippet(raw_text="original")
    await store.add(snippet)
    snippet.raw_text = "updated"
    snippet.author = "bob"
    await store.update(snippet)
    got = await store.get(snippet.id)
    assert got is not None
    assert got.raw_text == "updated"
    assert got.author == "bob"


# ---------------------------------------------------------------------------
# list_recent
# ---------------------------------------------------------------------------


async def test_list_recent_filters_by_project(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(project="alpha", raw_text="a"))
    await store.add(_make_snippet(project="beta", raw_text="b"))
    results = await store.list_recent(project="alpha")
    assert all(s.project == "alpha" for s in results)
    assert len(results) == 1


async def test_list_recent_filters_by_author(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(author="alice", raw_text="a"))
    await store.add(_make_snippet(author="bob", raw_text="b"))
    results = await store.list_recent(author="alice")
    assert all(s.author == "alice" for s in results)
    assert len(results) == 1


async def test_list_recent_filters_by_tags(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(tags=["bug", "data"], raw_text="a"))
    await store.add(_make_snippet(tags=["feature"], raw_text="b"))
    await store.add(_make_snippet(tags=[], raw_text="c"))
    results = await store.list_recent(tags=["bug"])
    assert len(results) == 1
    assert "bug" in results[0].tags


async def test_list_recent_excludes_inactive(store: SQLiteDocumentStore) -> None:
    s = _make_snippet()
    await store.add(s)
    await store.deactivate(s.id)
    results = await store.list_recent(active_only=True)
    assert len(results) == 0


async def test_list_recent_includes_inactive_when_requested(
    store: SQLiteDocumentStore,
) -> None:
    s = _make_snippet()
    await store.add(s)
    await store.deactivate(s.id)
    results = await store.list_recent(active_only=False)
    assert len(results) == 1


async def test_list_recent_respects_limit(store: SQLiteDocumentStore) -> None:
    for i in range(5):
        await store.add(_make_snippet(raw_text=f"snippet {i}"))
    results = await store.list_recent(n=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# get_all_active
# ---------------------------------------------------------------------------


async def test_get_all_active_filters_by_project(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(project="a", raw_text="1"))
    await store.add(_make_snippet(project="b", raw_text="2"))
    s = _make_snippet(project="a", raw_text="3")
    await store.add(s)
    await store.deactivate(s.id)

    results = await store.get_all_active(project="a")
    assert len(results) == 1
    assert results[0].raw_text == "1"


async def test_get_all_active_returns_all_when_no_project(
    store: SQLiteDocumentStore,
) -> None:
    await store.add(_make_snippet(project="a", raw_text="1"))
    await store.add(_make_snippet(project="b", raw_text="2"))
    results = await store.get_all_active()
    assert len(results) == 2


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_active_only(store: SQLiteDocumentStore) -> None:
    s1 = _make_snippet(raw_text="1")
    s2 = _make_snippet(raw_text="2")
    await store.add(s1)
    await store.add(s2)
    await store.deactivate(s1.id)

    assert await store.count(active_only=True) == 1
    assert await store.count(active_only=False) == 2


async def test_count_by_project(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(project="a", raw_text="1"))
    await store.add(_make_snippet(project="a", raw_text="2"))
    await store.add(_make_snippet(project="b", raw_text="3"))
    assert await store.count(project="a") == 2
    assert await store.count(project="b") == 1


# ---------------------------------------------------------------------------
# Project operations
# ---------------------------------------------------------------------------


async def test_list_projects(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(project="zeta", raw_text="1"))
    await store.add(_make_snippet(project="alpha", raw_text="2"))
    s = _make_snippet(project="inactive_proj", raw_text="3")
    await store.add(s)
    await store.deactivate(s.id)

    projects = await store.list_projects()
    assert projects == ["alpha", "zeta"]  # sorted, excludes inactive-only project


async def test_rename_project(store: SQLiteDocumentStore) -> None:
    await store.add(_make_snippet(project="old_name", raw_text="1"))
    await store.add(_make_snippet(project="old_name", raw_text="2"))
    await store.add(_make_snippet(project="other", raw_text="3"))

    count = await store.rename_project("old_name", "new_name")
    assert count == 2
    projects = await store.list_projects()
    assert "old_name" not in projects
    assert "new_name" in projects


# ---------------------------------------------------------------------------
# Snippet type round-trip
# ---------------------------------------------------------------------------


async def test_snippet_type_preserved(store: SQLiteDocumentStore) -> None:
    for stype in SnippetType:
        s = _make_snippet(snippet_type=stype, raw_text=f"type={stype.value}")
        await store.add(s)
        got = await store.get(s.id)
        assert got is not None
        assert got.snippet_type == stype


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_db_property_raises_before_initialize() -> None:
    store = SQLiteDocumentStore(path="unused.db")
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = store.db
