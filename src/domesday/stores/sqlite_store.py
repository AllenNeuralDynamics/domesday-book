"""SQLite-backed document store for snippet metadata and raw text."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from domesday.core import models

logger = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS snippets (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL DEFAULT 'default',
    raw_text    TEXT NOT NULL,
    summary     TEXT DEFAULT '',
    snippet_type TEXT DEFAULT 'prose',
    author      TEXT DEFAULT 'anonymous',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    tags        TEXT DEFAULT '[]',
    source_file TEXT,
    parent_id   TEXT,
    is_active   INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_snippets_project ON snippets(project);
CREATE INDEX IF NOT EXISTS idx_snippets_active ON snippets(is_active);
CREATE INDEX IF NOT EXISTS idx_snippets_project_active ON snippets(project, is_active);
CREATE INDEX IF NOT EXISTS idx_snippets_created ON snippets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_snippets_author ON snippets(author);
"""

# Migration: add project column to existing databases
MIGRATIONS = [
    """\
    ALTER TABLE snippets ADD COLUMN project TEXT NOT NULL DEFAULT 'default';
    """,
]


@dataclass
class SQLiteDocumentStore:
    """DocumentStore implementation backed by a single SQLite file.

    Usage:
        store = SQLiteDocumentStore(path="./data/domesday.db")
        await store.initialize()
        await store.add(snippet)
    """

    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables if they don't exist. Call once at startup."""
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._apply_migrations()
        await self._db.commit()
        logger.info("SQLite store initialized at %s", self.path)

    async def _apply_migrations(self) -> None:
        """Apply schema migrations for existing databases."""
        for i, migration in enumerate(MIGRATIONS):
            try:
                await self.db.executescript(migration)
                logger.debug("Applied migration %d", i)
            except Exception:
                logger.debug("Migration %d already applied or not needed", i)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            logger.debug("SQLite connection closed")

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store not initialized — call await store.initialize()")
        return self._db

    # ---------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------

    async def add(self, snippet: models.Snippet) -> models.Snippet:
        await self.db.execute(
            """
            INSERT INTO snippets
                (id, project, raw_text, summary, snippet_type, author,
                 created_at, updated_at, tags, source_file, parent_id, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snippet.id,
                snippet.project,
                snippet.raw_text,
                snippet.summary,
                snippet.snippet_type.value,
                snippet.author,
                snippet.created_at.isoformat(),
                snippet.updated_at.isoformat(),
                json.dumps(snippet.tags),
                snippet.source_file,
                snippet.parent_id,
                int(snippet.is_active),
            ),
        )
        await self.db.commit()
        logger.debug(
            "Stored snippet %s (project=%s, author=%s, %d chars)",
            snippet.id[:8],
            snippet.project,
            snippet.author,
            len(snippet.raw_text),
        )
        return snippet

    async def get(self, snippet_id: str) -> models.Snippet | None:
        async with self.db.execute(
            "SELECT * FROM snippets WHERE id = ?", (snippet_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                logger.debug("Retrieved snippet %s", snippet_id[:8])
                return self._row_to_snippet(row)
            logger.debug("Snippet %s not found", snippet_id[:8])
            return None

    async def update(self, snippet: models.Snippet) -> models.Snippet:
        snippet.updated_at = datetime.now(UTC)
        await self.db.execute(
            """
            UPDATE snippets SET
                project = ?, raw_text = ?, summary = ?, snippet_type = ?,
                author = ?, updated_at = ?, tags = ?, source_file = ?,
                parent_id = ?, is_active = ?
            WHERE id = ?
            """,
            (
                snippet.project,
                snippet.raw_text,
                snippet.summary,
                snippet.snippet_type.value,
                snippet.author,
                snippet.updated_at.isoformat(),
                json.dumps(snippet.tags),
                snippet.source_file,
                snippet.parent_id,
                int(snippet.is_active),
                snippet.id,
            ),
        )
        await self.db.commit()
        logger.debug("Updated snippet %s", snippet.id[:8])
        return snippet

    async def deactivate(self, snippet_id: str) -> None:
        await self.db.execute(
            "UPDATE snippets SET is_active = 0, updated_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), snippet_id),
        )
        await self.db.commit()
        logger.info("Deactivated snippet %s", snippet_id[:8])

    async def list_recent(
        self,
        n: int = 20,
        *,
        project: str | None = None,
        author: str | None = None,
        tags: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> list[models.Snippet]:
        conditions: list[str] = []
        params: list[object] = []

        if active_only:
            conditions.append("is_active = 1")
        if project is not None:
            conditions.append("project = ?")
            params.append(project)
        if author:
            conditions.append("author = ?")
            params.append(author)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM snippets {where} ORDER BY created_at DESC LIMIT ?"
        params.append(n)

        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        snippets = [self._row_to_snippet(r) for r in rows]

        # Tag filtering done in Python since tags are JSON-encoded
        if tags:
            tag_set = set(tags)
            pre_filter = len(snippets)
            snippets = [s for s in snippets if tag_set.intersection(s.tags)]
            logger.debug(
                "Tag filter %s: %d → %d snippets",
                tags,
                pre_filter,
                len(snippets),
            )

        logger.debug(
            "list_recent(n=%d, project=%s, author=%s) → %d results",
            n,
            project,
            author,
            len(snippets),
        )
        return snippets

    async def get_all_active(
        self, *, project: str | None = None
    ) -> list[models.Snippet]:
        """Return all active snippets, optionally filtered by project."""
        if project is not None:
            query = "SELECT * FROM snippets WHERE is_active = 1 AND project = ? ORDER BY created_at"
            params: tuple[object, ...] = (project,)
        else:
            query = "SELECT * FROM snippets WHERE is_active = 1 ORDER BY created_at"
            params = ()

        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        result = [self._row_to_snippet(r) for r in rows]
        logger.debug("get_all_active(project=%s) → %d snippets", project, len(result))
        return result

    async def count(
        self, *, active_only: bool = True, project: str | None = None
    ) -> int:
        conditions: list[str] = []
        params: list[object] = []

        if active_only:
            conditions.append("is_active = 1")
        if project is not None:
            conditions.append("project = ?")
            params.append(project)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        async with self.db.execute(
            f"SELECT COUNT(*) FROM snippets {where}", params
        ) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def list_projects(self) -> list[str]:
        """Return all distinct project names."""
        async with self.db.execute(
            "SELECT DISTINCT project FROM snippets WHERE is_active = 1 ORDER BY project"
        ) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def rename_project(self, old_name: str, new_name: str) -> int:
        """Rename all snippets from old_name to new_name. Returns count updated."""
        cursor = await self.db.execute(
            "UPDATE snippets SET project = ?, updated_at = ? WHERE project = ?",
            (new_name, datetime.now(UTC).isoformat(), old_name),
        )
        await self.db.commit()
        logger.info(
            "Renamed project '%s' → '%s' in SQLite (%d rows)",
            old_name,
            new_name,
            cursor.rowcount,
        )
        return cursor.rowcount

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    @staticmethod
    def _row_to_snippet(row: aiosqlite.Row) -> models.Snippet:
        return models.Snippet(
            id=row["id"],
            project=row["project"],
            raw_text=row["raw_text"],
            summary=row["summary"],
            snippet_type=models.SnippetType(row["snippet_type"]),
            author=row["author"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            tags=json.loads(row["tags"]),
            source_file=row["source_file"],
            parent_id=row["parent_id"],
            is_active=bool(row["is_active"]),
        )
