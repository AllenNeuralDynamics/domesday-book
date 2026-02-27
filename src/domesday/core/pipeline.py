"""Core pipeline: ingest, retrieve, and generate.

This module knows about the protocols but not the implementations.
All backends are injected at construction time.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from domesday.core import models, protocols

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a research assistant with access to a team knowledge base called Domesday.
Answer the user's question using ONLY the provided context snippets.
If the context doesn't contain the answer, say so clearly — do not fabricate.
Cite snippets by their ID in square brackets, e.g. [snippet-abc123].
If snippets contradict each other, note the conflict and prefer the more recent
entry unless context suggests otherwise.
When citing, use the short ID (first 8 chars).
"""


@dataclass
class Pipeline:
    """Orchestrates ingest, retrieval, and generation.

    All dependencies are injected — swap any backend without
    changing this code.

    The `default_project` is used when no project is specified
    on individual operations. Set it in config or override per-call.
    """

    doc_store: protocols.DocumentStore
    vec_store: protocols.VectorStore
    embedder: protocols.Embedder
    generator: protocols.Generator
    chunker: protocols.Chunker
    default_project: str
    reranker: protocols.Reranker | None = None

    def _resolve_project(self, project: str | None) -> str:
        """Use explicit project if given, otherwise fall back to default."""
        return project if project is not None else self.default_project

    # ---------------------------------------------------------------
    # Ingest
    # ---------------------------------------------------------------

    async def add_snippet(
        self,
        text: str,
        *,
        project: str | None = None,
        author: str = "anonymous",
        tags: list[str] | None = None,
        source_file: str | None = None,
        snippet_type: models.SnippetType = models.SnippetType.PROSE,
    ) -> models.Snippet:
        """Add a single snippet: store, chunk, embed, index."""
        resolved_project = self._resolve_project(project)

        snippet = models.Snippet(
            raw_text=text,
            project=resolved_project,
            author=author,
            tags=tags or [],
            source_file=source_file,
            snippet_type=snippet_type,
        )

        # 1. Persist metadata + raw text
        snippet = await self.doc_store.add(snippet)

        # 2. Chunk
        chunks = self.chunker.chunk(snippet)
        logger.debug("Snippet %s → %d chunks", snippet.id[:8], len(chunks))

        # 3. Embed
        if chunks:
            chunk_texts = [c.text for c in chunks]
            logger.debug(
                "Embedding %d chunks for snippet %s", len(chunks), snippet.id[:8]
            )
            embeddings = await self.embedder.embed(chunk_texts)

            # 4. Index in vector store with project metadata
            await self.vec_store.add_chunks(
                chunks, embeddings, project=resolved_project
            )
            logger.debug("Indexed %d chunks in vector store", len(chunks))

        logger.info(
            "Added snippet %s to project '%s' (%d chunks)",
            snippet.id[:8],
            resolved_project,
            len(chunks),
        )
        return snippet

    async def ingest_file(
        self,
        path: Path,
        *,
        project: str | None = None,
        author: str = "anonymous",
        delimiter: str | None = None,
    ) -> list[models.Snippet]:
        """Ingest a single file, splitting into snippets if delimited.

        Supports .md, .txt, .py, .json, .csv, .yaml/.yml.
        Markdown files are split on '---' or '## ' headings by default.
        """
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        logger.debug(
            "Ingesting file %s (%d chars, type=%s)",
            path.name,
            len(text),
            suffix,
        )

        type_map: dict[str, models.SnippetType] = {
            ".py": models.SnippetType.CODE,
            ".json": models.SnippetType.CODE,
            ".yaml": models.SnippetType.CODE,
            ".yml": models.SnippetType.CODE,
            ".csv": models.SnippetType.TABLE,
            ".md": models.SnippetType.PROSE,
            ".txt": models.SnippetType.PROSE,
        }
        snippet_type = type_map.get(suffix, models.SnippetType.PROSE)

        sections = self._split_file(text, suffix, delimiter)

        snippets: list[models.Snippet] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            snippet = await self.add_snippet(
                section,
                project=project,
                author=author,
                source_file=str(path),
                snippet_type=snippet_type,
            )
            snippets.append(snippet)

        return snippets

    async def ingest_directory(
        self,
        directory: Path,
        *,
        project: str | None = None,
        author: str = "anonymous",
        extensions: set[str] | None = None,
        recursive: bool = True,
    ) -> models.IngestResult:
        """Bulk-ingest all matching files from a directory."""
        if extensions is None:
            extensions = {".md", ".txt", ".py", ".json", ".csv", ".yaml", ".yml"}

        glob_pattern = "**/*" if recursive else "*"
        files = [
            f
            for f in directory.glob(glob_pattern)
            if f.is_file() and f.suffix.lower() in extensions
        ]

        logger.info(
            "Bulk ingest: %d files in %s (project=%s, extensions=%s)",
            len(files),
            directory,
            self._resolve_project(project),
            extensions,
        )

        total_snippets = 0
        total_chunks = 0
        errors: list[str] = []

        for file_path in sorted(files):
            try:
                snippets = await self.ingest_file(
                    file_path, project=project, author=author
                )
                total_snippets += len(snippets)
                total_chunks += sum(len(self.chunker.chunk(s)) for s in snippets)
                logger.info("Ingested %s → %d snippets", file_path, len(snippets))
            except Exception as exc:
                msg = f"{file_path}: {exc}"
                logger.error("Failed to ingest %s", msg)
                errors.append(msg)

        result = models.IngestResult(
            snippets_created=total_snippets,
            chunks_created=total_chunks,
            files_processed=len(files) - len(errors),
            errors=errors,
        )
        logger.info(
            "Bulk ingest complete: %d files → %d snippets, %d chunks, %d errors",
            result.files_processed,
            result.snippets_created,
            result.chunks_created,
            len(result.errors),
        )
        return result

    # ---------------------------------------------------------------
    # Revision
    # ---------------------------------------------------------------

    async def revise_snippet(
        self,
        snippet_id: str,
        new_text: str,
        *,
        author: str = "anonymous",
    ) -> models.Snippet:
        """Create a new snippet that supersedes an existing one.

        Inherits the project from the original snippet.
        """
        old = await self.doc_store.get(snippet_id)
        if old is None:
            raise ValueError(f"Snippet {snippet_id} not found")

        logger.info(
            "Revising snippet %s (project=%s) by %s",
            snippet_id[:8],
            old.project,
            author,
        )

        await self.doc_store.deactivate(snippet_id)
        await self.vec_store.delete_by_snippet(snippet_id)

        return await self.add_snippet(
            new_text,
            project=old.project,
            author=author,
            tags=old.tags,
            source_file=old.source_file,
            snippet_type=old.snippet_type,
        )

    # ---------------------------------------------------------------
    # Retrieval + Generation
    # ---------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        project: str | None = None,
        k: int = 10,
        tags: Sequence[str] | None = None,
        min_score: float = 0.0,
    ) -> list[models.SearchResult]:
        """Semantic search over the knowledge base.

        Args:
            project: Scope search to a specific project. If None, uses
                     default_project. Pass "__all__" to search across
                     all projects.
            min_score: Minimum cosine similarity threshold (0.0–1.0).
        """
        threshold = min_score

        # Resolve project: None → default, "__all__" → no filter
        if project == "__all__":
            search_project = None
        else:
            search_project = self._resolve_project(project)

        logger.debug(
            "Search: query='%s', project=%s, k=%d, threshold=%.3f",
            query[:60],
            search_project or "(all)",
            k,
            threshold,
        )

        query_embedding = (await self.embedder.embed([query]))[0]

        raw_results = await self.vec_store.search(
            query_embedding, k=k, project=search_project, filter_tags=tags
        )

        results: list[models.SearchResult] = []
        filtered_count = 0
        for chunk_id, snippet_id, score in raw_results:
            if score < threshold:
                filtered_count += 1
                continue
            snippet = await self.doc_store.get(snippet_id)
            if snippet is None or not snippet.is_active:
                continue
            results.append(
                models.SearchResult(
                    snippet=snippet,
                    chunk_text="",
                    score=score,
                    chunk_id=chunk_id,
                )
            )

        if filtered_count > 0:
            logger.debug(
                "Score threshold %.3f filtered out %d/%d results",
                threshold,
                filtered_count,
                len(raw_results),
            )

        # Optional LLM-based reranking
        if self.reranker is not None and results:
            pre_rerank = len(results)
            logger.debug("Reranking %d results with LLM", pre_rerank)
            results = await self.reranker.rerank(query, results)
            logger.debug("Reranker: %d → %d results", pre_rerank, len(results))

        logger.info(
            "Search complete: %d results for query='%s'",
            len(results),
            query[:60],
        )
        return results

    async def ask(
        self,
        question: str,
        *,
        project: str | None = None,
        k: int = 10,
        tags: Sequence[str] | None = None,
        min_score: float = 0.0,
        system_prompt: str | None = None,
    ) -> models.RAGResponse:
        """Full RAG: retrieve relevant context, then generate an answer."""
        logger.info(
            "RAG ask: '%s' (project=%s)", question[:60], project or self.default_project
        )
        results = await self.search(
            question, project=project, k=k, tags=tags, min_score=min_score
        )
        logger.debug("Retrieved %d snippets for generation context", len(results))

        response = await self.generator.generate(
            query=question,
            context=results,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        response.query = question
        logger.info(
            "RAG answer generated: %d chars from %d sources",
            len(response.answer),
            len(response.sources),
        )
        return response

    # ---------------------------------------------------------------
    # Project management
    # ---------------------------------------------------------------

    async def list_projects(self) -> list[str]:
        """Return all project names that have active snippets."""
        return await self.doc_store.list_projects()

    async def rename_project(self, old_name: str, new_name: str) -> int:
        """Rename a project across both document and vector stores.

        Returns the number of snippets renamed.
        """
        snippet_count = await self.doc_store.rename_project(old_name, new_name)
        await self.vec_store.rename_project(old_name, new_name)  # type: ignore[attr-defined]
        logger.info(
            "Renamed project '%s' → '%s' (%d snippets)",
            old_name,
            new_name,
            snippet_count,
        )
        return snippet_count

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _split_file(
        text: str,
        suffix: str,
        delimiter: str | None,
    ) -> list[str]:
        """Split file contents into sections for multi-snippet ingest."""
        if delimiter:
            return text.split(delimiter)

        if suffix == ".md":
            if "\n---\n" in text:
                return text.split("\n---\n")
            import re

            parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
            return [p for p in parts if p.strip()]

        return [text]
