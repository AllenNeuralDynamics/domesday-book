"""CLI interface for domesday.

Usage:
    domes -p vbo add "Some important caveat."
    domes -p vbo add --file notes.md --author ben
    domes -p vbo ingest ./project-notes/ --author ben
    domes -p vbo search "VBO timestamp issues"
    domes -p vbo ask "What are the known caveats?"
    domes -p vbo list --n 10
    domes projects
    domes -p vbo stats
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

import rich.status
import typer

from domesday import config
from domesday.core import models

app = typer.Typer(
    help="domesday — a shared knowledge base that keeps your team's AI tools informed.",
)


def _run(coro):
    """Run an async coroutine from synchronous Typer commands."""
    return asyncio.run(coro)


@app.callback()
def _callback(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(help="Path to domesday.toml config file."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(
            "--project", "-p", help="Project to operate on (overrides config default)."
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show INFO-level logs."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Show DEBUG-level logs (very verbose)."),
    ] = False,
) -> None:
    """Configure logging and store shared state."""
    import dotenv

    dotenv.load_dotenv()

    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["project"] = project


# -------------------------------------------------------------------
# add
# -------------------------------------------------------------------


@app.command()
def add(
    ctx: typer.Context,
    text: Annotated[str | None, typer.Argument(help="Snippet text to add.")] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Read snippet from file."),
    ] = None,
    author: Annotated[
        str,
        typer.Option("--author", "-a", help="Who is adding this."),
    ] = "anonymous",
    tags: Annotated[
        str, typer.Option("--tags", "-t", help="Comma-separated tags.")
    ] = "",
    snippet_type: Annotated[
        models.SnippetType,
        typer.Option("--type", help="Snippet type."),
    ] = models.SnippetType.PROSE,
) -> None:
    """Add a single snippet to the knowledge base."""
    if file:
        content = file.read_text(encoding="utf-8")
    elif text:
        content = text
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        edited = typer.edit(text="# Enter your snippet below\n")
        if not edited:
            typer.echo("Aborted — no content provided.", err=True)
            return
        content = edited

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    async def _add() -> None:
        with rich.status.Status("Adding snippet..."):
            pipeline = await config.build_pipeline(ctx.obj["config_path"])
            snippet = await pipeline.add_snippet(
                content,
                project=ctx.obj["project"],
                author=author,
                tags=tag_list,
                snippet_type=snippet_type,
            )
        typer.echo(
            f"✓ Added snippet {snippet.id[:8]} to project '{snippet.project}' "
            f"({len(content)} chars)"
        )

    _run(_add())


# -------------------------------------------------------------------
# ingest
# -------------------------------------------------------------------


@app.command()
def ingest(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="File or directory to ingest.")],
    author: Annotated[
        str,
        typer.Option("--author", "-a", help="Who is adding these."),
    ] = "anonymous",
    extensions: Annotated[
        str,
        typer.Option(
            "--extensions", "-e", help="Comma-separated file extensions to include."
        ),
    ] = ".md,.txt,.py,.json,.csv,.yaml,.yml",
    no_recursive: Annotated[
        bool,
        typer.Option("--no-recursive", help="Don't recurse into subdirectories."),
    ] = False,
) -> None:
    """Ingest a file or directory into the knowledge base."""
    ext_set = {
        e.strip() if e.strip().startswith(".") else f".{e.strip()}"
        for e in extensions.split(",")
    }

    async def _ingest() -> None:
        pipeline = await config.build_pipeline(ctx.obj["config_path"])
        project = ctx.obj["project"]

        if path.is_file():
            content = path.read_text(encoding="utf-8")
            with rich.status.Status("Ingesting file..."):
                snippet = await pipeline.add_snippet(
                    content,
                    project=project,
                    author=author,
                    source_file=str(path),
                )
            proj_label = project or pipeline.default_project
            typer.echo(
                f"✓ Ingested {path.name} → snippet {snippet.id[:8]} "
                f"(project: {proj_label})"
            )
        else:
            with rich.status.Status("Ingesting directory..."):
                result = await pipeline.ingest_directory(
                    path,
                    project=project,
                    author=author,
                    extensions=ext_set,
                    recursive=not no_recursive,
                )
            proj_label = project or pipeline.default_project
            typer.echo(
                f"✓ Ingested {result.files_processed} files → "
                f"{result.snippets_created} snippets, "
                f"{result.chunks_created} chunks "
                f"(project: {proj_label})"
            )
            if result.errors:
                typer.echo(f"⚠ {len(result.errors)} errors:", err=True)
                for err in result.errors:
                    typer.echo(f"  - {err}", err=True)

    _run(_ingest())


# -------------------------------------------------------------------
# search
# -------------------------------------------------------------------


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Search query.")],
    n: Annotated[int, typer.Option("--n", "-n", help="Number of results.")] = 5,
    tags: Annotated[
        str,
        typer.Option("--tags", "-t", help="Filter by tags (comma-separated)."),
    ] = "",
    all_projects: Annotated[
        bool,
        typer.Option("--all-projects", help="Search across all projects."),
    ] = False,
) -> None:
    """Semantic search over the knowledge base."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    async def _search() -> None:
        with rich.status.Status("Searching..."):
            cfg = config.Config.load(ctx.obj["config_path"])
            pipeline = await config.build_pipeline(ctx.obj["config_path"])
            project = "__all__" if all_projects else ctx.obj["project"]
            results = await pipeline.search(
                query, project=project, k=n, tags=tag_list, min_score=cfg.retrieval.min_score
            )

        if not results:
            typer.echo("No results found.")
            return

        for i, r in enumerate(results, 1):
            sid = r.snippet.id[:8]
            score = f"{r.score:.3f}"
            author = r.snippet.author
            date = r.snippet.created_at.strftime("%Y-%m-%d")
            proj = r.snippet.project
            tags_str = f" [{', '.join(r.snippet.tags)}]" if r.snippet.tags else ""
            typer.echo(f"\n{'─' * 60}")
            typer.echo(
                f"  #{i}  snippet-{sid}  score={score}  "
                f"{author} {date}  project={proj}{tags_str}"
            )
            typer.echo(f"{'─' * 60}")
            text = r.snippet.raw_text
            if len(text) > 500:
                text = text[:500] + "…"
            typer.echo(text)

    _run(_search())


# -------------------------------------------------------------------
# ask
# -------------------------------------------------------------------


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    n: Annotated[
        int,
        typer.Option("--n", "-n", help="Number of context snippets to retrieve."),
    ] = 8,
    show_sources: Annotated[
        bool,
        typer.Option(
            "--show-sources", "-s", help="Print source snippets after answer."
        ),
    ] = False,
    all_projects: Annotated[
        bool,
        typer.Option("--all-projects", help="Search across all projects."),
    ] = False,
) -> None:
    """Ask a question — retrieves context and generates an answer."""

    async def _ask() -> None:
        with rich.status.Status("Thinking..."):
            cfg = config.Config.load(ctx.obj["config_path"])
            pipeline = await config.build_pipeline(ctx.obj["config_path"])
            project = "__all__" if all_projects else ctx.obj["project"]
            response = await pipeline.ask(
                question, project=project, k=n, min_score=cfg.retrieval.min_score
            )

        typer.echo(f"\n{response.answer}\n")

        if show_sources:
            typer.echo(f"{'─' * 60}")
            typer.echo(f"Sources ({len(response.sources)} snippets retrieved):")
            for r in response.sources:
                sid = r.snippet.id[:8]
                proj = r.snippet.project
                typer.echo(f"  [{sid}] ({proj}) {r.snippet.raw_text[:80]}…")

    _run(_ask())


# -------------------------------------------------------------------
# list
# -------------------------------------------------------------------


@app.command("list")
def list_snippets(
    ctx: typer.Context,
    n: Annotated[
        int, typer.Option("--n", "-n", help="Number of snippets to show.")
    ] = 20,
    author: Annotated[
        str | None, typer.Option("--author", "-a", help="Filter by author.")
    ] = None,
    all_projects: Annotated[
        bool,
        typer.Option("--all-projects", help="List across all projects."),
    ] = False,
) -> None:
    """List recent snippets."""

    async def _list() -> None:
        pipeline = await config.build_pipeline(ctx.obj["config_path"])
        project = (
            None if all_projects else (ctx.obj["project"] or pipeline.default_project)
        )
        snippets = await pipeline.doc_store.list_recent(
            n, project=project, author=author
        )

        if not snippets:
            typer.echo("No snippets found.")
            return

        for s in snippets:
            sid = s.id[:8]
            date = s.created_at.strftime("%Y-%m-%d %H:%M")
            tags_str = f" [{', '.join(s.tags)}]" if s.tags else ""
            preview = s.raw_text[:80].replace("\n", " ")
            typer.echo(
                f"  {sid}  {s.project:12s}  {date}  {s.author:12s}{tags_str}  {preview}…"
            )

    _run(_list())


# -------------------------------------------------------------------
# projects
# -------------------------------------------------------------------


@app.command()
def projects(ctx: typer.Context) -> None:
    """List all projects in the knowledge base."""

    async def _projects() -> None:
        pipeline = await config.build_pipeline(ctx.obj["config_path"])
        project_list = await pipeline.list_projects()

        if not project_list:
            typer.echo("No projects found.")
            return

        typer.echo("Projects:")
        for proj in project_list:
            count = await pipeline.doc_store.count(project=proj)
            typer.echo(f"  {proj:20s}  {count} snippets")

    _run(_projects())


# -------------------------------------------------------------------
# rename-project
# -------------------------------------------------------------------


@app.command("rename-project")
def rename_project(
    ctx: typer.Context,
    old_name: Annotated[str, typer.Argument(help="Current project name.")],
    new_name: Annotated[str, typer.Argument(help="New project name.")],
) -> None:
    """Rename a project (updates all snippets and vector metadata)."""

    async def _rename() -> None:
        pipeline = await config.build_pipeline(ctx.obj["config_path"])
        count = await pipeline.rename_project(old_name, new_name)
        typer.echo(f"✓ Renamed '{old_name}' → '{new_name}' ({count} snippets)")

    _run(_rename())


# -------------------------------------------------------------------
# stats
# -------------------------------------------------------------------


@app.command()
def stats(
    ctx: typer.Context,
    all_projects: Annotated[
        bool,
        typer.Option("--all-projects", help="Show stats across all projects."),
    ] = False,
) -> None:
    """Show knowledge base statistics."""

    async def _stats() -> None:
        pipeline = await config.build_pipeline(ctx.obj["config_path"])
        if all_projects:
            project_list = await pipeline.list_projects()
            total_snippets = 0
            for proj in project_list:
                count = await pipeline.doc_store.count(project=proj)
                total_snippets += count
                typer.echo(f"  {proj:20s}  {count} snippets")
            typer.echo(f"  {'TOTAL':20s}  {total_snippets} snippets")
        else:
            project = ctx.obj["project"] or pipeline.default_project
            doc_count = await pipeline.doc_store.count(project=project)
            typer.echo(f"Project:  {project}")
            typer.echo(f"Snippets: {doc_count}")

        vec_count = pipeline.vec_store.count()
        typer.echo(f"Chunks (all projects): {vec_count}")

    _run(_stats())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
