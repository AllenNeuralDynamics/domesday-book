"""MCP server exposing domesday as tools for any MCP-compatible client.

Run directly:
    python -m domesday.mcp_server

Or configure in Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "domesday": {
                "command": "python",
                "args": ["-m", "domesday.mcp_server"],
                "env": {
                    "DOMESDAY_DATA_DIR": "/path/to/your/data",
                    "DOMESDAY_DEFAULT_PROJECT": "my-project"
                }
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import mcp.server
import mcp.server.stdio
import mcp.types

from domesday import config
from domesday.core import models, pipeline

logger = logging.getLogger(__name__)

server = mcp.server.Server("domesday")

_pipeline: pipeline.Pipeline | None = None


async def _get_pipeline() -> pipeline.Pipeline:
    global _pipeline
    if _pipeline is None:
        config_path = Path("domesday.toml") if Path("domesday.toml").exists() else None
        logger.info("Initializing MCP server pipeline")
        _pipeline = await config.build_pipeline(config_path)
        logger.info("MCP server pipeline ready")
    return _pipeline


# Shared schema fragment for the project parameter
_PROJECT_SCHEMA = {
    "type": "string",
    "description": (
        "Project to search/add within. Defaults to the configured default project. "
        "Use 'all' to search across all projects."
    ),
    "default": "all",
}


# -------------------------------------------------------------------
# Tool definitions
# -------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[mcp.types.Tool]:
    return [
        mcp.types.Tool(
            name="search_knowledge",
            description=(
                "Semantic search over the team knowledge base. Returns the most "
                "relevant snippets for a given query. Use this to find specific "
                "facts, caveats, or context about the project."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "project": _PROJECT_SCHEMA,
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: filter results to snippets with these tags.",
                    },
                },
                "required": ["query"],
            },
        ),
        mcp.types.Tool(
            name="add_snippet",
            description=(
                "Add a new knowledge snippet to the team knowledge base. "
                "Use this to capture caveats, conclusions, notes, or any "
                "project-relevant information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The snippet content (prose, code, table, etc.).",
                    },
                    "project": _PROJECT_SCHEMA,
                    "author": {
                        "type": "string",
                        "description": "Who is adding this (default: 'mcp-client').",
                        "default": "mcp-client",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorization.",
                    },
                },
                "required": ["text"],
            },
        ),
        mcp.types.Tool(
            name="get_snippet",
            description="Retrieve a specific snippet by its ID (full or 8-char short ID).",
            inputSchema={
                "type": "object",
                "properties": {
                    "snippet_id": {
                        "type": "string",
                        "description": "Full UUID or 8-character short ID of the snippet.",
                    },
                },
                "required": ["snippet_id"],
            },
        ),
        mcp.types.Tool(
            name="list_recent",
            description="List the most recently added snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of snippets to return (default 10).",
                        "default": 10,
                    },
                    "project": _PROJECT_SCHEMA,
                    "author": {
                        "type": "string",
                        "description": "Optional: filter by author.",
                    },
                },
            },
        ),
        mcp.types.Tool(
            name="list_projects",
            description="List all projects in the knowledge base with snippet counts.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        mcp.types.Tool(
            name="rename_project",
            description="Rename a project. Updates all snippets and vector metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "Current project name.",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New project name.",
                    },
                },
                "required": ["old_name", "new_name"],
            },
        ),
        mcp.types.Tool(
            name="ask",
            description=(
                "Ask a question that will be answered using the knowledge base. "
                "Retrieves relevant snippets and generates a grounded answer "
                "with citations. Use this for complex questions that need synthesis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer from the knowledge base.",
                    },
                    "project": _PROJECT_SCHEMA,
                    "n_context": {
                        "type": "integer",
                        "description": "Number of context snippets to retrieve (default 8).",
                        "default": 8,
                    },
                },
                "required": ["question"],
            },
        ),
    ]


# -------------------------------------------------------------------
# Tool handlers
# -------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp.types.TextContent]:
    logger.info(
        "MCP tool call: %s(%s)", name, {k: str(v)[:80] for k, v in arguments.items()}
    )
    pipeline = await _get_pipeline()

    if name == "search_knowledge":
        project = arguments.get("project")
        cfg = config.Config.load(None)
        results = await pipeline.search(
            arguments["query"],
            project=project,
            k=arguments.get("n_results", 5),
            tags=arguments.get("tags"),
            min_score=cfg.retrieval.min_score,
        )
        if not results:
            return [
                mcp.types.TextContent(type="text", text="No relevant snippets found.")
            ]

        lines: list[str] = []
        for r in results:
            sid = r.snippet.id[:8]
            score = f"{r.score:.3f}"
            proj = r.snippet.project
            tags_str = f" tags=[{', '.join(r.snippet.tags)}]" if r.snippet.tags else ""
            lines.append(
                f"[snippet-{sid}] (score={score}, project={proj}, {r.snippet.author}, "
                f"{r.snippet.created_at.strftime('%Y-%m-%d')}{tags_str})\n"
                f"{r.snippet.raw_text}\n"
            )
        return [mcp.types.TextContent(type="text", text="\n---\n".join(lines))]

    if name == "add_snippet":
        snippet = await pipeline.add_snippet(
            arguments["text"],
            project=arguments["project"],
            author=arguments.get("author", "mcp-client"),
            tags=arguments.get("tags", []),
        )
        return [
            mcp.types.TextContent(
                type="text",
                text=(
                    f"Added snippet {snippet.id[:8]} to project '{snippet.project}': "
                    f"{snippet.raw_text[:100]}…"
                ),
            )
        ]

    if name == "get_snippet":
        snippet_id = arguments["snippet_id"]
        found: models.Snippet | None = await pipeline.doc_store.get(snippet_id)
        if found is None:
            all_active = await pipeline.doc_store.get_all_active()
            matches = [s for s in all_active if s.id.startswith(snippet_id)]
            found = matches[0] if len(matches) == 1 else None

        if found is None:
            return [
                mcp.types.TextContent(
                    type="text", text=f"Snippet '{snippet_id}' not found."
                )
            ]

        return [
            mcp.types.TextContent(
                type="text",
                text=json.dumps(found.to_dict(), indent=2, default=str),
            )
        ]

    if name == "list_recent":
        project = arguments.get("project")
        snippets = await pipeline.doc_store.list_recent(
            n=arguments.get("n", 10),
            project=project if project != "all" else None,
            author=arguments.get("author"),
        )
        if not snippets:
            return [mcp.types.TextContent(type="text", text="No snippets found.")]

        lines = []
        for s in snippets:
            sid = s.id[:8]
            date = s.created_at.strftime("%Y-%m-%d %H:%M")
            preview = s.raw_text[:120].replace("\n", " ")
            lines.append(f"[{sid}] {s.project} {date} ({s.author}) {preview}…")
        return [mcp.types.TextContent(type="text", text="\n".join(lines))]

    if name == "list_projects":
        project_list = await pipeline.list_projects()
        if not project_list:
            return [mcp.types.TextContent(type="text", text="No projects found.")]

        lines = []
        for proj in project_list:
            count = await pipeline.doc_store.count(project=proj)
            lines.append(f"  {proj}: {count} snippets")
        return [
            mcp.types.TextContent(type="text", text="Projects:\n" + "\n".join(lines))
        ]

    if name == "rename_project":
        count = await pipeline.rename_project(
            arguments["old_name"], arguments["new_name"]
        )
        return [
            mcp.types.TextContent(
                type="text",
                text=f"Renamed '{arguments['old_name']}' → '{arguments['new_name']}' ({count} snippets)",
            )
        ]

    if name == "ask":
        project = arguments.get("project")
        cfg = config.Config.load(None)
        response = await pipeline.ask(
            arguments["question"],
            project=project,
            k=arguments.get("n_context", 8),
            min_score=cfg.retrieval.min_score,
        )
        source_ids = ", ".join(
            f"snippet-{r.snippet.id[:8]} ({r.snippet.project})"
            for r in response.sources
        )
        return [
            mcp.types.TextContent(
                type="text",
                text=f"{response.answer}\n\n[Sources: {source_ids}]",
            )
        ]

    return [mcp.types.TextContent(type="text", text=f"Unknown tool: {name}")]


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
