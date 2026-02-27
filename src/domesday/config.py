"""Configuration and dependency wiring.

Reads domesday.toml (or env vars) and constructs the Pipeline
with the requested backends. This is the only module that knows
about concrete implementations.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domesday import chunking, embedders, generators
from domesday.core import pipeline as core_pipeline
from domesday.core import protocols
from domesday.stores import chroma_store, sqlite_store

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("domesday.toml")

DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": "./data",
    "default_project": "main",
    "document_store": {
        "backend": "sqlite",
    },
    "vector_store": {
        "backend": "chroma",
        "collection_name": "domesday",
    },
    "embedder": {
        "backend": "voyage",
        "model": "voyage-3-large",
    },
    "generator": {
        "backend": "claude",
        "model": "claude-sonnet-4-6",
    },
    "chunker": {
        "max_tokens": 400,
        "overlap_tokens": 50,
    },
    "retrieval": {
        "min_score": 0.3,
    },
    "reranker": {
        "enabled": False,
        "model": "claude-haiku-4-5",
        "relevance_threshold": 0.5,
    },
}


@dataclass
class Config:
    """Parsed configuration with typed accessors."""

    raw: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG))

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load from TOML file, falling back to defaults."""
        if path is None:
            path = DEFAULT_CONFIG_PATH

        raw = dict(DEFAULT_CONFIG)

        if path.exists():
            with open(path, "rb") as f:
                user_cfg = tomllib.load(f)
            # Shallow merge per section
            for key, value in user_cfg.items():
                if isinstance(value, dict) and isinstance(raw.get(key), dict):
                    raw[key] = {**raw[key], **value}
                else:
                    raw[key] = value
            logger.info("Loaded config from %s", path)
        else:
            logger.debug("No config file at %s, using defaults", path)

        # Env var overrides (flat, for CI/deployment)
        env_map: dict[str, str | tuple[str, str]] = {
            "DOMESDAY_DATA_DIR": "data_dir",
            "DOMESDAY_DEFAULT_PROJECT": "default_project",
            "DOMESDAY_EMBEDDER_BACKEND": ("embedder", "backend"),
            "DOMESDAY_EMBEDDER_MODEL": ("embedder", "model"),
            "DOMESDAY_GENERATOR_MODEL": ("generator", "model"),
        }
        for env_key, config_path in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                if isinstance(config_path, tuple):
                    raw[config_path[0]][config_path[1]] = val
                else:
                    raw[config_path] = val
                logger.debug("Env override: %s = %s", env_key, val)

        return cls(raw=raw)

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["data_dir"])

    @property
    def min_score(self) -> float:
        return float(self.raw["retrieval"]["min_score"])

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))


# -------------------------------------------------------------------
# Factory functions
# -------------------------------------------------------------------


def _build_doc_store(cfg: Config) -> sqlite_store.SQLiteDocumentStore:
    section = cfg.section("document_store")
    backend = section["backend"]

    if backend == "sqlite":
        path = section.get("path", cfg.data_dir / "domesday.db")
        logger.debug("Document store: SQLite at %s", path)
        return sqlite_store.SQLiteDocumentStore(path=path)

    raise ValueError(f"Unknown document_store backend: {backend}")


def _build_vec_store(cfg: Config) -> chroma_store.ChromaVectorStore:
    section = cfg.section("vector_store")
    backend = section["backend"]

    if backend == "chroma":
        path = section.get("path", cfg.data_dir / "chroma")
        collection = section["collection_name"]
        # Resolve embedding model name to store in collection metadata
        emb_section = cfg.section("embedder")
        emb_model = str(emb_section.get("model", "")) or str(
            emb_section.get("backend", "")
        )
        logger.debug(
            "Vector store: Chroma at %s (collection=%s, embedding_model=%s)",
            path,
            collection,
            emb_model,
        )
        return chroma_store.ChromaVectorStore(
            path=path,
            collection_name=collection,
            embedding_model=emb_model,
        )

    raise ValueError(f"Unknown vector_store backend: {backend}")


def _build_embedder(cfg: Config) -> protocols.Embedder:
    section = cfg.section("embedder")
    backend = section["backend"]
    model: str = section.get("model", "")

    if backend == "voyage":
        logger.debug("Embedder: Voyage (model=%s)", model)
        return (
            embedders.VoyageEmbedder(model=model)
            if model
            else embedders.VoyageEmbedder()
        )
    if backend == "openai":
        logger.debug("Embedder: OpenAI (model=%s)", model or "<class default>")
        return (
            embedders.OpenAIEmbedder(model=model)
            if model
            else embedders.OpenAIEmbedder()
        )
    if backend == "local":
        logger.debug(
            "Embedder: local sentence-transformers (model=%s)",
            model or "<class default>",
        )
        return (
            embedders.SentenceTransformerEmbedder(model=model)
            if model
            else embedders.SentenceTransformerEmbedder()
        )

    raise ValueError(f"Unknown embedder backend: {backend}")


def _build_generator(cfg: Config) -> protocols.Generator:
    section = cfg.section("generator")
    backend = section["backend"]
    model: str = section.get("model", "")

    if backend == "claude":
        logger.debug("Generator: Claude (model=%s)", model)
        return (
            generators.ClaudeGenerator(model=model)
            if model
            else generators.ClaudeGenerator()
        )

    raise ValueError(f"Unknown generator backend: {backend}")


def _build_chunker(cfg: Config) -> protocols.Chunker:
    section = cfg.section("chunker")
    max_tokens = int(section["max_tokens"])
    overlap = int(section["overlap_tokens"])
    logger.debug("Chunker: max_tokens=%d, overlap=%d", max_tokens, overlap)
    return chunking.SimpleChunker(max_tokens=max_tokens, overlap_tokens=overlap)


def _build_reranker(cfg: Config) -> protocols.Reranker | None:
    section = cfg.section("reranker")
    if not section["enabled"]:
        logger.debug("Reranker: disabled")
        return None

    from domesday.eval import llm_judge

    model = str(section["model"])
    threshold = float(section["relevance_threshold"])
    logger.debug("Reranker: enabled (model=%s, threshold=%.2f)", model, threshold)
    return llm_judge.LLMReranker(model=model, relevance_threshold=threshold)


async def build_pipeline(config_path: Path | None = None) -> core_pipeline.Pipeline:
    """Build a fully wired Pipeline from config.

    This is the main entry point for CLI, MCP server, and web app.
    """
    cfg = Config.load(config_path)
    logger.info(
        "Building pipeline (data_dir=%s, project=%s)",
        cfg.data_dir,
        cfg.raw["default_project"],
    )

    doc_store = _build_doc_store(cfg)
    await doc_store.initialize()

    vec_store = _build_vec_store(cfg)
    vec_store.initialize()

    embedder = _build_embedder(cfg)
    generator = _build_generator(cfg)
    chunker = _build_chunker(cfg)
    reranker = _build_reranker(cfg)

    retrieval_cfg = cfg.section("retrieval")
    min_score = float(retrieval_cfg["min_score"])
    default_project = str(cfg.raw["default_project"])

    pipeline = core_pipeline.Pipeline(
        doc_store=doc_store,
        vec_store=vec_store,
        embedder=embedder,
        generator=generator,
        chunker=chunker,
        default_project=default_project,
        reranker=reranker,
    )

    logger.info(
        "Pipeline ready (min_score=%.2f, reranker=%s)",
        min_score,
        "enabled" if reranker else "disabled",
    )
    return pipeline
