"""Configuration and dependency wiring.

Reads domesday.toml (or env vars) and constructs the Pipeline
with the requested backends. This is the only module that knows
about concrete implementations.

Sources (highest → lowest priority):
  1. Init kwargs
  2. Environment variables (prefix ``DOMESDAY_``, nested delimiter ``__``)
  3. domesday.toml (or the path passed to ``Config.load``)
  4. Field defaults
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pydantic
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from domesday import chunking, embedders, generators
from domesday.core import pipeline as core_pipeline
from domesday.core import protocols
from domesday.stores import chroma_store, sqlite_store

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("domesday.toml")


# ---------------------------------------------------------------------------
# Nested section models
# ---------------------------------------------------------------------------


class DocumentStoreConfig(pydantic.BaseModel):
    backend: str = "sqlite"
    path: Path | None = None


class VectorStoreConfig(pydantic.BaseModel):
    backend: str = "chroma"
    collection_name: str = "domesday"
    path: Path | None = None


class EmbedderConfig(pydantic.BaseModel):
    backend: str = "voyage"
    model: str = "voyage-3-large"


class GeneratorConfig(pydantic.BaseModel):
    backend: str = "claude"
    model: str = "claude-sonnet-4-6"


class ChunkerConfig(pydantic.BaseModel):
    max_tokens: int = 400
    overlap_tokens: int = 50


class RetrievalConfig(pydantic.BaseModel):
    min_score: float = 0.3


class RerankerConfig(pydantic.BaseModel):
    enabled: bool = False
    model: str = "claude-haiku-4-5"
    relevance_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class Config(BaseSettings):
    """Parsed configuration.

    Sources (highest → lowest priority): init kwargs → env vars → TOML file → defaults.

    Env vars use prefix ``DOMESDAY_`` and ``__`` as the nested delimiter, e.g.::

        DOMESDAY_DATA_DIR=/tmp/data
        DOMESDAY_EMBEDDER__BACKEND=local
        DOMESDAY_EMBEDDER__MODEL=all-MiniLM-L6-v2
    """

    model_config = SettingsConfigDict(
        env_prefix="DOMESDAY_",
        env_nested_delimiter="__",
        toml_file=str(DEFAULT_CONFIG_PATH),
    )

    data_dir: Path = Path("./data")
    default_project: str = "main"
    document_store: DocumentStoreConfig = DocumentStoreConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    generator: GeneratorConfig = GeneratorConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load config, optionally from a non-default TOML path."""
        if path is None:
            return cls()
        # Inject a custom TOML source for the given path
        toml_source = TomlConfigSettingsSource(cls, toml_file=path)

        class _CustomToml(cls):  # type: ignore[valid-type]
            @classmethod
            def settings_customise_sources(
                cls2,
                settings_cls: type[BaseSettings],
                init_settings: PydanticBaseSettingsSource,
                env_settings: PydanticBaseSettingsSource,
                dotenv_settings: PydanticBaseSettingsSource,
                file_secret_settings: PydanticBaseSettingsSource,
            ) -> tuple[PydanticBaseSettingsSource, ...]:
                return (init_settings, env_settings, toml_source)

        return _CustomToml()

    # ------------------------------------------------------------------
    # Convenience accessors (backward-compatible)
    # ------------------------------------------------------------------

    @property
    def min_score(self) -> float:
        return self.retrieval.min_score

    def section(self, name: str) -> dict[str, Any]:
        """Return a named section as a plain dict (for factory functions)."""
        val = getattr(self, name, None)
        if val is None:
            return {}
        if isinstance(val, pydantic.BaseModel):
            return val.model_dump()
        return {}


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
        cfg.default_project,
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
    default_project = cfg.default_project

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
