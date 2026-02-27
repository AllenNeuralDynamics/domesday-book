"""Tests for domesday.config — Config loading, env overrides, and factory functions."""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest
import logging

from domesday.config import (
    Config,
    _build_chunker,
    _build_doc_store,
    _normalize_vec_store_collection_name,
)


# ---------------------------------------------------------------------------
# Config.load() — defaults
# ---------------------------------------------------------------------------


def test_config_loads_with_defaults() -> None:
    """Config() should produce valid defaults for all sections."""
    cfg = Config()
    assert cfg.default_project == "main"
    assert cfg.document_store.backend == "sqlite"
    assert cfg.embedder.backend == "voyage"
    assert cfg.generator.backend == "claude"
    assert cfg.chunker.max_tokens == 400
    assert cfg.reranker.enabled is False


# ---------------------------------------------------------------------------
# Config.load() — no config file
# ---------------------------------------------------------------------------


def test_config_load_nonexistent_path(caplog) -> None:
    caplog.set_level(logging.WARNING)
    cfg = Config.load(Path("/nonexistent/path/domesday.toml"))
    # this should log a warning, but not raise an exception:
    assert any(
        rec.levelname == "WARNING" and "Config file" in rec.message
        for rec in caplog.records
    )
    # And it should still return a Config instance with defaults and env vars:
    assert isinstance(cfg, Config)


# ---------------------------------------------------------------------------
# load() — TOML file merging
# ---------------------------------------------------------------------------


def test_config_load_from_toml_overrides_section_keys() -> None:
    toml_content = textwrap.dedent(
        """\
        [embedder]
        backend = "local"
        model = "all-MiniLM-L6-v2"
    """
    )
    with tempfile.NamedTemporaryFile(
        suffix=".toml", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(toml_content)
        tmp_path = Path(f.name)
    try:
        cfg = Config.load(tmp_path)
        assert cfg.embedder.backend == "local"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_config_load_from_toml_preserves_unset_sections() -> None:
    """If TOML only overrides one section, other sections are still present."""
    toml_content = textwrap.dedent(
        """\
        [embedder]
        backend = "local"
    """
    )
    with tempfile.NamedTemporaryFile(
        suffix=".toml", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(toml_content)
        tmp_path = Path(f.name)
    try:
        cfg = Config.load(tmp_path)
        assert hasattr(cfg.document_store, "backend")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_config_load_top_level_scalar_override() -> None:
    toml_content = 'default_project = "lab-notes"\n'
    with tempfile.NamedTemporaryFile(
        suffix=".toml", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(toml_content)
        tmp_path = Path(f.name)
    try:
        cfg = Config.load(tmp_path)
        assert cfg.default_project == "lab-notes"
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load() — environment variable overrides
# ---------------------------------------------------------------------------


def test_config_env_override_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOMESDAY_DATA_DIR", "/tmp/custom_data")
    cfg = Config.load(Path("/nonexistent.toml"))
    assert cfg.data_dir == Path("/tmp/custom_data")


def test_config_env_override_default_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOMESDAY_DEFAULT_PROJECT", "env-project")
    cfg = Config.load(Path("/nonexistent.toml"))
    assert cfg.default_project == "env-project"


def test_config_env_override_embedder_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOMESDAY_EMBEDDER__BACKEND", "openai")
    cfg = Config.load(Path("/nonexistent.toml"))
    assert cfg.embedder.backend == "openai"


def test_config_env_override_embedder_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOMESDAY_EMBEDDER__MODEL", "text-embedding-3-large")
    cfg = Config.load(Path("/nonexistent.toml"))
    assert cfg.embedder.model == "text-embedding-3-large"


def test_config_env_overrides_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars take precedence over TOML values."""
    toml_content = textwrap.dedent("""\
        default_project = "from-toml"
    """)
    with tempfile.NamedTemporaryFile(
        suffix=".toml", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(toml_content)
        tmp_path = Path(f.name)
    try:
        monkeypatch.setenv("DOMESDAY_DEFAULT_PROJECT", "from-env")
        cfg = Config.load(tmp_path)
        assert cfg.default_project == "from-env"
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_build_chunker_uses_config_values() -> None:
    from domesday.chunking import SimpleChunker

    cfg = Config()
    cfg.chunker.max_tokens = 200
    cfg.chunker.overlap_tokens = 25
    chunker = _build_chunker(cfg)
    assert isinstance(chunker, SimpleChunker)
    assert chunker.max_tokens == 200
    assert chunker.overlap_tokens == 25


def test_build_doc_store_creates_sqlite(tmp_path: Path) -> None:
    cfg = Config()
    cfg.document_store.path = tmp_path / "test.db"
    store = _build_doc_store(cfg)
    from domesday.stores.sqlite_store import SQLiteDocumentStore
    assert isinstance(store, SQLiteDocumentStore)


def test_build_doc_store_unknown_backend_raises() -> None:
    cfg = Config()
    cfg.document_store.backend = "postgres"
    with pytest.raises(ValueError, match="Unknown document_store backend"):
        _build_doc_store(cfg)


def test_normalize_vec_store_collection_name() -> None:
    assert _normalize_vec_store_collection_name("voyage-4-large") == "voyage-4"
    assert _normalize_vec_store_collection_name("voyage-4-lite") == "voyage-4"
    assert _normalize_vec_store_collection_name("voyage-3-large") == "voyage-3-large"
    assert _normalize_vec_store_collection_name("text-embedding-3-large") == "text-embedding-3-large"
