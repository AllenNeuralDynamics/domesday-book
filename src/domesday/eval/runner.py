"""Evaluation runner: load corpus, run queries, sweep parameters.

Usage:
    # Run once with current config
    python -m domesday.eval.runner

    # Sweep parameters
    python -m domesday.eval.runner --sweep

    # Interactive mode (inspect individual results)
    python -m domesday.eval.runner --interactive
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.fixtures import test_corpus as test_corpus_module

from domesday import chunking, config
from domesday.core import models, pipeline
from domesday.eval import llm_judge
from domesday.eval import models as eval_models

logger = logging.getLogger(__name__)


@dataclass
class EvalRunner:
    """Runs evaluation queries against a loaded pipeline and scores results."""

    pipeline: pipeline.Pipeline
    snippet_id_map: dict[str, str]  # id_prefix → full snippet UUID

    @classmethod
    async def create(
        cls,
        config_path: Path | None = None,
        snippets: Sequence[test_corpus_module.TestSnippet] | None = None,
    ) -> EvalRunner:
        """Build pipeline and ingest test corpus."""
        from tests.fixtures import test_corpus

        pl = await config.build_pipeline(config_path)
        snippets = snippets or test_corpus.SNIPPETS
        id_map: dict[str, str] = {}

        # Check if already loaded (idempotent for repeated runs)
        existing_count = await pl.doc_store.count()
        if existing_count >= len(snippets):
            logger.info(
                "Corpus appears loaded (%d snippets), resolving IDs...", existing_count
            )
            all_active = await pl.doc_store.get_all_active()
            # Rebuild map from source_file field which stores id_prefix
            for s in all_active:
                if s.source_file and s.source_file.startswith("test:"):
                    prefix = s.source_file.removeprefix("test:")
                    id_map[prefix] = s.id
        else:
            logger.info("Loading %d test snippets...", len(snippets))
            for ts in snippets:
                snippet = await pl.add_snippet(
                    ts.text,
                    author=ts.author,
                    project=ts.project,
                    tags=ts.tags,
                    source_file=f"test:{ts.id_prefix}",
                )
                id_map[ts.id_prefix] = snippet.id

        logger.info("Ready: %d snippets mapped", len(id_map))
        return cls(pipeline=pl, snippet_id_map=id_map)

    def _resolve_prefix(self, results: list[models.SearchResult]) -> list[str]:
        """Map retrieved snippet UUIDs back to test id_prefixes."""
        uuid_to_prefix = {v: k for k, v in self.snippet_id_map.items()}
        return [uuid_to_prefix.get(r.snippet.id, r.snippet.id[:8]) for r in results]

    async def run_query(
        self,
        eq: test_corpus_module.EvalQuery,
        *,
        k: int = 10,
        min_score: float = 0.0,
    ) -> eval_models.QueryResult:
        """Evaluate a single query against ground truth."""
        results = await self.pipeline.search(eq.query, k=k, min_score=min_score)

        retrieved_prefixes = self._resolve_prefix(results)
        scores = [r.score for r in results]

        return eval_models.QueryResult(
            query=eq.query,
            category=eq.category,
            relevant_ids=eq.relevant_ids,
            retrieved_ids=retrieved_prefixes,
            retrieved_scores=scores,
            irrelevant_ids=eq.irrelevant_ids,
        )

    async def run_all(
        self,
        queries: Sequence[test_corpus_module.EvalQuery] | None = None,
        *,
        k: int = 10,
        min_score: float = 0.0,
        run_name: str = "default",
        params: dict[str, object] | None = None,
        judge: bool = False,
    ) -> eval_models.EvalRun:
        """Run all eval queries and return aggregated metrics.

        Args:
            judge: If True, also run the full RAG pipeline and have an LLM
                   judge score each generated answer. Slower and costs API
                   calls, but measures generation quality — not just retrieval.
        """
        from tests.fixtures import test_corpus

        queries = queries or test_corpus.EVAL_QUERIES
        eval_run = eval_models.EvalRun(run_name=run_name, params=params or {})

        for eq in queries:
            result = await self.run_query(eq, k=k, min_score=min_score)
            eval_run.results.append(result)

        if judge:
            judge_instance = llm_judge.LLMJudge()
            logger.info("Running LLM judge on %d generated answers...", len(queries))

            # Run full RAG for each query and judge the response
            items: list[tuple[str, models.RAGResponse, bool]] = []
            for eq in queries:
                rag_response = await self.pipeline.ask(
                    eq.query, k=k, min_score=min_score
                )
                is_negative = eq.category == "negative"
                items.append((eq.query, rag_response, is_negative))

            judgments = await judge_instance.judge_batch(items)

            # Attach judgments to the eval run
            eval_run.judgments = judgments  # type: ignore[attr-defined]

            # Print judgment summary
            avg_composite = sum(j.composite for j in judgments) / len(judgments)
            avg_faith = sum(j.faithfulness for j in judgments) / len(judgments)
            avg_abstain = sum(j.abstention for j in judgments) / len(judgments)
            logger.info(
                "Judge scores — composite=%.3f faithfulness=%.3f abstention=%.3f",
                avg_composite,
                avg_faith,
                avg_abstain,
            )

        return eval_run


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Defines the parameter grid to sweep."""

    min_scores: list[float]
    k_values: list[int]
    chunk_max_tokens: list[int]
    chunk_overlaps: list[int]

    @classmethod
    def default(cls) -> SweepConfig:
        return cls(
            min_scores=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
            k_values=[5, 10, 15],
            chunk_max_tokens=[200, 400, 600],
            chunk_overlaps=[0, 50, 100],
        )

    @classmethod
    def quick(cls) -> SweepConfig:
        """Smaller grid for fast iteration."""
        return cls(
            min_scores=[0.2, 0.3, 0.4],
            k_values=[5, 10],
            chunk_max_tokens=[400],
            chunk_overlaps=[50],
        )


async def run_sweep(
    sweep_config: SweepConfig | None = None,
    config_path: Path | None = None,
    output_path: Path | None = None,
) -> list[eval_models.EvalRun]:
    """Sweep parameter combinations and report results.

    Note: chunk_max_tokens and chunk_overlaps require re-ingesting the corpus
    (different chunking → different embeddings). min_score and k are cheap to
    sweep because they only affect the retrieval filter.
    """
    from tests.fixtures import test_corpus

    sweep = sweep_config or SweepConfig.quick()
    all_runs: list[eval_models.EvalRun] = []

    for max_tokens, overlap in itertools.product(
        sweep.chunk_max_tokens, sweep.chunk_overlaps
    ):
        if overlap >= max_tokens:
            continue  # nonsensical combination

        logger.info("=== Chunk config: max=%d overlap=%d ===", max_tokens, overlap)

        # Build pipeline with these chunk settings
        # For a real sweep, we'd rebuild the pipeline with different chunker.
        # For now, we re-ingest with each chunk config using a separate data dir.
        run_data_dir = Path(f"./data/eval_sweep/chunk{max_tokens}_overlap{overlap}")
        run_data_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary config override
        cfg = config.Config.load(config_path)
        cfg.raw["data_dir"] = str(run_data_dir)  # type: ignore[union-attr]
        cfg.raw["chunker"] = {"max_tokens": max_tokens, "overlap_tokens": overlap}  # type: ignore[union-attr]

        # Build pipeline manually with overridden chunker
        pl = await config.build_pipeline(config_path)
        # Override chunker
        pl.chunker = chunking.SimpleChunker(
            max_tokens=max_tokens, overlap_tokens=overlap
        )

        runner = await EvalRunner.create(config_path, test_corpus.SNIPPETS)
        # Override the pipeline's chunker and min_score for this run
        runner.pipeline.chunker = chunking.SimpleChunker(
            max_tokens=max_tokens, overlap_tokens=overlap
        )

        for k_val, min_score in itertools.product(sweep.k_values, sweep.min_scores):
            run_name = f"chunk{max_tokens}_overlap{overlap}_k{k_val}_min{min_score}"
            params: dict[str, object] = {
                "chunk_max_tokens": max_tokens,
                "chunk_overlap": overlap,
                "k": k_val,
                "min_score": min_score,
            }

            eval_run = await runner.run_all(
                k=k_val,
                min_score=min_score,
                run_name=run_name,
                params=params,
            )
            all_runs.append(eval_run)
            logger.info(
                "%s → P=%.3f R=%.3f MRR=%.3f Neg=%.3f",
                run_name,
                eval_run.mean_precision,
                eval_run.mean_recall,
                eval_run.mean_mrr,
                eval_run.negative_accuracy,
            )

    # Sort by composite score (weighted recall + negative accuracy)
    all_runs.sort(
        key=lambda r: (
            r.mean_recall * 0.5 + r.negative_accuracy * 0.3 + r.mean_mrr * 0.2
        ),
        reverse=True,
    )

    # Save results
    if output_path is None:
        output_path = Path("./data/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [
        {
            "run_name": run.run_name,
            "params": run.params,
            "summary": run.summary(),
            "by_category": run.summary_by_category(),
        }
        for run in all_runs
    ]
    output_path.write_text(json.dumps(serializable, indent=2, default=str))
    logger.info("Results saved to %s", output_path)

    return all_runs


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


async def interactive(config_path: Path | None = None) -> None:
    """Interactive REPL for inspecting retrieval results."""
    from tests.fixtures import test_corpus

    runner = await EvalRunner.create(config_path)

    print("\n domesday eval — interactive mode")
    print("Commands: query <text>, eval <index>, judge <index>, sweep, report, quit\n")

    while True:
        try:
            raw = input("domesday> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.lower() == "report":
            eval_run = await runner.run_all(run_name="interactive-full")
            print(eval_run.format_report())
            continue

        if raw.lower().startswith("eval"):
            parts = raw.split(maxsplit=1)
            idx = int(parts[1]) if len(parts) > 1 else 0
            if idx >= len(test_corpus.EVAL_QUERIES):
                print(f"Index out of range (0–{len(test_corpus.EVAL_QUERIES) - 1})")
                continue
            eq = test_corpus.EVAL_QUERIES[idx]
            result = await runner.run_query(eq, k=10)
            print(f"\n  Query:    {eq.query}")
            print(f"  Category: {eq.category}")
            print(f"  Expected: {eq.relevant_ids}")
            print(f"  Got:      {result.retrieved_ids}")
            print(f"  Scores:   {[f'{s:.3f}' for s in result.retrieved_scores]}")
            print(
                f"  P={result.precision_at_k:.3f} R={result.recall:.3f} MRR={result.mrr:.3f}"
            )
            if result.has_irrelevant_intrusion:
                print("  ⚠ IRRELEVANT INTRUSION")
            print()
            continue

        if raw.lower().startswith("judge"):
            parts = raw.split(maxsplit=1)
            idx = int(parts[1]) if len(parts) > 1 else 0
            if idx >= len(test_corpus.EVAL_QUERIES):
                print(f"Index out of range (0–{len(test_corpus.EVAL_QUERIES) - 1})")
                continue
            eq = test_corpus.EVAL_QUERIES[idx]
            print(f"\n  Generating answer for: {eq.query}")
            rag_response = await runner.pipeline.ask(eq.query, k=10)
            print(f"\n  Answer:\n  {rag_response.answer[:300]}...")
            print("\n  Judging...")
            judge = llm_judge.LLMJudge()
            scores = await judge.judge(
                eq.query, rag_response, is_negative=(eq.category == "negative")
            )
            print(f"  Faithfulness:      {scores.faithfulness:.2f}")
            print(f"  Citation accuracy: {scores.citation_accuracy:.2f}")
            print(f"  Abstention:        {scores.abstention:.2f}")
            print(f"  Synthesis:         {scores.synthesis:.2f}")
            print(f"  Relevance:         {scores.relevance:.2f}")
            print(f"  Composite:         {scores.composite:.2f}")
            print(f"  Explanation:       {scores.explanation}")
            print()
            continue

        # Default: treat as a free query
        results = await runner.pipeline.search(raw, k=5)
        if not results:
            print("  (no results above threshold)\n")
        else:
            for i, r in enumerate(results, 1):
                prefix = "?"
                uuid_to_prefix = {v: k for k, v in runner.snippet_id_map.items()}
                prefix = uuid_to_prefix.get(r.snippet.id, r.snippet.id[:8])
                print(
                    f"  {i}. [{prefix}] score={r.score:.3f} "
                    f"({r.snippet.author}) {r.snippet.raw_text[:80]}…"
                )
            print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="domesday evaluation")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    parser.add_argument(
        "--judge",
        "-j",
        action="store_true",
        help="Also judge generation quality with LLM",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Use quick sweep grid")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.interactive:
        await interactive(args.config)
    elif args.sweep:
        sweep_cfg = SweepConfig.quick() if args.quick else SweepConfig.default()
        runs = await run_sweep(sweep_cfg, args.config)
        print("\nTop 5 configurations:")
        for run in runs[:5]:
            print(run.format_report())
    else:
        # Default: single run with current config, print report
        runner = await EvalRunner.create(args.config)
        eval_run = await runner.run_all(run_name="default", judge=args.judge)
        print(eval_run.format_report())


if __name__ == "__main__":
    asyncio.run(main())
