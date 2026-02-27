"""Evaluation data models and metric computation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Result of evaluating a single query."""

    query: str
    category: str
    relevant_ids: list[str]  # ground truth
    retrieved_ids: list[str]  # what the system returned (by id_prefix)
    retrieved_scores: list[float]
    irrelevant_ids: list[str] = field(default_factory=list)

    @property
    def precision_at_k(self) -> float:
        """Fraction of retrieved results that are relevant."""
        if not self.retrieved_ids:
            # If nothing retrieved and nothing should be: perfect
            return 1.0 if not self.relevant_ids else 0.0
        relevant_set = set(self.relevant_ids)
        hits = sum(1 for r in self.retrieved_ids if r in relevant_set)
        return hits / len(self.retrieved_ids)

    @property
    def recall(self) -> float:
        """Fraction of relevant snippets that were retrieved."""
        if not self.relevant_ids:
            # Negative query: recall is 1.0 if we retrieved nothing
            return 1.0 if not self.retrieved_ids else 0.0
        relevant_set = set(self.relevant_ids)
        hits = sum(1 for r in self.retrieved_ids if r in relevant_set)
        return hits / len(self.relevant_ids)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant result."""
        if not self.relevant_ids:
            return 1.0 if not self.retrieved_ids else 0.0
        relevant_set = set(self.relevant_ids)
        for i, r in enumerate(self.retrieved_ids):
            if r in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @property
    def has_irrelevant_intrusion(self) -> bool:
        """Did we retrieve something that's explicitly marked as wrong?"""
        if not self.irrelevant_ids:
            return False
        bad_set = set(self.irrelevant_ids)
        return any(r in bad_set for r in self.retrieved_ids)

    @property
    def negative_correct(self) -> bool:
        """For negative queries: did we correctly return nothing?"""
        return len(self.retrieved_ids) == 0


@dataclass(slots=True)
class EvalRun:
    """Aggregated results of an evaluation run across all queries."""

    run_name: str
    params: dict[str, object] = field(default_factory=dict)
    results: list[QueryResult] = field(default_factory=list)

    def _by_category(self, category: str) -> list[QueryResult]:
        return [r for r in self.results if r.category == category]

    def _mean(self, values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # --- Aggregate metrics ---

    @property
    def mean_precision(self) -> float:
        non_negative = [r for r in self.results if r.relevant_ids]
        return self._mean([r.precision_at_k for r in non_negative])

    @property
    def mean_recall(self) -> float:
        non_negative = [r for r in self.results if r.relevant_ids]
        return self._mean([r.recall for r in non_negative])

    @property
    def mean_mrr(self) -> float:
        non_negative = [r for r in self.results if r.relevant_ids]
        return self._mean([r.mrr for r in non_negative])

    @property
    def negative_accuracy(self) -> float:
        """Fraction of negative queries that correctly returned no results."""
        neg = self._by_category("negative")
        if not neg:
            return 1.0
        return self._mean([1.0 if r.negative_correct else 0.0 for r in neg])

    @property
    def specificity_accuracy(self) -> float:
        """Fraction of specificity queries with no irrelevant intrusions."""
        spec = self._by_category("specificity")
        if not spec:
            return 1.0
        return self._mean([0.0 if r.has_irrelevant_intrusion else 1.0 for r in spec])

    def summary(self) -> dict[str, float]:
        return {
            "mean_precision": round(self.mean_precision, 3),
            "mean_recall": round(self.mean_recall, 3),
            "mean_mrr": round(self.mean_mrr, 3),
            "negative_accuracy": round(self.negative_accuracy, 3),
            "specificity_accuracy": round(self.specificity_accuracy, 3),
            "n_queries": len(self.results),
        }

    def summary_by_category(self) -> dict[str, dict[str, float]]:
        categories = sorted(set(r.category for r in self.results))
        out: dict[str, dict[str, float]] = {}
        for cat in categories:
            cat_results = self._by_category(cat)
            has_relevant = [r for r in cat_results if r.relevant_ids]
            out[cat] = {
                "n": len(cat_results),
                "precision": (
                    round(self._mean([r.precision_at_k for r in has_relevant]), 3)
                    if has_relevant
                    else 0.0
                ),
                "recall": (
                    round(self._mean([r.recall for r in has_relevant]), 3)
                    if has_relevant
                    else 0.0
                ),
                "mrr": (
                    round(self._mean([r.mrr for r in has_relevant]), 3)
                    if has_relevant
                    else 0.0
                ),
            }
        return out

    def format_report(self) -> str:
        """Human-readable evaluation report."""
        lines: list[str] = []
        lines.append(f"{'═' * 60}")
        lines.append(f"  Eval Run: {self.run_name}")
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            lines.append(f"  Params: {param_str}")
        lines.append(f"{'═' * 60}")

        s = self.summary()
        lines.append(f"  Precision:     {s['mean_precision']:.3f}")
        lines.append(f"  Recall:        {s['mean_recall']:.3f}")
        lines.append(f"  MRR:           {s['mean_mrr']:.3f}")
        lines.append(f"  Neg. accuracy: {s['negative_accuracy']:.3f}")
        lines.append(f"  Specificity:   {s['specificity_accuracy']:.3f}")
        lines.append(f"{'─' * 60}")

        by_cat = self.summary_by_category()
        for cat, metrics in by_cat.items():
            lines.append(
                f"  {cat:15s}  n={metrics['n']:<3d}  "
                f"P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  "
                f"MRR={metrics['mrr']:.3f}"
            )

        lines.append(f"{'─' * 60}")

        # Flag failures
        failures = [
            r
            for r in self.results
            if (r.relevant_ids and r.recall == 0.0)
            or (not r.relevant_ids and not r.negative_correct)
            or r.has_irrelevant_intrusion
        ]
        if failures:
            lines.append("  ⚠ Failures:")
            for f in failures:
                reason = ""
                if f.relevant_ids and f.recall == 0.0:
                    reason = "zero recall"
                elif not f.relevant_ids and not f.negative_correct:
                    reason = f"false positive ({len(f.retrieved_ids)} results)"
                elif f.has_irrelevant_intrusion:
                    reason = "irrelevant intrusion"
                lines.append(f'    [{f.category}] "{f.query[:50]}" — {reason}')
        else:
            lines.append("  ✓ No failures")

        lines.append(f"{'═' * 60}")
        return "\n".join(lines)
