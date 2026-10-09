"""
nanoevals.types
~~~~~~~~~~~~~~~
Typed data structures for evaluation outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Output from a single evaluator on a single input."""

    key: str
    """Evaluator / feedback key, e.g. 'faithfulness'."""

    score: float | bool
    """Normalised score.  bool → binary pass/fail; float → 0.0–1.0."""

    comment: str = ""
    """Human-readable reasoning or explanation from the evaluator."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional extra data (raw LLM response, matched tokens, etc.)."""

    def passed(self, threshold: float = 0.5) -> bool:
        """Return True if the score meets *threshold*."""
        if isinstance(self.score, bool):
            return self.score
        return self.score >= threshold

    def __repr__(self) -> str:
        return f"EvalResult(key={self.key!r}, score={self.score!r})"


@dataclass
class SampleResult:
    """All evaluator results for a single dataset sample."""

    sample_index: int
    results: list[EvalResult] = field(default_factory=list)

    def get(self, key: str) -> EvalResult | None:
        """Look up a result by evaluator key."""
        return next((r for r in self.results if r.key == key), None)

    def scores(self) -> dict[str, float | bool]:
        return {r.key: r.score for r in self.results}


@dataclass
class BatchResult:
    """Results for an entire evaluation run over a dataset."""

    samples: list[SampleResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, float]:
        """Return mean score per evaluator key (bools cast to 0/1)."""
        totals: dict[str, list[float]] = {}
        for sample in self.samples:
            for result in sample.results:
                score = float(result.score)
                totals.setdefault(result.key, []).append(score)
        return {k: sum(v) / len(v) for k, v in totals.items()}

    def pass_rates(self, threshold: float = 0.5) -> dict[str, float]:
        """Return fraction of samples that *passed* per key."""
        counts: dict[str, list[bool]] = {}
        for sample in self.samples:
            for result in sample.results:
                counts.setdefault(result.key, []).append(result.passed(threshold))
        return {k: sum(v) / len(v) for k, v in counts.items()}

    def to_dict(self) -> list[dict[str, Any]]:
        """Flat, JSON-serialisable export."""
        rows = []
        for sample in self.samples:
            row: dict[str, Any] = {"sample_index": sample.sample_index}
            for result in sample.results:
                row[f"{result.key}_score"] = result.score
                row[f"{result.key}_comment"] = result.comment
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        return f"BatchResult(n_samples={len(self.samples)}, summary={self.summary()})"
